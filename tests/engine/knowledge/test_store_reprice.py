"""Tests for forecast repricing store methods."""

import pytest
from datetime import date, timedelta

from nexus.engine.knowledge.store import KnowledgeStore
from nexus.engine.projection.models import ForecastQuestion, ForecastResolution, ForecastRun


@pytest.fixture
async def store(tmp_path):
    s = KnowledgeStore(tmp_path / "test.db")
    await s.initialize()
    yield s
    await s.close()


async def _seed_forecast(store, *, topic_slug="ai", engine="structural",
                         probability=0.65, resolution_days=14,
                         target_variable="kg_native", external_ref=None):
    """Insert a forecast run with one question, return the question id."""
    run_date = date.today()
    run = ForecastRun(
        topic_slug=topic_slug,
        topic_name=topic_slug.title(),
        engine=engine,
        generated_for=run_date,
        summary="Test forecast",
        questions=[
            ForecastQuestion(
                question="Will X happen?",
                forecast_type="binary",
                target_variable=target_variable,
                probability=probability,
                resolution_criteria="observable outcome",
                resolution_date=run_date + timedelta(days=resolution_days),
                horizon_days=resolution_days,
                signpost="Key indicator to watch",
                external_ref=external_ref,
            ),
        ],
    )
    await store.save_forecast_run(run)
    return run.questions[0].question_id


@pytest.mark.asyncio
async def test_get_open_forecasts_returns_unresolved(store):
    """get_open_forecasts should return forecasts with pending resolutions."""
    qid = await _seed_forecast(store, topic_slug="ai")
    open_forecasts = await store.get_open_forecasts()
    assert len(open_forecasts) >= 1
    ids = [f["forecast_question_id"] for f in open_forecasts]
    assert qid in ids


@pytest.mark.asyncio
async def test_get_open_forecasts_excludes_resolved(store):
    """Resolved forecasts should not appear in open forecasts."""
    qid = await _seed_forecast(store)
    # Resolve it
    resolution = ForecastResolution(
        forecast_question_id=qid,
        outcome_status="resolved",
        resolved_bool=True,
    )
    await store.set_forecast_resolution(resolution)
    open_forecasts = await store.get_open_forecasts()
    ids = [f["forecast_question_id"] for f in open_forecasts]
    assert qid not in ids


@pytest.mark.asyncio
async def test_get_open_forecasts_filter_by_topic(store):
    """get_open_forecasts(topic_slug=...) should filter by topic."""
    await _seed_forecast(store, topic_slug="ai")
    await _seed_forecast(store, topic_slug="energy")
    ai_forecasts = await store.get_open_forecasts(topic_slug="ai")
    assert all(f["topic_slug"] == "ai" for f in ai_forecasts)
    assert len(ai_forecasts) >= 1


@pytest.mark.asyncio
async def test_update_forecast_probability(store):
    """update_forecast_probability should change probability and set updated_at."""
    qid = await _seed_forecast(store, probability=0.65)
    await store.update_forecast_probability(qid, 0.80, source="reprice")

    # Verify the probability changed
    cursor = await store.db.execute(
        "SELECT probability, updated_at FROM forecast_questions WHERE id = ?", (qid,)
    )
    row = await cursor.fetchone()
    assert row[0] == pytest.approx(0.80)
    assert row[1] is not None  # updated_at set


@pytest.mark.asyncio
async def test_update_forecast_probability_creates_history(store):
    """update_forecast_probability should insert into history table."""
    qid = await _seed_forecast(store, probability=0.65)
    await store.update_forecast_probability(qid, 0.80, source="reprice")
    await store.update_forecast_probability(qid, 0.72, source="daily")

    history = await store.get_forecast_probability_history(qid)
    assert len(history) == 2
    assert history[0]["probability"] == pytest.approx(0.80)
    assert history[0]["source"] == "reprice"
    assert history[1]["probability"] == pytest.approx(0.72)
    assert history[1]["source"] == "daily"


@pytest.mark.asyncio
async def test_get_forecast_probability_history_empty(store):
    """History should be empty for a forecast that was never repriced."""
    qid = await _seed_forecast(store, probability=0.65)
    history = await store.get_forecast_probability_history(qid)
    assert history == []


@pytest.mark.asyncio
async def test_update_forecast_probability_with_market_prob(store):
    """update_forecast_probability should optionally store market probability."""
    qid = await _seed_forecast(store, probability=0.65)
    await store.update_forecast_probability(qid, 0.80, source="kalshi_refresh", market_probability=0.73)

    history = await store.get_forecast_probability_history(qid)
    assert len(history) == 1
    assert history[0]["market_probability"] == pytest.approx(0.73)


@pytest.mark.asyncio
async def test_get_open_forecasts_includes_topic_and_engine(store):
    """Open forecasts should include topic_slug, engine, and key fields."""
    qid = await _seed_forecast(store, topic_slug="iran", engine="actor")
    open_forecasts = await store.get_open_forecasts()
    forecast = next(f for f in open_forecasts if f["forecast_question_id"] == qid)
    assert forecast["topic_slug"] == "iran"
    assert forecast["engine"] == "actor"
    assert "probability" in forecast
    assert "question" in forecast
    assert "resolution_date" in forecast
    assert "external_ref" in forecast
    assert "target_variable" in forecast
