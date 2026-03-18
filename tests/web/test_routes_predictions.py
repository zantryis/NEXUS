"""Tests for predictions route — unit tests for helpers + integration with seeded data.

Covers: _derive_verdict, _clean_question, _enrich_forecast, _group_by_market,
_group_by_event, predictions page with seeded forecast data.
"""

import pytest
from datetime import date

from httpx import AsyncClient, ASGITransport

from nexus.engine.knowledge.events import Event
from nexus.engine.knowledge.store import KnowledgeStore
from nexus.engine.projection.models import ForecastQuestion, ForecastRun
from nexus.web.app import create_app
from nexus.web.routes.predictions import (
    _clean_question,
    _derive_verdict,
    _enrich_forecast,
    _group_by_event,
    _group_by_market,
    ENGINE_INFO,
    ENGINE_ORDER,
)


# ── Unit tests: _derive_verdict ──


class TestDeriveVerdict:
    def test_high_probability_yes(self):
        assert _derive_verdict(0.85) == "yes"

    def test_low_probability_no(self):
        assert _derive_verdict(0.15) == "no"

    def test_midrange_uncertain(self):
        assert _derive_verdict(0.50) == "uncertain"

    def test_boundary_65_yes(self):
        assert _derive_verdict(0.65) == "yes"

    def test_boundary_35_no(self):
        assert _derive_verdict(0.35) == "no"

    def test_none_returns_none(self):
        assert _derive_verdict(None) is None


# ── Unit tests: _clean_question ──


class TestCleanQuestion:
    def test_kalshi_format(self):
        group, display = _clean_question("UK PM?: Will Starmer resign by June?")
        assert group == "UK PM"
        assert display == "Will Starmer resign by June"

    def test_colon_separator(self):
        group, display = _clean_question("Iran Sanctions: Will new sanctions be imposed?")
        assert group == "Iran Sanctions"
        assert display == "Will new sanctions be imposed"

    def test_no_separator(self):
        group, display = _clean_question("Will it rain tomorrow?")
        assert group == ""
        assert display == "Will it rain tomorrow"

    def test_trailing_question_mark_stripped(self):
        _, display = _clean_question("Simple question?")
        assert not display.endswith("?")


# ── Unit tests: _enrich_forecast ──


class TestEnrichForecast:
    def test_basic_enrichment(self):
        q = {
            "engine": "actor",
            "question": "Event Title?: Market Question?",
            "probability": 0.75,
            "target_variable": "kalshi_aligned",
            "target_metadata": {"kalshi_ticker": "IRAN-Y", "kalshi_implied": 0.60},
        }
        result = _enrich_forecast(q, date(2026, 3, 17))
        assert result["ticker"] == "IRAN-Y"
        assert result["market_prob"] == 0.60
        assert result["verdict"] == "yes"
        assert result["source_type"] == "kalshi"
        assert result["gap_pp"] == 15.0
        assert result["display_question"] == "Market Question"
        assert result["group_title"] == "Event Title"
        assert result["engine_label"] == "Actor"

    def test_kg_native_source_type(self):
        q = {
            "engine": "structural",
            "question": "Will X happen?",
            "probability": 0.30,
            "target_variable": "kg_native",
            "target_metadata": {},
        }
        result = _enrich_forecast(q, date(2026, 3, 17))
        assert result["source_type"] == "kg"
        assert result["run_label"] == "independent"

    def test_resolution_date_enrichment(self):
        q = {
            "engine": "actor",
            "question": "Test?",
            "probability": 0.5,
            "target_variable": "kalshi_aligned",
            "target_metadata": {},
            "resolution_date": "2026-03-24",
        }
        result = _enrich_forecast(q, date(2026, 3, 17))
        assert result["days_until_resolution"] == 7
        assert result["resolution_date_str"] == "2026-03-24"

    def test_resolved_hit(self):
        q = {
            "engine": "actor",
            "question": "Test?",
            "probability": 0.8,
            "target_variable": "kalshi_benchmark",
            "target_metadata": {},
            "outcome_status": "resolved",
            "brier_score": 0.04,
        }
        result = _enrich_forecast(q, date(2026, 3, 17))
        assert result["status_label"] == "hit"

    def test_resolved_miss(self):
        q = {
            "engine": "naked",
            "question": "Test?",
            "probability": 0.5,
            "target_variable": "kalshi_benchmark",
            "target_metadata": {},
            "outcome_status": "resolved",
            "brier_score": 0.40,
        }
        result = _enrich_forecast(q, date(2026, 3, 17))
        assert result["status_label"] == "miss"

    def test_no_market_prob_gap_is_none(self):
        q = {
            "engine": "actor",
            "question": "Test?",
            "probability": 0.7,
            "target_variable": "kg_native",
            "target_metadata": {},
        }
        result = _enrich_forecast(q, date(2026, 3, 17))
        assert result["gap_pp"] is None


# ── Unit tests: _group_by_market ──


class TestGroupByMarket:
    def _make_forecast(self, engine, question, prob, ticker="", run_label=None, source_type="kalshi"):
        return {
            "engine": engine,
            "question": question,
            "display_question": question,
            "group_title": "",
            "probability": prob,
            "ticker": ticker,
            "market_prob": 0.5,
            "source_type": source_type,
            "run_label": run_label,
            "days_until_resolution": 7,
            "resolution_date_str": "2026-03-24",
            "generated_for": "2026-03-17",
            "status_label": "pending",
            "outcome_status": None,
            "resolved_bool": None,
            "brier_score": None,
            "engine_info": ENGINE_INFO.get(engine, {}),
            "engine_class": ENGINE_INFO.get(engine, {}).get("class", "unknown"),
            "engine_label": ENGINE_INFO.get(engine, {}).get("label", engine),
        }

    def test_groups_same_question_across_engines(self):
        forecasts = [
            self._make_forecast("actor", "Will X happen?", 0.7, ticker="X-Y"),
            self._make_forecast("structural", "Will X happen?", 0.6, ticker="X-Y"),
        ]
        markets = _group_by_market(forecasts)
        assert len(markets) == 1
        assert markets[0]["engine_count"] == 2

    def test_separates_different_questions(self):
        forecasts = [
            self._make_forecast("actor", "Q1?", 0.7, ticker="T1"),
            self._make_forecast("actor", "Q2?", 0.3, ticker="T2"),
        ]
        markets = _group_by_market(forecasts)
        assert len(markets) == 2

    def test_consensus_probability(self):
        forecasts = [
            self._make_forecast("actor", "Q?", 0.8, ticker="T"),
            self._make_forecast("structural", "Q?", 0.6, ticker="T"),
        ]
        markets = _group_by_market(forecasts)
        assert markets[0]["consensus_prob"] == pytest.approx(0.7, abs=0.01)

    def test_spread_calculation(self):
        forecasts = [
            self._make_forecast("actor", "Q?", 0.9, ticker="T"),
            self._make_forecast("naked", "Q?", 0.3, ticker="T"),
        ]
        markets = _group_by_market(forecasts)
        assert markets[0]["spread_pp"] == pytest.approx(60.0)

    def test_single_engine_no_spread(self):
        forecasts = [self._make_forecast("actor", "Q?", 0.7, ticker="T")]
        markets = _group_by_market(forecasts)
        assert markets[0]["spread_pp"] is None

    def test_engine_order_preserved(self):
        forecasts = [
            self._make_forecast("debate", "Q?", 0.5, ticker="T"),
            self._make_forecast("actor", "Q?", 0.7, ticker="T"),
            self._make_forecast("structural", "Q?", 0.6, ticker="T"),
        ]
        markets = _group_by_market(forecasts)
        engines = [name for name, _ in markets[0]["engine_list"]]
        # structural before actor before debate
        assert engines.index("structural") < engines.index("actor")
        assert engines.index("actor") < engines.index("debate")

    def test_anchored_and_independent_separate_groups(self):
        forecasts = [
            self._make_forecast("actor", "Q?", 0.7, ticker="T", run_label="anchored"),
            self._make_forecast("actor", "Q?", 0.6, ticker="T", run_label="independent"),
        ]
        markets = _group_by_market(forecasts)
        assert len(markets) == 2


# ── Unit tests: _group_by_event ──


class TestGroupByEvent:
    def test_groups_by_title(self):
        markets = [
            {"group_title": "UK PM", "source_type": "kalshi"},
            {"group_title": "UK PM", "source_type": "kalshi"},
            {"group_title": "Iran", "source_type": "kalshi"},
        ]
        events = _group_by_event(markets)
        assert len(events) == 2
        titles = {e["title"] for e in events}
        assert titles == {"UK PM", "Iran"}

    def test_standalone_group(self):
        markets = [
            {"group_title": "", "source_type": "kg"},
        ]
        events = _group_by_event(markets)
        assert events[0]["title"] == "Standalone"


# ── Integration tests: predictions page with data ──


@pytest.fixture
async def predictions_app(tmp_path):
    """App with seeded forecast runs (pending + resolved)."""
    (tmp_path / "config.yaml").write_text("preset: balanced\ntopics:\n  - name: test\n")
    app = create_app(tmp_path / "test.db", data_dir=tmp_path)
    store = KnowledgeStore(tmp_path / "test.db")
    await store.initialize()

    # Seed events first (needed for topic_slug foreign key)
    e = Event(
        date=date(2026, 3, 15), summary="Test event",
        significance=5, entities=["Test"],
        sources=[{"url": "https://test.com", "outlet": "test"}],
    )
    await store.add_events([e], "test-topic")

    # Seed a pending forecast run
    pending_run = ForecastRun(
        topic_slug="test-topic",
        topic_name="Test Topic",
        engine="actor",
        generated_for=date(2026, 3, 15),
        summary="Test forecast",
        questions=[
            ForecastQuestion(
                question="Will X happen by March 30?",
                target_variable="kalshi_aligned",
                target_metadata={"kalshi_ticker": "X-YES", "kalshi_implied": 0.55},
                probability=0.70,
                resolution_criteria="X happens before March 30",
                resolution_date=date(2026, 3, 30),
                horizon_days=14,
                signpost="Watch for X indicators",
            ),
        ],
    )
    await store.save_forecast_run(pending_run)

    # Seed a resolved forecast run
    resolved_run = ForecastRun(
        topic_slug="test-topic",
        topic_name="Test Topic",
        engine="structural",
        generated_for=date(2026, 3, 10),
        summary="Old forecast",
        questions=[
            ForecastQuestion(
                question="Did Y happen?",
                target_variable="kalshi_benchmark",
                target_metadata={"kalshi_ticker": "Y-YES", "kalshi_implied": 0.40, "run_label": "anchored"},
                probability=0.80,
                resolution_criteria="Y resolved by market close",
                resolution_date=date(2026, 3, 14),
                horizon_days=3,
                signpost="Market close on March 14",
            ),
        ],
    )
    run_id = await store.save_forecast_run(resolved_run)

    # Resolve the question
    cursor = await store.db.execute(
        "SELECT fq.id FROM forecast_questions fq "
        "JOIN forecast_runs fr ON fq.forecast_run_id = fr.id "
        "WHERE fr.id = ?", (run_id,),
    )
    row = await cursor.fetchone()
    if row:
        await store.db.execute(
            "UPDATE forecast_resolutions SET outcome_status='resolved', "
            "resolved_bool=1, brier_score=0.04 WHERE forecast_question_id=?",
            (row[0],),
        )
        await store.db.commit()

    app.state.store = store
    yield app
    await store.close()


@pytest.fixture
async def pred_client(predictions_app):
    transport = ASGITransport(app=predictions_app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def test_predictions_with_data_returns_200(pred_client):
    resp = await pred_client.get("/predictions")
    assert resp.status_code == 200
    assert "Predictions" in resp.text


async def test_predictions_shows_pending_market(pred_client):
    resp = await pred_client.get("/predictions")
    assert resp.status_code == 200
    assert "X-YES" in resp.text or "Will X happen" in resp.text


async def test_predictions_shows_resolved_market(pred_client):
    resp = await pred_client.get("/predictions")
    assert resp.status_code == 200
    assert "Y-YES" in resp.text or "Did Y happen" in resp.text


async def test_predictions_shows_engine_info(pred_client):
    resp = await pred_client.get("/predictions")
    assert resp.status_code == 200
    assert "Actor" in resp.text or "Structural" in resp.text


async def test_predictions_empty_state(tmp_path):
    """No forecast data → shows empty state."""
    (tmp_path / "config.yaml").write_text("preset: balanced\ntopics:\n  - name: test\n")
    app = create_app(tmp_path / "test.db", data_dir=tmp_path)
    store = KnowledgeStore(tmp_path / "test.db")
    await store.initialize()
    app.state.store = store

    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.get("/predictions")
    assert resp.status_code == 200
    assert "No predictions" in resp.text
    await store.close()


# ── ENGINE_INFO coverage ──


def test_engine_info_covers_all_ordered_engines():
    """Every engine in ENGINE_ORDER should have info."""
    for engine in ENGINE_ORDER:
        assert engine in ENGINE_INFO
        assert "label" in ENGINE_INFO[engine]
        assert "class" in ENGINE_INFO[engine]


def test_engine_order_has_six_engines():
    assert len(ENGINE_ORDER) == 6
