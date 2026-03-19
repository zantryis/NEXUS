"""Tests for benchmark route — engine comparison on resolved predictions."""

import pytest
from datetime import date

from httpx import AsyncClient, ASGITransport

from nexus.engine.knowledge.events import Event
from nexus.engine.knowledge.store import KnowledgeStore
from nexus.engine.projection.models import ForecastQuestion, ForecastRun
from nexus.web.app import create_app


@pytest.fixture
async def benchmark_app(tmp_path):
    """App with seeded benchmark forecast runs."""
    (tmp_path / "config.yaml").write_text("preset: balanced\ntopics:\n  - name: test\n")
    app = create_app(tmp_path / "test.db", data_dir=tmp_path)
    store = KnowledgeStore(tmp_path / "test.db")
    await store.initialize()

    # Seed events
    e = Event(
        date=date(2026, 3, 10), summary="Test event",
        significance=5, entities=["Test"],
        sources=[{"url": "https://test.com", "outlet": "test"}],
    )
    await store.add_events([e], "test-topic")

    # Seed a benchmark forecast run (resolved)
    run = ForecastRun(
        topic_slug="test-topic",
        topic_name="Test Topic",
        engine="actor",
        generated_for=date(2026, 3, 10),
        summary="Benchmark run",
        questions=[
            ForecastQuestion(
                question="Will X happen by March 14?",
                target_variable="kalshi_benchmark",
                target_metadata={"kalshi_ticker": "X-YES", "kalshi_implied": 0.50, "run_label": "anchored"},
                probability=0.75,
                resolution_criteria="Market settles",
                resolution_date=date(2026, 3, 14),
                horizon_days=3,
                signpost="Watch X",
            ),
        ],
    )
    run_id = await store.save_forecast_run(run)

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
            "resolved_bool=1, brier_score=0.0625 WHERE forecast_question_id=?",
            (row[0],),
        )
        await store.db.commit()

    app.state.store = store
    yield app
    await store.close()


@pytest.fixture
async def bench_client(benchmark_app):
    transport = ASGITransport(app=benchmark_app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def test_benchmark_returns_200(bench_client):
    resp = await bench_client.get("/benchmark")
    assert resp.status_code == 200
    assert "Benchmark" in resp.text


async def test_benchmark_shows_resolved_data(bench_client):
    resp = await bench_client.get("/benchmark")
    assert resp.status_code == 200
    assert "X-YES" in resp.text or "Will X happen" in resp.text


async def test_benchmark_shows_engine_comparison(bench_client):
    resp = await bench_client.get("/benchmark")
    assert resp.status_code == 200
    assert "Actor" in resp.text


async def test_benchmark_dev_mode(bench_client):
    resp = await bench_client.get("/benchmark?dev=1")
    assert resp.status_code == 200
    assert "dev mode" in resp.text


async def test_benchmark_empty_state(tmp_path):
    (tmp_path / "config.yaml").write_text("preset: balanced\ntopics:\n  - name: test\n")
    app = create_app(tmp_path / "test.db", data_dir=tmp_path)
    store = KnowledgeStore(tmp_path / "test.db")
    await store.initialize()
    app.state.store = store

    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.get("/benchmark")
    assert resp.status_code == 200
    assert "No benchmark results" in resp.text
    await store.close()


async def test_benchmark_not_shown_on_forward_look(bench_client):
    """Benchmark data should NOT appear on the Forward Look page."""
    resp = await bench_client.get("/forward-look")
    assert resp.status_code == 200
    # The benchmark ticker should not be on the public Forward Look page
    assert "X-YES" not in resp.text
