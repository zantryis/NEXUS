"""Tests for the Run Now pipeline trigger endpoints."""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import patch, AsyncMock
from httpx import AsyncClient, ASGITransport

from nexus.engine.knowledge.store import KnowledgeStore
from nexus.web.app import create_app


@pytest.fixture
async def app(tmp_path):
    """Create app with minimal config and injected store."""
    (tmp_path / "config.yaml").write_text("preset: balanced\ntopics:\n  - name: test\n")
    app = create_app(tmp_path / "test.db", data_dir=tmp_path)

    store = KnowledgeStore(tmp_path / "test.db")
    await store.initialize()
    app.state.store = store

    yield app
    await store.close()


@pytest.fixture
async def app_no_config(tmp_path):
    """App where config.yaml exists initially (for middleware) but gets removed."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text("preset: balanced\ntopics:\n  - name: test\n")
    app = create_app(tmp_path / "test.db", data_dir=tmp_path)

    store = KnowledgeStore(tmp_path / "test.db")
    await store.initialize()
    app.state.store = store

    # Remove config so trigger_pipeline finds it missing
    config_path.unlink()

    yield app
    await store.close()


async def test_trigger_pipeline_success(app):
    """POST /api/pipeline/run returns running status when idle."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        with patch("nexus.web.routes.dashboard.asyncio.create_task"):
            resp = await client.post("/api/pipeline/run")
    assert resp.status_code == 200
    assert "Pipeline started" in resp.text
    assert "hx-get" in resp.text  # polling enabled


async def test_trigger_rejects_when_running(app):
    """POST /api/pipeline/run returns current status if pipeline already running."""
    store = app.state.store
    await store.start_pipeline_run(["test"], trigger="manual")

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/api/pipeline/run")
    assert resp.status_code == 200
    assert "already running" in resp.text


async def test_trigger_rejects_during_cooldown(app):
    """POST /api/pipeline/run enforces 30-min cooldown after completion."""
    store = app.state.store
    run_id = await store.start_pipeline_run(["test"], trigger="manual")
    await store.complete_pipeline_run(run_id, article_count=5, event_count=3, cost_usd=0.02)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/api/pipeline/run")
    assert resp.status_code == 200
    assert "Cooldown" in resp.text


async def test_trigger_no_config(app_no_config):
    """POST /api/pipeline/run returns error when config missing."""
    transport = ASGITransport(app=app_no_config)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        with patch("nexus.web.routes.dashboard.asyncio.create_task"):
            resp = await client.post("/api/pipeline/run")
    assert resp.status_code == 200
    assert "No config found" in resp.text


async def test_status_endpoint_running(app):
    """GET /api/pipeline/status returns running status when pipeline active."""
    store = app.state.store
    await store.start_pipeline_run(["test"], trigger="manual")

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/pipeline/status")
    assert resp.status_code == 200
    assert "running" in resp.text.lower()
    assert "hx-get" in resp.text  # continues polling


async def test_status_endpoint_idle(app):
    """GET /api/pipeline/status returns last run summary when idle."""
    store = app.state.store
    run_id = await store.start_pipeline_run(["test"], trigger="manual")
    await store.complete_pipeline_run(run_id, article_count=10, event_count=5, cost_usd=0.03)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/pipeline/status")
    assert resp.status_code == 200
    assert "5 events" in resp.text
    assert "$0.03" in resp.text


async def test_status_endpoint_failed(app):
    """GET /api/pipeline/status shows error for failed runs."""
    store = app.state.store
    run_id = await store.start_pipeline_run(["test"], trigger="manual")
    await store.fail_pipeline_run(run_id, "Connection timeout")

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/pipeline/status")
    assert resp.status_code == 200
    assert "Connection timeout" in resp.text
    assert "status-error" in resp.text


async def test_status_endpoint_empty(app):
    """GET /api/pipeline/status when no runs exist."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/pipeline/status")
    assert resp.status_code == 200
    assert "No pipeline runs yet" in resp.text


async def test_dashboard_shows_run_button(app):
    """Dashboard includes pipeline controls when data exists."""
    store = app.state.store
    # Seed minimal data so topics_data is populated
    from nexus.engine.knowledge.events import Event
    from datetime import date
    e = Event(date=date(2026, 3, 13), summary="Test event", significance=5,
              entities=["Test"], sources=[{"url": "http://x", "outlet": "test",
              "affiliation": "private", "country": "US", "language": "en"}])
    await store.add_events([e], "test")

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/")
    assert resp.status_code == 200
    assert "run-pipeline-btn" in resp.text or "pipeline-controls" in resp.text


async def test_dashboard_shows_last_run(app):
    """Dashboard shows last run time when a run exists."""
    store = app.state.store
    from nexus.engine.knowledge.events import Event
    from datetime import date
    e = Event(date=date(2026, 3, 13), summary="Test event", significance=5,
              entities=["Test"], sources=[{"url": "http://x", "outlet": "test",
              "affiliation": "private", "country": "US", "language": "en"}])
    await store.add_events([e], "test")

    run_id = await store.start_pipeline_run(["test"], trigger="manual")
    await store.complete_pipeline_run(run_id, article_count=5, event_count=3, cost_usd=0.01)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/")
    assert resp.status_code == 200
    assert "Last run:" in resp.text
