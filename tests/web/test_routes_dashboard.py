"""Tests for dashboard route — edge cases beyond test_app.py smoke tests.

Covers: briefing detail, pipeline trigger guards, pipeline status,
audio path traversal, find_briefing fallback, find_audio multi-language,
topic emoji mapping.
"""

import pytest
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, patch

from httpx import AsyncClient, ASGITransport

from nexus.engine.knowledge.store import KnowledgeStore
from nexus.web.app import create_app
from nexus.web.routes.dashboard import (
    _find_audio,
    _find_briefing,
    _topic_emoji,
    briefing_age_badge,
)


# ── Unit tests for pure helpers ──


class TestTopicEmoji:
    def test_iran_topic(self):
        assert "&#127758;" in _topic_emoji("iran-us")

    def test_ai_topic(self):
        assert "&#129302;" in _topic_emoji("ai-research")

    def test_unknown_topic_fallback(self):
        assert "&#128196;" in _topic_emoji("obscure-topic")

    def test_case_insensitive(self):
        assert "&#129302;" in _topic_emoji("AI-Research")


class TestFindBriefing:
    def test_exact_date_match(self, tmp_path):
        briefing_dir = tmp_path / "artifacts" / "briefings"
        briefing_dir.mkdir(parents=True)
        (briefing_dir / "2026-03-17.md").write_text("# March 17")
        result = _find_briefing(tmp_path, date(2026, 3, 17))
        assert result is not None
        assert result[0] == date(2026, 3, 17)
        assert "March 17" in result[1]

    def test_fallback_to_most_recent(self, tmp_path):
        briefing_dir = tmp_path / "artifacts" / "briefings"
        briefing_dir.mkdir(parents=True)
        (briefing_dir / "2026-03-15.md").write_text("# March 15")
        (briefing_dir / "2026-03-14.md").write_text("# March 14")
        result = _find_briefing(tmp_path, date(2026, 3, 17))
        assert result is not None
        assert result[0] == date(2026, 3, 15)  # most recent

    def test_no_briefings_dir(self, tmp_path):
        result = _find_briefing(tmp_path, date(2026, 3, 17))
        assert result is None

    def test_empty_briefings_dir(self, tmp_path):
        (tmp_path / "artifacts" / "briefings").mkdir(parents=True)
        result = _find_briefing(tmp_path, date(2026, 3, 17))
        assert result is None


class TestFindAudio:
    def test_primary_audio_found(self, tmp_path):
        audio_dir = tmp_path / "artifacts" / "audio"
        audio_dir.mkdir(parents=True)
        (audio_dir / "2026-03-17.mp3").write_bytes(b"\xff\xfb")
        result = _find_audio(tmp_path, date(2026, 3, 17))
        assert result["available"] is True
        assert result["primary"] == "2026-03-17"
        assert any(l["code"] == "en" for l in result["languages"])

    def test_no_audio(self, tmp_path):
        result = _find_audio(tmp_path, date(2026, 3, 17))
        assert result["available"] is False
        assert result["languages"] == []

    def test_language_variants(self, tmp_path):
        audio_dir = tmp_path / "artifacts" / "audio"
        audio_dir.mkdir(parents=True)
        (audio_dir / "2026-03-17.mp3").write_bytes(b"\xff\xfb")
        (audio_dir / "2026-03-17-zh.mp3").write_bytes(b"\xff\xfb")
        (audio_dir / "2026-03-17-es.mp3").write_bytes(b"\xff\xfb")
        result = _find_audio(tmp_path, date(2026, 3, 17))
        assert result["available"] is True
        codes = {l["code"] for l in result["languages"]}
        assert codes == {"en", "zh", "es"}

    def test_only_language_variant_no_primary(self, tmp_path):
        """Language variant without primary .mp3 still counts as available."""
        audio_dir = tmp_path / "artifacts" / "audio"
        audio_dir.mkdir(parents=True)
        (audio_dir / "2026-03-17-zh.mp3").write_bytes(b"\xff\xfb")
        result = _find_audio(tmp_path, date(2026, 3, 17))
        assert result["available"] is True
        assert result["primary"] is None


# ── Integration tests ──


@pytest.fixture
async def dashboard_app(tmp_path):
    """App with config, no seeded data."""
    (tmp_path / "config.yaml").write_text("preset: balanced\ntopics:\n  - name: test\n")
    app = create_app(tmp_path / "test.db", data_dir=tmp_path)
    store = KnowledgeStore(tmp_path / "test.db")
    await store.initialize()
    app.state.store = store
    app.state.data_dir = tmp_path
    yield app
    await store.close()


@pytest.fixture
async def client(dashboard_app):
    transport = ASGITransport(app=dashboard_app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


# ── Briefing detail route ──


async def test_briefing_detail_valid_date(dashboard_app, tmp_path):
    briefing_dir = tmp_path / "artifacts" / "briefings"
    briefing_dir.mkdir(parents=True)
    (briefing_dir / "2026-03-15.md").write_text("# Test Briefing\n\nContent here.")

    transport = ASGITransport(app=dashboard_app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.get("/briefings/2026-03-15")
    assert resp.status_code == 200
    assert "Test Briefing" in resp.text


async def test_briefing_detail_invalid_date_format(client):
    resp = await client.get("/briefings/not-a-date")
    assert resp.status_code == 404


async def test_briefing_detail_no_briefing_for_date(client):
    resp = await client.get("/briefings/2020-01-01")
    assert resp.status_code == 404


# ── Pipeline trigger guards ──


async def test_pipeline_trigger_already_running(dashboard_app):
    store = dashboard_app.state.store
    await store.start_pipeline_run(["test"], "scheduled")

    transport = ASGITransport(app=dashboard_app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.post("/api/pipeline/run")
    assert resp.status_code == 200
    assert "already running" in resp.text


async def test_pipeline_trigger_cooldown(dashboard_app):
    store = dashboard_app.state.store
    run_id = await store.start_pipeline_run(["test"], "manual")
    await store.complete_pipeline_run(run_id, event_count=5, cost_usd=0.01)

    transport = ASGITransport(app=dashboard_app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.post("/api/pipeline/run")
    assert resp.status_code == 200
    assert "Cooldown" in resp.text or "cooldown" in resp.text


async def test_pipeline_trigger_no_config(tmp_path):
    """No config.yaml should reject trigger."""
    app = create_app(tmp_path / "test.db", data_dir=tmp_path)
    store = KnowledgeStore(tmp_path / "test.db")
    await store.initialize()
    app.state.store = store
    app.state.data_dir = tmp_path

    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.post("/api/pipeline/run")
    assert resp.status_code == 200
    assert "No config" in resp.text or "setup" in resp.text.lower()
    await store.close()


# ── Pipeline status endpoint ──


async def test_pipeline_status_no_runs(client):
    resp = await client.get("/api/pipeline/status")
    assert resp.status_code == 200
    assert "No pipeline runs" in resp.text


async def test_pipeline_status_running(dashboard_app):
    store = dashboard_app.state.store
    await store.start_pipeline_run(["test"], "manual")

    transport = ASGITransport(app=dashboard_app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.get("/api/pipeline/status")
    assert resp.status_code == 200
    assert "running" in resp.text.lower()


async def test_pipeline_status_completed(dashboard_app):
    store = dashboard_app.state.store
    run_id = await store.start_pipeline_run(["test"], "manual")
    await store.complete_pipeline_run(run_id, event_count=42, cost_usd=0.15)

    transport = ASGITransport(app=dashboard_app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.get("/api/pipeline/status")
    assert resp.status_code == 200
    assert "42 events" in resp.text
    assert "$0.15" in resp.text


async def test_pipeline_status_failed(dashboard_app):
    store = dashboard_app.state.store
    run_id = await store.start_pipeline_run(["test"], "manual")
    await store.fail_pipeline_run(run_id, "LLM timeout")

    transport = ASGITransport(app=dashboard_app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.get("/api/pipeline/status")
    assert resp.status_code == 200
    assert "failed" in resp.text.lower()
    assert "LLM timeout" in resp.text


# ── Audio path traversal ──


async def test_audio_path_traversal_blocked(dashboard_app, tmp_path):
    audio_dir = tmp_path / "artifacts" / "audio"
    audio_dir.mkdir(parents=True)
    (audio_dir / "legit.mp3").write_bytes(b"\xff\xfb")

    transport = ASGITransport(app=dashboard_app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.get("/audio/../../../etc/passwd")
    assert resp.status_code == 404


async def test_audio_non_mp3_blocked(dashboard_app, tmp_path):
    audio_dir = tmp_path / "artifacts" / "audio"
    audio_dir.mkdir(parents=True)
    (audio_dir / "secret.txt").write_text("secrets")

    transport = ASGITransport(app=dashboard_app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.get("/audio/secret.txt")
    assert resp.status_code == 404


# ── Health endpoint ──


async def test_health_no_config(tmp_path):
    """Health endpoint returns 503 when config.yaml is missing."""
    app = create_app(tmp_path / "test.db", data_dir=tmp_path)
    store = KnowledgeStore(tmp_path / "test.db")
    await store.initialize()
    app.state.store = store
    app.state.data_dir = tmp_path

    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.get("/api/health")
    assert resp.status_code == 503
    body = resp.json()
    assert body["status"] == "critical"
    await store.close()
