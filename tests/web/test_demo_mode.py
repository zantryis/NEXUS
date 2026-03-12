"""Tests for demo mode middleware."""

import pytest
from unittest.mock import patch

import yaml
from httpx import AsyncClient, ASGITransport

from nexus.engine.knowledge.store import KnowledgeStore
from nexus.config.loader import load_config
from nexus.web.app import create_app


@pytest.fixture
async def demo_app(tmp_path):
    """App with config, in demo mode."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    config_dict = {
        "preset": "balanced",
        "user": {"name": "Demo", "timezone": "UTC", "output_language": "en"},
        "topics": [{"name": "AI", "priority": "high"}],
        "briefing": {"schedule": "06:00", "style": "analytical"},
        "audio": {"enabled": False, "tts_backend": "gemini"},
        "telegram": {"enabled": False},
        "budget": {"daily_limit_usd": 1.00, "warning_threshold_usd": 0.50},
    }
    (data_dir / "config.yaml").write_text(yaml.dump(config_dict, sort_keys=False))

    db_path = data_dir / "knowledge.db"
    app = create_app(db_path, data_dir=data_dir)
    store = KnowledgeStore(db_path)
    await store.initialize()
    app.state.store = store
    app.state.audio_dir = data_dir / "artifacts" / "audio"
    app.state.config = load_config(data_dir / "config.yaml")
    return app


@pytest.mark.asyncio
@patch.dict("os.environ", {"NEXUS_DEMO_MODE": "1"})
async def test_demo_blocks_settings_post(demo_app):
    """POST /settings/* is blocked in demo mode."""
    transport = ASGITransport(app=demo_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/settings/user", data={
            "name": "Hacker", "timezone": "UTC", "output_language": "en",
        })
        assert resp.status_code == 403
        assert "demo" in resp.json()["error"].lower()


@pytest.mark.asyncio
@patch.dict("os.environ", {"NEXUS_DEMO_MODE": "1"})
async def test_demo_blocks_setup_post(demo_app):
    """POST /setup/* is blocked in demo mode."""
    transport = ASGITransport(app=demo_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/setup/step/1", data={"preset": "balanced"})
        assert resp.status_code == 403


@pytest.mark.asyncio
@patch.dict("os.environ", {"NEXUS_DEMO_MODE": "1"})
async def test_demo_allows_dashboard_get(demo_app):
    """GET / is allowed in demo mode."""
    transport = ASGITransport(app=demo_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/")
        assert resp.status_code == 200


@pytest.mark.asyncio
@patch.dict("os.environ", {"NEXUS_DEMO_MODE": "1"})
async def test_demo_allows_settings_get(demo_app):
    """GET /settings is allowed in demo mode (read-only view)."""
    transport = ASGITransport(app=demo_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/settings")
        assert resp.status_code == 200


@pytest.mark.asyncio
@patch.dict("os.environ", {"NEXUS_DEMO_MODE": "1"})
async def test_demo_shows_badge(demo_app):
    """Demo mode badge appears in dashboard HTML."""
    transport = ASGITransport(app=demo_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/")
        assert "Demo" in resp.text


@pytest.mark.asyncio
@patch.dict("os.environ", {"NEXUS_DEMO_MODE": ""})
async def test_no_demo_allows_settings_post(demo_app):
    """POST /settings/* is allowed when demo mode is off."""
    transport = ASGITransport(app=demo_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/settings/user", data={
            "name": "Normal", "timezone": "UTC", "output_language": "en",
        }, follow_redirects=False)
        assert resp.status_code == 303  # redirect after save
