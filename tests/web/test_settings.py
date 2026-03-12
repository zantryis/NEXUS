"""Tests for the settings page."""

import pytest
import yaml
from httpx import AsyncClient, ASGITransport

from nexus.engine.knowledge.store import KnowledgeStore
from nexus.config.loader import load_config
from nexus.web.app import create_app


@pytest.fixture
async def settings_app(tmp_path):
    """App with a valid config for settings testing."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    config_dict = {
        "preset": "balanced",
        "user": {"name": "Tristan", "timezone": "America/Denver", "output_language": "en"},
        "topics": [
            {"name": "AI/ML Research", "priority": "high"},
            {"name": "Formula 1", "priority": "medium"},
        ],
        "briefing": {"schedule": "06:00", "style": "analytical"},
        "audio": {"enabled": True, "tts_backend": "gemini"},
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
    return app, data_dir


@pytest.mark.asyncio
async def test_settings_page_loads(settings_app):
    app, _ = settings_app
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/settings")
        assert resp.status_code == 200
        assert "Settings" in resp.text
        assert "Gemini" in resp.text
        assert "Tristan" in resp.text


@pytest.mark.asyncio
async def test_settings_shows_all_providers(settings_app):
    app, _ = settings_app
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/settings")
        assert "OpenAI" in resp.text
        assert "ElevenLabs" in resp.text
        assert "Telegram Bot" in resp.text


@pytest.mark.asyncio
async def test_settings_post_user_updates_config(settings_app):
    app, data_dir = settings_app
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/settings/user", data={
            "name": "NewName", "timezone": "Europe/London", "output_language": "fr",
        }, follow_redirects=False)
        assert resp.status_code == 303
        assert "saved=user" in resp.headers["location"]

    # Verify config was updated
    raw = yaml.safe_load((data_dir / "config.yaml").read_text())
    assert raw["user"]["name"] == "NewName"
    assert raw["user"]["timezone"] == "Europe/London"


@pytest.mark.asyncio
async def test_settings_post_shows_restart_banner(settings_app):
    app, _ = settings_app
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # First save something
        await client.post("/settings/user", data={
            "name": "Test", "timezone": "UTC", "output_language": "en",
        })
        # Then follow redirect to settings?saved=user
        resp = await client.get("/settings?saved=user")
        assert "Restart" in resp.text


@pytest.mark.asyncio
async def test_settings_post_keys(settings_app):
    app, data_dir = settings_app
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/settings/keys", data={
            "OPENAI_API_KEY": "sk-test-key",
        }, follow_redirects=False)
        assert resp.status_code == 303

    env_path = data_dir.parent / ".env"
    assert "sk-test-key" in env_path.read_text()


@pytest.mark.asyncio
async def test_settings_add_topic(settings_app):
    app, data_dir = settings_app
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/settings/topics", data={
            "new_topic": "Cryptocurrency",
        }, follow_redirects=False)
        assert resp.status_code == 303

    raw = yaml.safe_load((data_dir / "config.yaml").read_text())
    topic_names = [t["name"] for t in raw["topics"]]
    assert "Cryptocurrency" in topic_names
    assert len(raw["topics"]) == 3  # original 2 + new 1


@pytest.mark.asyncio
async def test_settings_toggle_audio(settings_app):
    app, data_dir = settings_app
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Disable audio (no checkbox = off)
        resp = await client.post("/settings/audio", data={
            "tts_backend": "openai",
            "voice_host_a": "nova",
            "voice_host_b": "onyx",
        }, follow_redirects=False)
        assert resp.status_code == 303

    raw = yaml.safe_load((data_dir / "config.yaml").read_text())
    assert raw["audio"]["enabled"] is False
    assert raw["audio"]["tts_backend"] == "openai"


@pytest.mark.asyncio
async def test_settings_custom_models(settings_app):
    """POST /settings/preset with custom saves per-stage model selections."""
    app, data_dir = settings_app
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/settings/preset", data={
            "preset": "custom",
            "model_filtering": "gpt-5-mini",
            "model_synthesis": "claude-sonnet-4-6",
            "model_agent": "deepseek-chat",
            "model_dialogue_script": "gemini-3.1-pro-preview",
            "model_knowledge_summary": "gpt-4.1-nano",
            "model_breaking_news": "gpt-4.1-nano",
            "model_discovery": "gpt-4.1-nano",
        }, follow_redirects=False)
        assert resp.status_code == 303

    raw = yaml.safe_load((data_dir / "config.yaml").read_text())
    assert raw["preset"] == "custom"
    assert raw["models"]["filtering"] == "gpt-5-mini"
    assert raw["models"]["synthesis"] == "claude-sonnet-4-6"
    assert raw["models"]["agent"] == "deepseek-chat"


@pytest.mark.asyncio
async def test_settings_preset_clears_custom_models(settings_app):
    """Switching from custom to a named preset clears custom model overrides."""
    app, data_dir = settings_app
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # First set custom models
        await client.post("/settings/preset", data={
            "preset": "custom",
            "model_filtering": "gpt-5-mini",
            "model_synthesis": "claude-sonnet-4-6",
            "model_agent": "deepseek-chat",
            "model_dialogue_script": "gemini-3.1-pro-preview",
            "model_knowledge_summary": "gpt-4.1-nano",
            "model_breaking_news": "gpt-4.1-nano",
            "model_discovery": "gpt-4.1-nano",
        })
        # Then switch back to a named preset
        await client.post("/settings/preset", data={"preset": "balanced"})

    raw = yaml.safe_load((data_dir / "config.yaml").read_text())
    assert raw["preset"] == "balanced"
    assert "models" not in raw


@pytest.mark.asyncio
async def test_settings_page_shows_model_choices(settings_app):
    """Settings page should include model selection dropdowns."""
    app, _ = settings_app
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/settings")
        assert "custom" in resp.text.lower()
        assert "gpt-5.4" in resp.text
        assert "claude-sonnet-4-6" in resp.text
        assert "deepseek-chat" in resp.text
