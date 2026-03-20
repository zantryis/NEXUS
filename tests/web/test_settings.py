"""Tests for the settings page."""


import pytest
from unittest.mock import patch
import yaml
from httpx import AsyncClient, ASGITransport

from nexus.engine.knowledge.store import KnowledgeStore
from nexus.config.loader import load_config
from nexus.web.app import create_app


@pytest.fixture(autouse=True)
def _clear_demo_mode(monkeypatch):
    """Ensure NEXUS_DEMO_MODE is not set (may leak from .env via load_dotenv)."""
    monkeypatch.delenv("NEXUS_DEMO_MODE", raising=False)


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
async def test_settings_post_shows_saved_banner(settings_app):
    """Non-restart settings changes show 'saved' banner (not restart banner)."""
    app, _ = settings_app
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Save user settings (no restart required)
        await client.post("/settings/user", data={
            "name": "Test", "timezone": "UTC", "output_language": "en",
        })
        resp = await client.get("/settings?saved=user")
        assert "Settings saved" in resp.text


@pytest.mark.asyncio
async def test_settings_restart_banner_only_when_needed(settings_app):
    """Restart banner only appears when restart=1 is in query params."""
    app, _ = settings_app
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/settings?saved=all&restart=1")
        assert "Restart" in resp.text
        assert "Restart Now" in resp.text


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
@patch.dict("os.environ", {"GEMINI_API_KEY": "old-key"}, clear=False)
async def test_settings_restart_rebuilds_env_from_file(settings_app):
    app, data_dir = settings_app
    env_path = data_dir.parent / ".env"
    env_path.write_text("GEMINI_API_KEY=new-key\n")

    captured = {}

    def fake_execve(executable, argv, env):
        captured["executable"] = executable
        captured["argv"] = argv
        captured["env"] = env

    transport = ASGITransport(app=app)
    with patch("nexus.web.routes.settings.os.execve", side_effect=fake_execve):
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/settings/restart", follow_redirects=False)

    assert resp.status_code == 303
    assert captured["env"]["GEMINI_API_KEY"] == "new-key"


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
            "model_filtering": "gpt-5.4-mini",
            "model_synthesis": "claude-sonnet-4-6",
            "model_agent": "deepseek-chat",
            "model_dialogue_script": "gemini-3.1-pro-preview",
            "model_knowledge_summary": "gpt-5.4-mini",
            "model_breaking_news": "gpt-5.4-mini",
            "model_discovery": "gpt-5.4-mini",
        }, follow_redirects=False)
        assert resp.status_code == 303

    raw = yaml.safe_load((data_dir / "config.yaml").read_text())
    assert raw["preset"] == "custom"
    assert raw["models"]["filtering"] == "gpt-5.4-mini"
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
            "model_filtering": "gpt-5.4-mini",
            "model_synthesis": "claude-sonnet-4-6",
            "model_agent": "deepseek-chat",
            "model_dialogue_script": "gemini-3.1-pro-preview",
            "model_knowledge_summary": "gpt-5.4-mini",
            "model_breaking_news": "gpt-5.4-mini",
            "model_discovery": "gpt-5.4-mini",
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


@pytest.mark.asyncio
async def test_settings_invalid_preset_rejected(settings_app):
    """Named presets must be validated before writing config."""
    app, data_dir = settings_app
    before = (data_dir / "config.yaml").read_text()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/settings/preset",
            data={"preset": "evil"},
            follow_redirects=False,
        )
        assert resp.status_code == 303
        assert resp.headers["location"] == "/settings?error=invalid-preset"

    assert (data_dir / "config.yaml").read_text() == before


@pytest.mark.asyncio
async def test_graph_invalid_params_return_400(settings_app):
    """Malformed graph filters should return a 400 instead of a 500."""
    app, _ = settings_app
    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/graph-data?min_events=abc&min_co=2")
        assert resp.status_code == 400


@pytest.mark.asyncio
async def test_remote_settings_requires_localhost_or_token(settings_app):
    """Remote settings access is blocked by default."""
    app, _ = settings_app
    transport = ASGITransport(app=app, client=("8.8.8.8", 12345))
    async with AsyncClient(transport=transport, base_url="http://external.example.com") as client:
        resp = await client.get("/settings")
        assert resp.status_code == 403


@pytest.mark.asyncio
@patch.dict("os.environ", {"NEXUS_ADMIN_TOKEN": "secret"})
async def test_remote_settings_token_unlocks_session(settings_app):
    """Remote settings can be used when an explicit admin token is provided."""
    app, data_dir = settings_app
    transport = ASGITransport(app=app, client=("8.8.8.8", 12345))
    async with AsyncClient(transport=transport, base_url="http://external.example.com") as client:
        resp = await client.get("/settings?admin_token=secret", follow_redirects=False)
        assert resp.status_code == 303
        assert resp.headers["location"] == "/settings"

        resp = await client.get("/settings")
        assert resp.status_code == 200

        resp = await client.post("/settings/user", data={
            "name": "RemoteAdmin", "timezone": "UTC", "output_language": "en",
        }, follow_redirects=False)
        assert resp.status_code == 303

    raw = yaml.safe_load((data_dir / "config.yaml").read_text())
    assert raw["user"]["name"] == "RemoteAdmin"


# ── Fix 1: Checkbox fallback to raw dict when config is None ──────


@pytest.fixture
async def settings_app_no_config_obj(tmp_path):
    """App with config.yaml on disk but NO app.state.config (simulates pre-restart)."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    config_dict = {
        "preset": "balanced",
        "user": {"name": "User", "timezone": "UTC", "output_language": "en"},
        "topics": [{"name": "Test", "priority": "high"}],
        "briefing": {"schedule": "06:00", "style": "analytical"},
        "audio": {"enabled": True, "tts_backend": "gemini"},
        "telegram": {"enabled": True},
    }
    (data_dir / "config.yaml").write_text(yaml.dump(config_dict, sort_keys=False))

    db_path = data_dir / "knowledge.db"
    app = create_app(db_path, data_dir=data_dir)
    store = KnowledgeStore(db_path)
    await store.initialize()
    app.state.store = store
    app.state.audio_dir = data_dir / "artifacts" / "audio"
    # Intentionally do NOT set app.state.config — simulates serve-only mode
    return app, data_dir


@pytest.mark.asyncio
async def test_audio_checkbox_reflects_yaml_when_config_none(settings_app_no_config_obj):
    """Audio checkbox should be checked when yaml says enabled, even without app.state.config."""
    app, _ = settings_app_no_config_obj
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/settings")
        assert resp.status_code == 200
        assert 'name="audio_enabled"' in resp.text
        idx = resp.text.index('name="audio_enabled"')
        snippet = resp.text[max(0, idx - 100):idx + 100]
        assert "checked" in snippet


@pytest.mark.asyncio
async def test_telegram_checkbox_reflects_yaml_when_config_none(settings_app_no_config_obj):
    """Telegram checkbox should be checked when yaml says enabled, even without app.state.config."""
    app, _ = settings_app_no_config_obj
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/settings")
        assert resp.status_code == 200
        idx = resp.text.index('name="telegram_enabled"')
        snippet = resp.text[max(0, idx - 100):idx + 100]
        assert "checked" in snippet


# ── Fix 2: Smart restart banner ──────────────────────────────────


@pytest.mark.asyncio
async def test_save_timezone_no_restart_banner(settings_app):
    """Saving only non-restart fields should NOT show restart banner."""
    app, _ = settings_app
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/settings/save", data={
            "name": "Tristan",
            "timezone": "Europe/London",
            "output_language": "en",
            "schedule": "07:00",
            "style": "analytical",
            "depth": "detailed",
            "preset": "balanced",
            "tts_backend": "gemini",
            "voice_host_a": "Kore",
            "voice_host_b": "Puck",
            "daily_limit_usd": "1.00",
            "warning_threshold_usd": "0.50",
            "degradation_strategy": "skip_expensive",
        }, follow_redirects=False)
        assert resp.status_code == 303
        location = resp.headers["location"]
        assert "saved=" in location
        assert "restart=1" not in location


@pytest.mark.asyncio
async def test_save_preset_change_shows_restart(settings_app):
    """Changing preset should trigger restart banner."""
    app, _ = settings_app
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/settings/save", data={
            "name": "Tristan",
            "timezone": "America/Denver",
            "output_language": "en",
            "schedule": "06:00",
            "style": "analytical",
            "depth": "detailed",
            "preset": "quality",  # Changed from "balanced"
            "tts_backend": "gemini",
            "voice_host_a": "Kore",
            "voice_host_b": "Puck",
            "daily_limit_usd": "1.00",
            "warning_threshold_usd": "0.50",
            "degradation_strategy": "skip_expensive",
        }, follow_redirects=False)
        assert resp.status_code == 303
        assert "restart=1" in resp.headers["location"]


@pytest.mark.asyncio
async def test_config_hot_reloaded_after_save(settings_app):
    """After saving, app.state.config should reflect new values without restart."""
    app, _ = settings_app
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await client.post("/settings/save", data={
            "name": "NewName",
            "timezone": "Europe/London",
            "output_language": "fr",
            "schedule": "08:00",
            "style": "conversational",
            "depth": "brief",
            "preset": "balanced",
            "tts_backend": "gemini",
            "voice_host_a": "Kore",
            "voice_host_b": "Puck",
            "daily_limit_usd": "2.00",
            "warning_threshold_usd": "1.00",
            "degradation_strategy": "stop_all",
        }, follow_redirects=False)

    assert app.state.config is not None
    assert app.state.config.user.name == "NewName"
    assert app.state.config.user.timezone == "Europe/London"


# ── Fix 6: Topic source discovery trigger ────────────────────────


@pytest.mark.asyncio
async def test_add_topic_triggers_discovery_if_no_registry(settings_app):
    """Adding a topic should trigger source discovery when no registry exists."""
    app, data_dir = settings_app
    transport = ASGITransport(app=app)

    with patch("nexus.web.routes.settings._trigger_topic_discovery") as mock_discover:
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/settings/topics", data={
                "new_topic": "Cryptocurrency",
            }, follow_redirects=False)
            assert resp.status_code == 303

        mock_discover.assert_called_once()
        call_args = mock_discover.call_args
        assert "Cryptocurrency" in str(call_args)
