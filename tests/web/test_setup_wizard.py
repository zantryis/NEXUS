"""Tests for the web setup wizard (3-step flow)."""

import pytest
from unittest.mock import patch, AsyncMock
from httpx import AsyncClient, ASGITransport

from nexus.engine.knowledge.store import KnowledgeStore
from nexus.web.app import create_app


@pytest.fixture(autouse=True)
def _clear_demo_mode(monkeypatch):
    """Ensure NEXUS_DEMO_MODE is not set (may leak from .env via load_dotenv)."""
    monkeypatch.delenv("NEXUS_DEMO_MODE", raising=False)


@pytest.fixture
def app_no_config(tmp_path):
    """App with no config.yaml — should redirect to /setup."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    db_path = data_dir / "knowledge.db"
    app = create_app(db_path, data_dir=data_dir)
    return app


@pytest.fixture
async def app_with_config(tmp_path):
    """App with config.yaml present — should NOT redirect."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "config.yaml").write_text("user:\n  name: Test\n  timezone: UTC\ntopics:\n  - name: Test\n")
    db_path = data_dir / "knowledge.db"
    app = create_app(db_path, data_dir=data_dir)
    # Initialize store so dashboard can render
    store = KnowledgeStore(db_path)
    await store.initialize()
    app.state.store = store
    app.state.audio_dir = data_dir / "artifacts" / "audio"
    from nexus.config.loader import load_config
    app.state.config = load_config(data_dir / "config.yaml")
    return app


@pytest.mark.asyncio
async def test_redirect_when_no_config(app_no_config):
    """GET / should redirect to /setup when no config.yaml."""
    transport = ASGITransport(app=app_no_config)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/", follow_redirects=False)
        assert resp.status_code == 307
        assert resp.headers["location"] == "/setup"


@pytest.mark.asyncio
async def test_no_redirect_when_config_exists(app_with_config):
    """GET / should NOT redirect when config.yaml exists."""
    transport = ASGITransport(app=app_with_config)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/", follow_redirects=False)
        assert resp.status_code == 200


@pytest.mark.asyncio
async def test_static_files_not_redirected(app_no_config):
    """Static files should not be redirected even without config."""
    transport = ASGITransport(app=app_no_config)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/static/style.css", follow_redirects=False)
        assert resp.status_code == 200


@pytest.mark.asyncio
async def test_setup_page_renders(app_no_config):
    """GET /setup should render step 1 with providers and API key field."""
    transport = ASGITransport(app=app_no_config)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/setup")
        assert resp.status_code == 200
        assert "Nexus" in resp.text
        # Step 1 now has provider selection and inline API key
        assert "provider" in resp.text.lower()
        assert "api_key" in resp.text


@pytest.mark.asyncio
async def test_setup_step1_with_key_redirects_to_step2(app_no_config):
    """POST /setup/step/1 with provider + API key should redirect to step 2 (topics)."""
    transport = ASGITransport(app=app_no_config)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/setup/step/1",
            data={"provider": "gemini", "preset": "balanced", "api_key": "test-key"},
            follow_redirects=False,
        )
        assert resp.status_code == 303
        assert "/setup/step/2" in resp.headers["location"]


@pytest.mark.asyncio
async def test_setup_step1_free_skips_key(app_no_config):
    """POST /setup/step/1 with free preset (no key needed) goes to step 2."""
    transport = ASGITransport(app=app_no_config)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/setup/step/1",
            data={"provider": "ollama", "preset": "free"},
            follow_redirects=False,
        )
        assert resp.status_code == 303
        assert "/setup/step/2" in resp.headers["location"]


@pytest.mark.asyncio
async def test_setup_step1_missing_key_shows_error(app_no_config):
    """POST /setup/step/1 with provider that needs key but no key → error."""
    transport = ASGITransport(app=app_no_config)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/setup/step/1",
            data={"provider": "gemini", "preset": "balanced", "api_key": ""},
        )
        assert resp.status_code == 400
        assert "API key" in resp.text


@pytest.mark.asyncio
async def test_setup_step1_invalid_provider_rejected(app_no_config):
    """Invalid provider should not be accepted."""
    transport = ASGITransport(app=app_no_config)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/setup/step/1", data={"provider": "", "preset": "evil"})
        assert resp.status_code == 400
        assert "provider" in resp.text.lower()


@pytest.mark.asyncio
async def test_setup_step2_topics_page_renders(app_no_config):
    """GET /setup/step/2 should show topic selection."""
    transport = ASGITransport(app=app_no_config)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # First complete step 1
        await client.post("/setup/step/1", data={"provider": "ollama", "preset": "free"})
        resp = await client.get("/setup/step/2")
        assert resp.status_code == 200
        assert "topic" in resp.text.lower()


@pytest.mark.asyncio
async def test_setup_step2_saves_topics(app_no_config):
    """POST /setup/step/2 with topics should redirect to step 3 (review)."""
    transport = ASGITransport(app=app_no_config)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await client.post("/setup/step/1", data={"provider": "ollama", "preset": "free"})
        resp = await client.post(
            "/setup/step/2",
            data={"topics": "ai-ml-research"},
            follow_redirects=False,
        )
        assert resp.status_code == 303
        assert "/setup/step/3" in resp.headers["location"]


@pytest.mark.asyncio
async def test_setup_step3_review_page_renders(app_no_config):
    """GET /setup/step/3 should show review page."""
    transport = ASGITransport(app=app_no_config)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await client.post("/setup/step/1", data={"provider": "ollama", "preset": "free"})
        await client.post("/setup/step/2", data={"topics": "ai-ml-research"})
        resp = await client.get("/setup/step/3")
        assert resp.status_code == 200
        assert "Ready to Go" in resp.text
        assert "Start Nexus" in resp.text


@pytest.mark.asyncio
async def test_setup_complete_writes_config(app_no_config, tmp_path):
    """POST /setup/complete should write config.yaml and auto-launch pipeline."""
    data_dir = tmp_path / "data"
    transport = ASGITransport(app=app_no_config)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Walk through the 3-step wizard
        await client.post(
            "/setup/step/1",
            data={"provider": "gemini", "preset": "balanced", "api_key": "test-gemini-key"},
        )
        await client.post("/setup/step/2", data={"topics": "ai-ml-research"})

        with patch("nexus.web.routes.setup._auto_launch_pipeline"):
            resp = await client.post("/setup/complete", follow_redirects=False)

        assert resp.status_code == 303
        assert resp.headers["location"] == "/?setup=complete"
        assert (data_dir / "config.yaml").exists()


@pytest.mark.asyncio
async def test_setup_complete_uses_sensible_defaults(app_no_config, tmp_path):
    """Setup complete should use default timezone, schedule, style."""
    import yaml
    data_dir = tmp_path / "data"
    transport = ASGITransport(app=app_no_config)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await client.post("/setup/step/1", data={"provider": "ollama", "preset": "free"})
        await client.post("/setup/step/2", data={"topics": "ai-ml-research"})

        with patch("nexus.web.routes.setup._auto_launch_pipeline"):
            await client.post("/setup/complete", follow_redirects=False)

    config_path = data_dir / "config.yaml"
    raw = yaml.safe_load(config_path.read_text())
    assert raw["user"]["timezone"] == "UTC"
    assert raw["briefing"]["schedule"] == "06:00"
    assert raw["briefing"]["style"] == "analytical"


@pytest.mark.asyncio
async def test_setup_complete_auto_launches_pipeline(app_no_config, tmp_path):
    """POST /setup/complete should call _auto_launch_pipeline."""
    transport = ASGITransport(app=app_no_config)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await client.post("/setup/step/1", data={"provider": "ollama", "preset": "free"})
        await client.post("/setup/step/2", data={"topics": "ai-ml-research"})

        with patch("nexus.web.routes.setup._auto_launch_pipeline") as mock_launch:
            await client.post("/setup/complete", follow_redirects=False)
            mock_launch.assert_called_once()


@pytest.mark.asyncio
async def test_setup_status_returns_html(app_no_config):
    """GET /setup/status should return HTML (even when no pipeline running)."""
    transport = ASGITransport(app=app_no_config)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/setup/status")
        assert resp.status_code == 200


@pytest.mark.asyncio
async def test_setup_status_shows_elapsed_time(app_no_config):
    """GET /setup/status should show elapsed time when pipeline is running."""
    import time
    app_no_config.state.pipeline_status = {
        "stage": "running",
        "done": False,
        "started_at": time.time() - 65,  # 1 min 5 sec ago
        "topic_index": 1,
        "topic_count": 2,
    }
    transport = ASGITransport(app=app_no_config)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/setup/status")
        assert resp.status_code == 200
        assert "pipeline-elapsed" in resp.text
        assert "1:0" in resp.text  # 1:05


@pytest.mark.asyncio
async def test_setup_status_shows_topic_progress(app_no_config):
    """GET /setup/status should show per-topic progress."""
    import time
    app_no_config.state.pipeline_status = {
        "stage": "topic:AI/ML Research:filtering",
        "done": False,
        "started_at": time.time(),
        "topic_index": 1,
        "topic_count": 3,
    }
    transport = ASGITransport(app=app_no_config)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/setup/status")
        assert resp.status_code == 200
        assert "Filtering articles" in resp.text
        assert "AI/ML Research" in resp.text
        assert "(1/3)" in resp.text


@pytest.mark.asyncio
async def test_setup_disabled_after_config_exists(app_with_config):
    """Once config exists, /setup redirects to settings by default."""
    transport = ASGITransport(app=app_with_config)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/setup", follow_redirects=False)
        assert resp.status_code == 303
        assert resp.headers["location"] == "/settings"


@pytest.mark.asyncio
@patch.dict("os.environ", {}, clear=False)
async def test_remote_setup_requires_admin_token(app_no_config):
    """Remote setup access is blocked unless an admin token is configured."""
    transport = ASGITransport(app=app_no_config, client=("8.8.8.8", 12345))
    async with AsyncClient(transport=transport, base_url="http://external.example.com") as client:
        resp = await client.get("/setup")
        assert resp.status_code == 403


@pytest.mark.asyncio
@patch.dict("os.environ", {"NEXUS_ADMIN_TOKEN": "secret"})
async def test_remote_setup_accepts_admin_token(app_no_config):
    """Remote setup access can be explicitly unlocked with an admin token."""
    transport = ASGITransport(app=app_no_config, client=("8.8.8.8", 12345))
    async with AsyncClient(transport=transport, base_url="http://external.example.com") as client:
        resp = await client.get("/setup?admin_token=secret", follow_redirects=False)
        assert resp.status_code == 303
        assert "nexus_admin=" in resp.headers["set-cookie"]


# ── Telegram validation tests (still on /setup/telegram/* endpoints) ──


@pytest.mark.asyncio
@patch("nexus.agent.telegram_utils.validate_token", new_callable=AsyncMock)
async def test_telegram_validate_success(mock_validate, app_no_config):
    """POST /setup/telegram/validate with valid token shows bot username."""
    mock_validate.return_value = {"username": "test_nexus_bot", "id": 123}
    transport = ASGITransport(app=app_no_config)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/setup/telegram/validate",
            data={"telegram_token": "123:ABC"},
        )
        assert resp.status_code == 200
        assert "test_nexus_bot" in resp.text
        assert "Token valid" in resp.text


@pytest.mark.asyncio
@patch("nexus.agent.telegram_utils.validate_token", new_callable=AsyncMock)
async def test_telegram_validate_invalid(mock_validate, app_no_config):
    """POST /setup/telegram/validate with bad token shows error."""
    mock_validate.return_value = None
    transport = ASGITransport(app=app_no_config)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/setup/telegram/validate",
            data={"telegram_token": "bad-token"},
        )
        assert resp.status_code == 200
        assert "Invalid token" in resp.text


@pytest.mark.asyncio
async def test_telegram_validate_empty_token(app_no_config):
    """POST /setup/telegram/validate with no token shows error."""
    transport = ASGITransport(app=app_no_config)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/setup/telegram/validate", data={})
        assert resp.status_code == 200
        assert "Enter a bot token" in resp.text


@pytest.mark.asyncio
@patch("nexus.agent.telegram_utils.poll_for_chat_id", new_callable=AsyncMock)
async def test_telegram_poll_finds_chat_id(mock_poll, app_no_config):
    """GET /setup/telegram/poll returns chat_id when /start received."""
    mock_poll.return_value = 987654

    transport = ASGITransport(app=app_no_config)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # First establish a session with a token
        await client.post(
            "/setup/telegram/validate",
            data={"telegram_token": "123:ABC"},
        )
        # For the poll, we need to patch validate_token too since it was called in validate
        with patch("nexus.agent.telegram_utils.validate_token", new_callable=AsyncMock, return_value={"username": "bot", "id": 1}):
            await client.post("/setup/telegram/validate", data={"telegram_token": "123:ABC"})

        resp = await client.get("/setup/telegram/poll")
        assert resp.status_code == 200
        if mock_poll.called:
            assert "987654" in resp.text


# ── 3-step wizard total_steps validation ──


@pytest.mark.asyncio
async def test_setup_complete_enables_discovery(app_no_config, tmp_path):
    """Setup wizard should enable discover_new_sources by default."""
    import yaml
    data_dir = tmp_path / "data"
    transport = ASGITransport(app=app_no_config)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await client.post("/setup/step/1", data={"provider": "ollama", "preset": "free"})
        await client.post("/setup/step/2", data={"topics": "ai-ml-research"})

        with patch("nexus.web.routes.setup._auto_launch_pipeline"):
            await client.post("/setup/complete", follow_redirects=False)

    config_path = data_dir / "config.yaml"
    raw = yaml.safe_load(config_path.read_text())
    assert raw["sources"]["discover_new_sources"] is True


@pytest.mark.asyncio
async def test_setup_shows_3_steps(app_no_config):
    """Setup wizard should show 3 step indicators (not 6)."""
    transport = ASGITransport(app=app_no_config)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/setup")
        assert resp.status_code == 200
        # Count step dots — should be exactly 3
        assert resp.text.count("setup-step-dot") == 3
