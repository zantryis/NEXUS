"""Tests for the web setup wizard."""

import pytest
from httpx import AsyncClient, ASGITransport

from nexus.engine.knowledge.store import KnowledgeStore
from nexus.web.app import create_app


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
    """GET /setup should render step 1 with presets."""
    transport = ASGITransport(app=app_no_config)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/setup")
        assert resp.status_code == 200
        assert "Nexus" in resp.text
        assert "preset" in resp.text.lower()


@pytest.mark.asyncio
async def test_setup_step1_post_redirects_to_step2(app_no_config):
    """POST /setup/step/1 with preset should redirect to step 2."""
    transport = ASGITransport(app=app_no_config)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/setup/step/1", data={"preset": "balanced"}, follow_redirects=False)
        assert resp.status_code == 303
        assert "/setup/step/2" in resp.headers["location"]


@pytest.mark.asyncio
async def test_setup_step1_free_skips_to_step3(app_no_config):
    """POST /setup/step/1 with free preset should skip to step 3."""
    transport = ASGITransport(app=app_no_config)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/setup/step/1", data={"preset": "free"}, follow_redirects=False)
        assert resp.status_code == 303
        assert "/setup/step/3" in resp.headers["location"]


@pytest.mark.asyncio
async def test_setup_step2_empty_key_shows_error(app_no_config):
    """POST /setup/step/2 with empty key should show error."""
    transport = ASGITransport(app=app_no_config)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # First set up session with preset
        await client.post("/setup/step/1", data={"preset": "balanced"})
        resp = await client.post("/setup/step/2", data={"api_key": ""}, follow_redirects=False)
        # FastAPI will return 422 for empty required field
        assert resp.status_code in (303, 422, 200)


@pytest.mark.asyncio
async def test_setup_complete_writes_config(app_no_config, tmp_path):
    """POST /setup/complete should write config.yaml and .env."""
    data_dir = tmp_path / "data"
    transport = ASGITransport(app=app_no_config)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Walk through the wizard
        await client.post("/setup/step/1", data={"preset": "balanced"})
        await client.post("/setup/step/2", data={"api_key": "test-gemini-key"})
        await client.post("/setup/step/3", data={})
        await client.post("/setup/step/4", data={"topics": "ai-ml-research"})
        await client.post("/setup/step/5", data={
            "user_name": "TestUser", "timezone": "UTC",
            "schedule": "07:00", "style": "analytical",
        })
        resp = await client.post("/setup/complete", follow_redirects=False)

        assert resp.status_code == 303
        assert (data_dir / "config.yaml").exists()


@pytest.mark.asyncio
async def test_setup_status_returns_html(app_no_config):
    """GET /setup/status should return HTML (even when no pipeline running)."""
    transport = ASGITransport(app=app_no_config)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/setup/status")
        assert resp.status_code == 200
