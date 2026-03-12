"""Tests for OpenAI OAuth web routes."""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock

import yaml
from httpx import AsyncClient, ASGITransport

from nexus.engine.knowledge.store import KnowledgeStore
from nexus.config.loader import load_config
from nexus.web.app import create_app


@pytest.fixture
async def oauth_app(tmp_path):
    """App with config for OAuth testing."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    config_dict = {
        "preset": "openai-balanced",
        "user": {"name": "Tester", "timezone": "UTC", "output_language": "en"},
        "topics": [{"name": "AI", "priority": "high"}],
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
async def test_oauth_initiate_redirects(oauth_app):
    """GET /auth/openai should redirect to OpenAI authorization URL."""
    app, _ = oauth_app
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/auth/openai", follow_redirects=False)
        assert resp.status_code == 307
        location = resp.headers["location"]
        assert "auth.openai.com" in location
        assert "code_challenge" in location


@pytest.mark.asyncio
async def test_oauth_callback_exchanges_code(oauth_app):
    """GET /auth/openai/callback with code should exchange for tokens."""
    app, data_dir = oauth_app
    transport = ASGITransport(app=app)

    # Set up PKCE session state
    if not hasattr(app.state, "oauth_sessions"):
        app.state.oauth_sessions = {}
    app.state.oauth_sessions["test-state"] = {
        "code_verifier": "test-verifier-string-that-is-long-enough-to-be-valid",
    }

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "access_token": "oauth-at-123",
        "refresh_token": "oauth-rt-456",
        "expires_in": 3600,
        "token_type": "Bearer",
    }

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_response):
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/auth/openai/callback",
                params={"code": "auth-code-xyz", "state": "test-state"},
                follow_redirects=False,
            )
            assert resp.status_code == 303
            assert "/settings" in resp.headers["location"]

    # Token file should exist
    token_path = data_dir / ".oauth-tokens.json"
    assert token_path.exists()


@pytest.mark.asyncio
async def test_oauth_callback_invalid_state(oauth_app):
    """Callback with unknown state should return error."""
    app, _ = oauth_app
    transport = ASGITransport(app=app)
    if not hasattr(app.state, "oauth_sessions"):
        app.state.oauth_sessions = {}

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get(
            "/auth/openai/callback",
            params={"code": "some-code", "state": "bad-state"},
            follow_redirects=False,
        )
        assert resp.status_code == 400


@pytest.mark.asyncio
async def test_oauth_revoke_clears_tokens(oauth_app):
    """POST /auth/openai/revoke should clear stored tokens."""
    app, data_dir = oauth_app
    token_path = data_dir / ".oauth-tokens.json"
    token_path.write_text('{"access_token": "x"}')

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/auth/openai/revoke", follow_redirects=False)
        assert resp.status_code == 303

    assert not token_path.exists()
