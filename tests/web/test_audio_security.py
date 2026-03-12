"""Tests for audio file serving security."""

import pytest
from httpx import AsyncClient, ASGITransport

from nexus.engine.knowledge.store import KnowledgeStore
from nexus.web.app import create_app


@pytest.fixture
async def audio_app(tmp_path):
    """App with an audio file for security testing."""
    import yaml

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    audio_dir = data_dir / "artifacts" / "audio"
    audio_dir.mkdir(parents=True)

    # Create a legit audio file
    (audio_dir / "briefing.mp3").write_bytes(b"fake-mp3-data")

    # Create a secret file outside audio dir
    (data_dir / "secret.txt").write_text("secret-data")

    config_dict = {
        "preset": "balanced",
        "user": {"name": "Test", "timezone": "UTC"},
        "topics": [{"name": "Test", "priority": "high"}],
    }
    (data_dir / "config.yaml").write_text(yaml.dump(config_dict))

    db_path = data_dir / "knowledge.db"
    app = create_app(db_path, data_dir=data_dir)
    store = KnowledgeStore(db_path)
    await store.initialize()
    app.state.store = store
    app.state.audio_dir = audio_dir
    from nexus.config.loader import load_config
    app.state.config = load_config(data_dir / "config.yaml")
    return app


@pytest.mark.asyncio
async def test_serve_valid_audio(audio_app):
    """Valid .mp3 file is served normally."""
    transport = ASGITransport(app=audio_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/audio/briefing.mp3")
        assert resp.status_code == 200
        assert resp.content == b"fake-mp3-data"


@pytest.mark.asyncio
async def test_path_traversal_blocked(audio_app):
    """Path traversal attempt returns 404."""
    transport = ASGITransport(app=audio_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/audio/../../secret.txt")
        assert resp.status_code in (404, 400)


@pytest.mark.asyncio
async def test_path_traversal_with_mp3_extension(audio_app):
    """Path traversal with .mp3 suffix is still blocked."""
    transport = ASGITransport(app=audio_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/audio/../secret.txt.mp3")
        assert resp.status_code in (404, 400)
        if resp.status_code == 200:
            assert b"secret-data" not in resp.content


@pytest.mark.asyncio
async def test_nonexistent_audio_returns_404(audio_app):
    """Non-existent file returns 404."""
    transport = ASGITransport(app=audio_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/audio/nonexistent.mp3")
        assert resp.status_code == 404
