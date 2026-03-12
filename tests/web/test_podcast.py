"""Tests for the podcast RSS feed route."""

import pytest
from httpx import AsyncClient, ASGITransport

from nexus.engine.knowledge.store import KnowledgeStore
from nexus.web.app import create_app


@pytest.fixture
async def podcast_app(tmp_path):
    """App with seeded audio files."""
    db_path = tmp_path / "test.db"
    app = create_app(db_path)

    # Create fake audio files
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    (audio_dir / "2026-03-10.mp3").write_bytes(b"fake-mp3-content")
    (audio_dir / "2026-03-09.mp3").write_bytes(b"fake-mp3-older")
    app.state.audio_dir = audio_dir

    # Initialize store
    store = KnowledgeStore(db_path)
    await store.initialize()
    app.state.store = store

    yield app
    await store.close()


@pytest.fixture
async def client(podcast_app):
    transport = ASGITransport(app=podcast_app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def test_feed_xml_returns_rss(client):
    resp = await client.get("/feed.xml")
    assert resp.status_code == 200
    assert "application/rss+xml" in resp.headers["content-type"]
    assert b"<rss" in resp.content
    assert b"Nexus Intelligence Briefing" in resp.content


async def test_feed_xml_contains_episodes(client):
    resp = await client.get("/feed.xml")
    assert b"2026-03-10" in resp.content
    assert b"2026-03-09" in resp.content
    assert b"audio/mpeg" in resp.content


async def test_feed_xml_empty_dir(tmp_path):
    """Feed should still return valid RSS with no episodes."""
    db_path = tmp_path / "test.db"
    app = create_app(db_path)
    app.state.audio_dir = tmp_path / "empty_audio"

    store = KnowledgeStore(db_path)
    await store.initialize()
    app.state.store = store

    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.get("/feed.xml")
    assert resp.status_code == 200
    assert b"<rss" in resp.content

    await store.close()
