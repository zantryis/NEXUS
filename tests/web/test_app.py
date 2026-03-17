"""Tests for the Nexus web dashboard."""

import pytest
from datetime import date
from httpx import AsyncClient, ASGITransport

from nexus.engine.knowledge.store import KnowledgeStore
from nexus.engine.knowledge.events import Event
from nexus.engine.projection.models import ProjectionItem, ThreadSnapshot, TopicProjection
from nexus.web.app import create_app


@pytest.fixture
async def seeded_app(tmp_path):
    """Create an app with a seeded in-memory DB."""
    db_path = tmp_path / "test.db"
    (tmp_path / "config.yaml").write_text("preset: balanced\ntopics:\n  - name: test\n")
    app = create_app(db_path, data_dir=tmp_path)

    # Manually initialize and seed the store before tests
    store = KnowledgeStore(db_path)
    await store.initialize()

    # Seed events
    e1 = Event(
        date=date(2026, 3, 9), summary="Iran sanctions announced",
        significance=8, entities=["US", "Iran", "Treasury Dept"],
        sources=[{"url": "https://reuters.com/1", "outlet": "reuters",
                  "affiliation": "private", "country": "US", "language": "en"}],
    )
    e2 = Event(
        date=date(2026, 3, 10), summary="Iran condemns sanctions",
        significance=7, entities=["Iran", "US"],
        sources=[{"url": "https://tass.com/1", "outlet": "tass",
                  "affiliation": "state", "country": "RU", "language": "ru"}],
    )
    event_ids = await store.add_events([e1, e2], "iran-us")

    # Seed thread
    tid = await store.upsert_thread("sanctions-escalation", "Sanctions Escalation", 8, "active")
    await store.link_thread_topic(tid, "iran-us")
    await store.link_thread_events(tid, event_ids)
    await store.add_convergence(tid, "Sanctions are real", ["reuters", "tass"])
    await store.add_divergence(
        tid, "Impact assessment",
        "reuters", "Devastating impact",
        "tass", "Minimal effect",
    )
    await store.upsert_thread_snapshot(ThreadSnapshot(
        thread_id=tid,
        snapshot_date=date(2026, 3, 9),
        status="active",
        significance=8,
        event_count=1,
        latest_event_date=date(2026, 3, 9),
    ))
    await store.upsert_thread_snapshot(ThreadSnapshot(
        thread_id=tid,
        snapshot_date=date(2026, 3, 10),
        status="active",
        significance=8,
        event_count=2,
        latest_event_date=date(2026, 3, 10),
    ))
    await store.save_projection(TopicProjection(
        topic_slug="iran-us",
        topic_name="Iran-US",
        generated_for=date(2026, 3, 10),
        summary="Forward look summary",
        items=[ProjectionItem(
            claim="Sanctions pressure is likely to persist.",
            confidence="medium",
            horizon_days=7,
            signpost="Watch for Treasury follow-through",
            evidence_thread_ids=[tid],
            review_after=date(2026, 3, 17),
        )],
    ))

    # Seed a page
    await store.save_page(
        "backstory:iran-us", "Iran-US Background", "backstory",
        "# History\n\nLong backstory...", "iran-us", 7, "hash1",
    )

    # Seed filter log
    await store.add_filter_log([
        {"run_date": "2026-03-10", "topic_slug": "iran-us",
         "url": "https://a.com", "title": "Accepted Article",
         "source_id": "reuters", "source_affiliation": "private",
         "source_country": "US", "outcome": "accepted", "passed_pass1": True,
         "relevance_score": 9.0, "significance_score": 8.0},
        {"run_date": "2026-03-10", "topic_slug": "iran-us",
         "url": "https://b.com", "title": "Rejected Article",
         "source_id": "blog", "source_affiliation": "private",
         "source_country": "US", "outcome": "rejected_relevance",
         "passed_pass1": False, "relevance_score": 2.0},
    ])

    # Inject the store directly into app state (lifespan doesn't run in test transport)
    app.state.store = store

    yield app

    await store.close()


@pytest.fixture
async def client(seeded_app):
    """Async HTTP client for testing."""
    transport = ASGITransport(app=seeded_app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def test_dashboard_returns_200(client):
    resp = await client.get("/")
    assert resp.status_code == 200
    assert "Nexus" in resp.text


async def test_dashboard_sets_security_headers(client):
    resp = await client.get("/")
    assert resp.headers["x-frame-options"] == "DENY"
    assert resp.headers["x-content-type-options"] == "nosniff"
    assert resp.headers["referrer-policy"] == "same-origin"
    assert "frame-ancestors 'none'" in resp.headers["content-security-policy"]


async def test_dashboard_setup_complete_banner(client):
    resp = await client.get("/?setup=complete")
    assert resp.status_code == 200
    assert "Setup complete" in resp.text


async def test_dashboard_shows_topics(client):
    resp = await client.get("/")
    assert "iran-us" in resp.text


async def test_dashboard_shows_briefing_or_empty(client):
    """Dashboard shows briefing content or welcome message."""
    resp = await client.get("/")
    # Should show either briefing content or empty state
    assert "Intelligence Briefing" in resp.text or "Welcome to Nexus" in resp.text


async def test_dashboard_shows_sidebar(client):
    """Dashboard sidebar has threads and topics."""
    resp = await client.get("/")
    assert "Active Threads" in resp.text or "Welcome to Nexus" in resp.text


async def test_dashboard_shows_health_panel(client, monkeypatch):
    monkeypatch.setenv("LITELLM_PROXY_URL", "https://proxy.example/v1")
    monkeypatch.setenv("LITELLM_PROXY_API_KEY", "secret")
    monkeypatch.setenv("LITELLM_MODEL_GPT", "gpt-5.4")
    resp = await client.get("/")
    assert resp.status_code == 200
    assert "System Health" in resp.text


async def test_api_health_returns_snapshot(seeded_app, monkeypatch):
    monkeypatch.setenv("LITELLM_PROXY_URL", "https://proxy.example/v1")
    monkeypatch.setenv("LITELLM_PROXY_API_KEY", "secret")
    monkeypatch.setenv("LITELLM_MODEL_GPT", "gpt-5.4")

    transport = ASGITransport(app=seeded_app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.get("/api/health")
    assert resp.status_code in (200, 503)
    body = resp.json()
    assert "status" in body
    assert "pipeline" in body
    assert "deliverables" in body


async def test_topic_detail(client):
    resp = await client.get("/topics/iran-us")
    assert resp.status_code == 200
    assert "Sanctions Escalation" in resp.text
    assert "Forward Look" in resp.text


async def test_thread_list(client):
    resp = await client.get("/threads/")
    assert resp.status_code == 200
    assert "Sanctions Escalation" in resp.text


async def test_thread_detail(client):
    resp = await client.get("/threads/sanctions-escalation")
    assert resp.status_code == 200
    assert "Sanctions Escalation" in resp.text
    assert "What Sources Agree On" in resp.text
    assert "Where They Disagree" in resp.text
    assert "Trajectory" in resp.text


async def test_thread_not_found(client):
    resp = await client.get("/threads/nonexistent")
    assert resp.status_code == 404


async def test_event_list(client):
    resp = await client.get("/events/")
    assert resp.status_code == 200
    assert "Iran sanctions announced" in resp.text


async def test_event_detail(client):
    resp = await client.get("/events/1")
    assert resp.status_code == 200
    assert "reuters" in resp.text


async def test_event_not_found(client):
    resp = await client.get("/events/99999")
    assert resp.status_code == 404


async def test_entity_list(client):
    resp = await client.get("/entities/")
    assert resp.status_code == 200
    assert "Iran" in resp.text


async def test_entity_search(client):
    resp = await client.get("/entities/?q=Iran")
    assert resp.status_code == 200
    assert "Iran" in resp.text


async def test_entity_detail(client):
    resp = await client.get("/entities/1")
    assert resp.status_code == 200


async def test_entity_not_found(client):
    resp = await client.get("/entities/99999")
    assert resp.status_code == 404


async def test_page_view(client):
    resp = await client.get("/pages/backstory:iran-us")
    assert resp.status_code == 200
    assert "History" in resp.text


async def test_page_not_found(client):
    resp = await client.get("/pages/nonexistent")
    assert resp.status_code == 404


async def test_filter_log(client):
    resp = await client.get("/filters/iran-us/2026-03-10")
    assert resp.status_code == 200
    assert "Accepted Article" in resp.text
    assert "Rejected Article" in resp.text


async def test_source_stats(client):
    resp = await client.get("/sources/")
    assert resp.status_code == 200
    assert "reuters" in resp.text


async def test_audio_route_serves_mp3(seeded_app, tmp_path):
    """Audio route serves existing MP3 files."""
    audio_dir = tmp_path / "artifacts" / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    (audio_dir / "2026-03-10.mp3").write_bytes(b"\xff\xfb\x90\x00" * 10)
    seeded_app.state.data_dir = tmp_path

    transport = ASGITransport(app=seeded_app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.get("/audio/2026-03-10.mp3")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "audio/mpeg"


async def test_audio_route_404_missing(seeded_app, tmp_path):
    """Audio route returns 404 for missing files."""
    seeded_app.state.data_dir = tmp_path

    transport = ASGITransport(app=seeded_app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.get("/audio/nonexistent.mp3")
        assert resp.status_code == 404


async def test_predictions_page_loads(client):
    """Predictions page returns 200 even with no forecast data."""
    resp = await client.get("/predictions")
    assert resp.status_code == 200
    assert "Predictions" in resp.text


async def test_predictions_page_shows_empty_state(client):
    """Predictions page shows empty state when no forecasts exist."""
    resp = await client.get("/predictions")
    assert resp.status_code == 200
    assert "No predictions yet" in resp.text


async def test_dashboard_shows_audio_player(seeded_app, tmp_path):
    """Dashboard shows audio player when MP3 exists for briefing date."""
    # Create briefing + audio
    briefing_dir = tmp_path / "artifacts" / "briefings"
    briefing_dir.mkdir(parents=True, exist_ok=True)
    today = date.today().isoformat()
    (briefing_dir / f"{today}.md").write_text("# Today's briefing")
    audio_dir = tmp_path / "artifacts" / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    (audio_dir / f"{today}.mp3").write_bytes(b"\xff\xfb\x90\x00" * 10)
    seeded_app.state.data_dir = tmp_path

    transport = ASGITransport(app=seeded_app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.get("/")
        assert resp.status_code == 200
        assert "audio-player-card" in resp.text
        assert "Daily Podcast" in resp.text
