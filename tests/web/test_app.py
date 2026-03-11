"""Tests for the Nexus web dashboard."""

import pytest
from datetime import date
from httpx import AsyncClient, ASGITransport

from nexus.engine.knowledge.store import KnowledgeStore
from nexus.engine.knowledge.events import Event
from nexus.web.app import create_app


@pytest.fixture
async def seeded_app(tmp_path):
    """Create an app with a seeded in-memory DB."""
    db_path = tmp_path / "test.db"
    app = create_app(db_path)

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


async def test_dashboard_shows_topics(client):
    resp = await client.get("/")
    assert "iran-us" in resp.text


async def test_dashboard_has_metrics(client):
    """Dashboard shows hero metrics grid."""
    resp = await client.get("/")
    assert "metrics-grid" in resp.text
    assert "Active Threads" in resp.text
    assert "Topics" in resp.text
    assert "Events" in resp.text


async def test_dashboard_has_thread_cards(client):
    """Dashboard shows redesigned thread cards with significance."""
    resp = await client.get("/")
    assert "Sanctions Escalation" in resp.text
    assert "thread-card-v2" in resp.text


async def test_topic_detail(client):
    resp = await client.get("/topics/iran-us")
    assert resp.status_code == 200
    assert "Sanctions Escalation" in resp.text


async def test_thread_list(client):
    resp = await client.get("/threads/")
    assert resp.status_code == 200
    assert "Sanctions Escalation" in resp.text


async def test_thread_detail(client):
    resp = await client.get("/threads/sanctions-escalation")
    assert resp.status_code == 200
    assert "Sanctions Escalation" in resp.text
    assert "Convergence" in resp.text
    assert "Divergence" in resp.text


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
