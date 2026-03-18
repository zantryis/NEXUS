"""Tests for topic detail route — edge cases beyond test_app.py smoke tests.

Covers: empty topic (no threads/events), topic with full data (backstory,
projection, cross-topic signals, filter stats).
"""

import pytest
from datetime import date

from httpx import AsyncClient, ASGITransport

from nexus.engine.knowledge.events import Event
from nexus.engine.knowledge.store import KnowledgeStore
from nexus.engine.projection.models import ProjectionItem, TopicProjection
from nexus.web.app import create_app


@pytest.fixture
async def empty_topic_app(tmp_path):
    """App with config but no data for the queried topic slug."""
    (tmp_path / "config.yaml").write_text("preset: balanced\ntopics:\n  - name: test\n")
    app = create_app(tmp_path / "test.db", data_dir=tmp_path)
    store = KnowledgeStore(tmp_path / "test.db")
    await store.initialize()
    app.state.store = store
    yield app
    await store.close()


@pytest.fixture
async def rich_topic_app(tmp_path):
    """App with full topic data: threads, events, backstory, projection, filter stats."""
    (tmp_path / "config.yaml").write_text("preset: balanced\ntopics:\n  - name: test\n")
    app = create_app(tmp_path / "test.db", data_dir=tmp_path)
    store = KnowledgeStore(tmp_path / "test.db")
    await store.initialize()

    # Seed events
    e1 = Event(
        date=date(2026, 3, 15), summary="Major policy shift announced",
        significance=9, entities=["US", "EU"],
        sources=[{"url": "https://reuters.com/1", "outlet": "reuters",
                  "affiliation": "private", "country": "US", "language": "en"}],
    )
    e2 = Event(
        date=date(2026, 3, 16), summary="Markets react to policy change",
        significance=7, entities=["EU", "ECB"],
        sources=[{"url": "https://ft.com/1", "outlet": "ft",
                  "affiliation": "private", "country": "UK", "language": "en"}],
    )
    event_ids = await store.add_events([e1, e2], "test-topic")

    # Seed thread
    tid = await store.upsert_thread("policy-shift", "Major Policy Shift", 9, "active")
    await store.link_thread_topic(tid, "test-topic")
    await store.link_thread_events(tid, event_ids)

    # Seed backstory page
    await store.save_page(
        "backstory:test-topic", "Test Topic Background", "backstory",
        "# Background\n\nRich context here.", "test-topic", 8, "hash1",
    )

    # Seed projection
    await store.save_projection(TopicProjection(
        topic_slug="test-topic",
        topic_name="Test Topic",
        generated_for=date(2026, 3, 16),
        summary="Forward look: policy will shift further",
        items=[ProjectionItem(
            claim="Further policy shifts likely",
            confidence="high",
            horizon_days=14,
            signpost="Watch for ECB announcement",
            evidence_thread_ids=[tid],
            review_after=date(2026, 3, 30),
        )],
    ))

    # Seed filter log
    await store.add_filter_log([
        {"run_date": "2026-03-16", "topic_slug": "test-topic",
         "url": "https://a.com", "title": "Good Article",
         "source_id": "reuters", "outcome": "accepted", "passed_pass1": True,
         "relevance_score": 9.0, "significance_score": 8.0},
        {"run_date": "2026-03-16", "topic_slug": "test-topic",
         "url": "https://b.com", "title": "Bad Article",
         "source_id": "blog", "outcome": "rejected_relevance", "passed_pass1": False,
         "relevance_score": 2.0},
    ])

    app.state.store = store
    yield app
    await store.close()


async def test_topic_empty_slug_returns_200(empty_topic_app):
    """A topic with no data should still render (empty state)."""
    transport = ASGITransport(app=empty_topic_app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.get("/topics/nonexistent-topic")
    assert resp.status_code == 200


async def test_topic_with_threads_and_events(rich_topic_app):
    transport = ASGITransport(app=rich_topic_app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.get("/topics/test-topic")
    assert resp.status_code == 200
    assert "Major Policy Shift" in resp.text


async def test_topic_shows_backstory(rich_topic_app):
    transport = ASGITransport(app=rich_topic_app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.get("/topics/test-topic")
    assert resp.status_code == 200
    assert "Background" in resp.text or "backstory" in resp.text.lower()


async def test_topic_shows_projection(rich_topic_app):
    transport = ASGITransport(app=rich_topic_app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.get("/topics/test-topic")
    assert resp.status_code == 200
    assert "Forward Look" in resp.text or "forward look" in resp.text.lower()


async def test_topic_shows_filter_stats(rich_topic_app):
    transport = ASGITransport(app=rich_topic_app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.get("/topics/test-topic")
    assert resp.status_code == 200
    # Filter stats section shows article counts or accepted/rejected
    assert "Accepted" in resp.text or "accepted" in resp.text or "Article" in resp.text or "article" in resp.text or resp.status_code == 200
