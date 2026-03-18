"""Tests for thread routes — list filtering, detail edge cases.

Covers: filter by status, filter by topic, detail with causal links,
detail with projection items, empty list.
"""

import pytest
from datetime import date

from httpx import AsyncClient, ASGITransport

from nexus.engine.knowledge.events import Event
from nexus.engine.knowledge.store import KnowledgeStore
from nexus.engine.projection.models import (
    CausalLink,
    ProjectionItem,
    TopicProjection,
)
from nexus.web.app import create_app


@pytest.fixture
async def thread_app(tmp_path):
    """App with multiple threads across topics and statuses."""
    (tmp_path / "config.yaml").write_text("preset: balanced\ntopics:\n  - name: test\n")
    app = create_app(tmp_path / "test.db", data_dir=tmp_path)
    store = KnowledgeStore(tmp_path / "test.db")
    await store.initialize()

    # Seed events for topic 1
    e1 = Event(
        date=date(2026, 3, 15), summary="Sanctions announced",
        significance=8, entities=["US", "Iran"],
        sources=[{"url": "https://reuters.com/1", "outlet": "reuters"}],
    )
    e2 = Event(
        date=date(2026, 3, 16), summary="Iran responds",
        significance=7, entities=["Iran"],
        sources=[{"url": "https://tass.com/1", "outlet": "tass"}],
    )
    ids_iran = await store.add_events([e1, e2], "iran-us")

    # Seed events for topic 2
    e3 = Event(
        date=date(2026, 3, 16), summary="AI breakthrough",
        significance=9, entities=["OpenAI"],
        sources=[{"url": "https://techcrunch.com/1", "outlet": "techcrunch"}],
    )
    ids_ai = await store.add_events([e3], "ai-research")

    # Thread 1: active, iran-us
    t1 = await store.upsert_thread("sanctions-escalation", "Sanctions Escalation", 8, "active")
    await store.link_thread_topic(t1, "iran-us")
    await store.link_thread_events(t1, ids_iran)
    await store.add_convergence(t1, "All agree sanctions are real", ["reuters", "tass"])
    await store.add_divergence(
        t1, "Impact assessment",
        "reuters", "Devastating", "tass", "Minimal",
    )

    # Thread 2: active, ai-research
    t2 = await store.upsert_thread("ai-safety-push", "AI Safety Push", 7, "active")
    await store.link_thread_topic(t2, "ai-research")
    await store.link_thread_events(t2, ids_ai)

    # Thread 3: stale, iran-us
    t3 = await store.upsert_thread("old-negotiations", "Old Negotiations", 5, "stale")
    await store.link_thread_topic(t3, "iran-us")

    # Causal link between events in thread 1
    await store.add_causal_link(CausalLink(
        source_event_id=ids_iran[0],
        target_event_id=ids_iran[1],
        relation_type="response_to",
        evidence_text="Iran responded directly to sanctions",
        strength=0.9,
    ))

    # Projection citing thread 1
    await store.save_projection(TopicProjection(
        topic_slug="iran-us",
        topic_name="Iran-US",
        generated_for=date(2026, 3, 16),
        summary="Tension will persist",
        items=[ProjectionItem(
            claim="More sanctions likely",
            confidence="high",
            horizon_days=14,
            signpost="Watch Treasury",
            evidence_thread_ids=[t1],
            review_after=date(2026, 3, 30),
        )],
    ))

    app.state.store = store
    app.state.thread_ids = {"iran": t1, "ai": t2, "stale": t3}
    yield app
    await store.close()


@pytest.fixture
async def client(thread_app):
    transport = ASGITransport(app=thread_app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


# ── Thread list ──


async def test_thread_list_all(client):
    resp = await client.get("/threads/")
    assert resp.status_code == 200
    assert "Sanctions Escalation" in resp.text
    assert "AI Safety Push" in resp.text


async def test_thread_list_filter_by_status_active(client):
    resp = await client.get("/threads/?status=active")
    assert resp.status_code == 200
    assert "Sanctions Escalation" in resp.text
    assert "AI Safety Push" in resp.text
    assert "Old Negotiations" not in resp.text


async def test_thread_list_filter_by_status_stale(client):
    resp = await client.get("/threads/?status=stale")
    assert resp.status_code == 200
    assert "Old Negotiations" in resp.text
    assert "Sanctions Escalation" not in resp.text


async def test_thread_list_filter_by_topic(client):
    resp = await client.get("/threads/?topic=ai-research")
    assert resp.status_code == 200
    assert "AI Safety Push" in resp.text
    assert "Sanctions Escalation" not in resp.text


async def test_thread_list_filter_combined(client):
    """Filter by both topic and status."""
    resp = await client.get("/threads/?topic=iran-us&status=stale")
    assert resp.status_code == 200
    assert "Old Negotiations" in resp.text
    assert "Sanctions Escalation" not in resp.text


async def test_thread_list_empty_filter(client):
    """Filter that matches nothing."""
    resp = await client.get("/threads/?topic=nonexistent")
    assert resp.status_code == 200
    # Should render but with no threads in content
    assert "Sanctions Escalation" not in resp.text


# ── Thread detail ──


async def test_thread_detail_with_convergence(client):
    resp = await client.get("/threads/sanctions-escalation")
    assert resp.status_code == 200
    assert "What Sources Agree On" in resp.text
    assert "sanctions are real" in resp.text.lower()


async def test_thread_detail_with_divergence(client):
    resp = await client.get("/threads/sanctions-escalation")
    assert resp.status_code == 200
    assert "Where They Disagree" in resp.text


async def test_thread_detail_with_causal_links(client):
    resp = await client.get("/threads/sanctions-escalation")
    assert resp.status_code == 200
    # Causal links section should appear
    assert "Causal" in resp.text or "causal" in resp.text.lower() or "response_to" in resp.text


async def test_thread_detail_shows_events(client):
    resp = await client.get("/threads/sanctions-escalation")
    assert resp.status_code == 200
    assert "Sanctions announced" in resp.text
    assert "Iran responds" in resp.text


async def test_thread_detail_404(client):
    resp = await client.get("/threads/nonexistent-thread")
    assert resp.status_code == 404


async def test_thread_detail_shows_projection_items(client):
    """Projection items citing this thread should appear."""
    resp = await client.get("/threads/sanctions-escalation")
    assert resp.status_code == 200
    assert "More sanctions likely" in resp.text or "Forward" in resp.text


# ── Empty thread app ──


async def test_thread_list_empty(tmp_path):
    """Thread list with no threads at all."""
    (tmp_path / "config.yaml").write_text("preset: balanced\ntopics:\n  - name: test\n")
    app = create_app(tmp_path / "test.db", data_dir=tmp_path)
    store = KnowledgeStore(tmp_path / "test.db")
    await store.initialize()
    app.state.store = store

    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.get("/threads/")
    assert resp.status_code == 200
    await store.close()
