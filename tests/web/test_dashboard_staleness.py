"""Tests for dashboard staleness badges, empty state guidance, and onboarding."""

import pytest
from datetime import date, timedelta
from httpx import AsyncClient, ASGITransport

from nexus.engine.knowledge.store import KnowledgeStore
from nexus.web.app import create_app
from nexus.web.routes.dashboard import briefing_age_badge


# ── Unit tests for staleness helper ──


class TestBriefingAgeBadge:
    def test_fresh_briefing_same_day(self):
        today = date(2026, 3, 17)
        level, label = briefing_age_badge(today, today)
        assert level == "fresh"
        assert label == "Today"

    def test_yesterday_briefing(self):
        today = date(2026, 3, 17)
        level, label = briefing_age_badge(date(2026, 3, 16), today)
        assert level == "recent"
        assert label == "Yesterday"

    def test_two_day_old_briefing(self):
        today = date(2026, 3, 17)
        level, label = briefing_age_badge(date(2026, 3, 15), today)
        assert level == "stale"
        assert "2 days ago" in label

    def test_week_old_briefing(self):
        today = date(2026, 3, 17)
        level, label = briefing_age_badge(date(2026, 3, 10), today)
        assert level == "stale"
        assert "7 days ago" in label


# ── Integration tests for dashboard rendering ──


@pytest.fixture
async def empty_app(tmp_path):
    """App with config but no pipeline data — simulates first-run."""
    (tmp_path / "config.yaml").write_text("preset: balanced\ntopics:\n  - name: test\n")
    app = create_app(tmp_path / "test.db", data_dir=tmp_path)
    store = KnowledgeStore(tmp_path / "test.db")
    await store.initialize()
    app.state.store = store
    yield app
    await store.close()


@pytest.fixture
async def seeded_app(tmp_path):
    """App with a briefing file and pipeline run — simulates normal use."""
    (tmp_path / "config.yaml").write_text("preset: balanced\ntopics:\n  - name: test\n")
    briefing_dir = tmp_path / "artifacts" / "briefings"
    briefing_dir.mkdir(parents=True)
    today = date.today()
    (briefing_dir / f"{today.isoformat()}.md").write_text("# Test Briefing\nContent here.")

    app = create_app(tmp_path / "test.db", data_dir=tmp_path)
    store = KnowledgeStore(tmp_path / "test.db")
    await store.initialize()
    app.state.store = store
    yield app
    await store.close()


@pytest.fixture
async def stale_app(tmp_path):
    """App with a 3-day-old briefing — simulates stale data."""
    (tmp_path / "config.yaml").write_text("preset: balanced\ntopics:\n  - name: test\n")
    briefing_dir = tmp_path / "artifacts" / "briefings"
    briefing_dir.mkdir(parents=True)
    stale_date = date.today() - timedelta(days=3)
    (briefing_dir / f"{stale_date.isoformat()}.md").write_text("# Stale Briefing")

    app = create_app(tmp_path / "test.db", data_dir=tmp_path)
    store = KnowledgeStore(tmp_path / "test.db")
    await store.initialize()
    app.state.store = store
    yield app
    await store.close()


async def test_dashboard_shows_staleness_badge_fresh(seeded_app):
    """Fresh briefing shows green 'Today' badge."""
    transport = ASGITransport(app=seeded_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/")
    assert resp.status_code == 200
    assert "staleness-fresh" in resp.text


async def test_dashboard_shows_staleness_badge_stale(stale_app):
    """3-day-old briefing shows red stale badge."""
    transport = ASGITransport(app=stale_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/")
    assert resp.status_code == 200
    assert "staleness-stale" in resp.text
    assert "days ago" in resp.text


async def test_dashboard_onboarding_shown_on_first_visit(empty_app):
    """Dashboard with setup=complete but no data shows pipeline empty state, not onboarding."""
    transport = ASGITransport(app=empty_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/?setup=complete")
    assert resp.status_code == 200
    # Onboarding card only shows when data exists; empty state shows welcome
    assert "Welcome to Nexus" in resp.text


async def test_dashboard_onboarding_shown_when_data_exists(seeded_app):
    """Dashboard with setup=complete AND data shows onboarding card."""
    transport = ASGITransport(app=seeded_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/?setup=complete")
    assert resp.status_code == 200
    assert "onboarding-card" in resp.text


async def test_onboarding_dismiss(seeded_app):
    """POST /api/dismiss-onboarding sets cookie."""
    transport = ASGITransport(app=seeded_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/api/dismiss-onboarding")
    assert resp.status_code == 200
    assert "nexus_onboarding" in resp.headers.get("set-cookie", "")


# ── Empty state guidance on list pages ──


async def test_thread_list_empty_state_guidance(empty_app):
    """Thread list shows guidance when empty, not just 'No threads found'."""
    transport = ASGITransport(app=empty_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/threads/")
    assert resp.status_code == 200
    assert "pipeline" in resp.text.lower()  # Should mention pipeline


async def test_entity_list_empty_state_guidance(empty_app):
    """Entity list shows guidance when empty."""
    transport = ASGITransport(app=empty_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/explore/")
    assert resp.status_code == 200
    assert "pipeline" in resp.text.lower()
