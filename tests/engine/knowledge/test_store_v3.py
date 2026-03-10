"""Tests for v3 store methods: breaking_alerts + feedback."""

import pytest
from pathlib import Path

from nexus.engine.knowledge.store import KnowledgeStore


@pytest.fixture
async def store(tmp_path):
    s = KnowledgeStore(tmp_path / "test.db")
    await s.initialize()
    yield s
    await s.close()


async def test_add_breaking_alert(store):
    row_id = await store.add_breaking_alert(
        "abc123", "Major event happened", "https://reuters.com/1", 9,
    )
    assert row_id > 0


async def test_is_alerted_true(store):
    await store.add_breaking_alert("hash1", "Test", "https://a.com", 8)
    assert await store.is_alerted("hash1") is True


async def test_is_alerted_false(store):
    assert await store.is_alerted("nonexistent") is False


async def test_breaking_alert_dedup(store):
    """Duplicate headline_hash should be ignored (INSERT OR IGNORE)."""
    await store.add_breaking_alert("dup", "First", "https://a.com", 8)
    await store.add_breaking_alert("dup", "Second", "https://b.com", 9)
    assert await store.is_alerted("dup") is True


async def test_add_feedback(store):
    row_id = await store.add_feedback("2026-03-10", "up", "Great briefing!")
    assert row_id > 0


async def test_get_feedback_by_date(store):
    await store.add_feedback("2026-03-10", "up", "Good")
    await store.add_feedback("2026-03-10", "down", "Too long")
    await store.add_feedback("2026-03-09", "up")

    fb = await store.get_feedback("2026-03-10")
    assert len(fb) == 2
    assert all(f["briefing_date"] == "2026-03-10" for f in fb)


async def test_get_feedback_all(store):
    await store.add_feedback("2026-03-10", "up")
    await store.add_feedback("2026-03-09", "down")

    fb = await store.get_feedback()
    assert len(fb) == 2


async def test_feedback_fields(store):
    await store.add_feedback("2026-03-10", "up", "Nice work")
    fb = await store.get_feedback("2026-03-10")
    assert fb[0]["rating"] == "up"
    assert fb[0]["comment"] == "Nice work"
    assert fb[0]["created_at"] is not None
