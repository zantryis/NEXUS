"""Tests for feed health tracking — polling success/failure, dead detection."""

import pytest

from nexus.engine.knowledge.store import KnowledgeStore


@pytest.fixture
async def store(tmp_path):
    s = KnowledgeStore(tmp_path / "test.db")
    await s.initialize()
    yield s
    await s.close()


async def test_record_success(store):
    """Successful poll increments total_polls and total_successes."""
    await store.record_feed_poll("https://example.com/rss", "topic-a", success=True)
    health = await store.get_feed_health("https://example.com/rss")
    assert health is not None
    assert health["total_polls"] == 1
    assert health["total_successes"] == 1
    assert health["consecutive_failures"] == 0
    assert health["status"] == "healthy"


async def test_record_failure(store):
    """Failed poll increments consecutive_failures."""
    await store.record_feed_poll("https://example.com/rss", "topic-a", success=False, error="timeout")
    health = await store.get_feed_health("https://example.com/rss")
    assert health["total_polls"] == 1
    assert health["total_successes"] == 0
    assert health["consecutive_failures"] == 1
    assert health["last_error"] == "timeout"
    assert health["status"] == "healthy"  # not degraded after 1 failure


async def test_success_resets_consecutive_failures(store):
    """A successful poll after failures resets the consecutive counter."""
    await store.record_feed_poll("https://x.com/rss", "t", success=False)
    await store.record_feed_poll("https://x.com/rss", "t", success=False)
    await store.record_feed_poll("https://x.com/rss", "t", success=True)
    health = await store.get_feed_health("https://x.com/rss")
    assert health["consecutive_failures"] == 0
    assert health["total_polls"] == 3
    assert health["total_successes"] == 1


async def test_degraded_after_3_failures(store):
    """Feed becomes degraded after 3 consecutive failures."""
    for _ in range(3):
        await store.record_feed_poll("https://x.com/rss", "t", success=False)
    health = await store.get_feed_health("https://x.com/rss")
    assert health["status"] == "degraded"


async def test_dead_after_5_failures(store):
    """Feed becomes dead after 5 consecutive failures."""
    for _ in range(5):
        await store.record_feed_poll("https://x.com/rss", "t", success=False)
    health = await store.get_feed_health("https://x.com/rss")
    assert health["status"] == "dead"


async def test_get_dead_feeds(store):
    """get_dead_feeds returns only dead feeds."""
    # One healthy feed
    await store.record_feed_poll("https://good.com/rss", "t", success=True)
    # One dead feed
    for _ in range(5):
        await store.record_feed_poll("https://bad.com/rss", "t", success=False)

    dead = await store.get_dead_feeds()
    assert len(dead) == 1
    assert dead[0]["source_url"] == "https://bad.com/rss"


async def test_get_all_feed_health(store):
    """get_all_feed_health returns health for all tracked feeds."""
    await store.record_feed_poll("https://a.com/rss", "t1", success=True)
    await store.record_feed_poll("https://b.com/rss", "t2", success=False)
    all_health = await store.get_all_feed_health()
    assert len(all_health) == 2
    urls = {h["source_url"] for h in all_health}
    assert urls == {"https://a.com/rss", "https://b.com/rss"}


async def test_revive_dead_feed_on_success(store):
    """A dead feed revives to healthy on a successful poll."""
    for _ in range(5):
        await store.record_feed_poll("https://x.com/rss", "t", success=False)
    assert (await store.get_feed_health("https://x.com/rss"))["status"] == "dead"

    await store.record_feed_poll("https://x.com/rss", "t", success=True)
    health = await store.get_feed_health("https://x.com/rss")
    assert health["status"] == "healthy"
    assert health["consecutive_failures"] == 0
