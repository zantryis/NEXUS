"""Tests for SQLite transaction safety in KnowledgeStore."""

from datetime import date

import pytest

from nexus.engine.knowledge.events import Event
from nexus.engine.knowledge.store import KnowledgeStore


@pytest.fixture
async def store(tmp_path):
    s = KnowledgeStore(tmp_path / "test.db")
    await s.initialize()
    yield s
    await s.close()


def _make_event(summary: str, sig: int = 5) -> Event:
    return Event(
        date=date(2026, 3, 17),
        summary=summary,
        significance=sig,
        sources=[{"url": "http://example.com", "outlet": "test"}],
        entities=["TestEntity"],
    )


class TestAddEventsTransaction:
    @pytest.mark.asyncio
    async def test_add_events_all_or_nothing(self, store):
        """If an error occurs mid-batch, no events should be committed."""
        good_event = _make_event("Good event")
        bad_event = _make_event("Bad event")
        # Corrupt the bad event's sources to trigger an error during insert
        bad_event.sources = [{"INVALID_KEY_ONLY": True}]

        # Monkey-patch to make the second event's source insert fail
        original_execute = store.db.execute
        call_count = 0

        async def failing_execute(sql, params=None):
            nonlocal call_count
            if "INSERT INTO event_sources" in sql:
                call_count += 1
                if call_count == 2:
                    raise RuntimeError("Simulated DB error on second source insert")
            return await original_execute(sql, params)

        store.db.execute = failing_execute

        with pytest.raises(RuntimeError, match="Simulated DB error"):
            await store.add_events([good_event, bad_event], "test-topic")

        # Restore original execute for the count query
        store.db.execute = original_execute

        # No events should have been committed
        cursor = await store.db.execute("SELECT COUNT(*) FROM events WHERE topic_slug = 'test-topic'")
        row = await cursor.fetchone()
        assert row[0] == 0

    @pytest.mark.asyncio
    async def test_add_events_success_commits_all(self, store):
        """Normal case: all events in a batch are committed together."""
        events = [_make_event(f"Event {i}") for i in range(3)]
        ids = await store.add_events(events, "test-topic")

        assert len(ids) == 3
        cursor = await store.db.execute("SELECT COUNT(*) FROM events WHERE topic_slug = 'test-topic'")
        row = await cursor.fetchone()
        assert row[0] == 3


class TestIsPipelineRunningAtomic:
    @pytest.mark.asyncio
    async def test_is_pipeline_running_basic(self, store):
        """Returns True when a run is in progress."""
        await store.start_pipeline_run(["test"], trigger="manual")
        assert await store.is_pipeline_running() is True

    @pytest.mark.asyncio
    async def test_is_pipeline_running_after_completion(self, store):
        """Returns False after run completes."""
        run_id = await store.start_pipeline_run(["test"], trigger="manual")
        await store.complete_pipeline_run(run_id)
        assert await store.is_pipeline_running() is False

    @pytest.mark.asyncio
    async def test_is_pipeline_running_no_side_effects_on_active_run(self, store):
        """Checking running status should not modify active (non-stale) runs."""
        run_id = await store.start_pipeline_run(["test"], trigger="manual")

        # Call is_pipeline_running — should not mark a fresh run as stale
        result = await store.is_pipeline_running()
        assert result is True

        # The run should still be 'running', not 'failed'
        cursor = await store.db.execute(
            "SELECT status FROM pipeline_runs WHERE id = ?", (run_id,)
        )
        row = await cursor.fetchone()
        assert row[0] == "running"
