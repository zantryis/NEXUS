"""Tests for pipeline_runs tracking in KnowledgeStore."""

import pytest

from nexus.engine.knowledge.store import KnowledgeStore


@pytest.fixture
async def store(tmp_path):
    """Create a KnowledgeStore with schema initialized."""
    s = KnowledgeStore(tmp_path / "test.db")
    await s.initialize()
    yield s
    await s.close()


async def test_start_and_complete_pipeline_run(store):
    """Happy path: start a run, complete it with stats."""
    run_id = await store.start_pipeline_run(["space", "cyber"], trigger="manual")
    assert run_id > 0

    await store.complete_pipeline_run(run_id, article_count=42, event_count=8, cost_usd=0.05)

    last = await store.get_last_pipeline_run()
    assert last is not None
    assert last["status"] == "completed"
    assert last["topics"] == ["space", "cyber"]
    assert last["article_count"] == 42
    assert last["event_count"] == 8
    assert last["cost_usd"] == 0.05
    assert last["trigger"] == "manual"
    assert last["completed_at"] is not None
    assert last["error"] is None


async def test_complete_pipeline_run_records_skipped_topics(store):
    """Skipped topic metadata should survive completion and readback."""
    run_id = await store.start_pipeline_run(["alpha", "beta"], trigger="scheduled")

    await store.complete_pipeline_run(
        run_id,
        article_count=12,
        event_count=4,
        cost_usd=0.02,
        skipped_topics=[{"name": "beta", "reason": "no sources configured"}],
    )

    last = await store.get_last_pipeline_run()
    assert last is not None
    assert last["status"] == "completed"
    assert last["skipped_topics"] == [
        {"name": "beta", "reason": "no sources configured"},
    ]


async def test_fail_pipeline_run(store):
    """Failed run records error message."""
    run_id = await store.start_pipeline_run(["test"], trigger="scheduled")
    await store.fail_pipeline_run(run_id, "API rate limit exceeded")

    last = await store.get_last_pipeline_run()
    assert last["status"] == "failed"
    assert last["error"] == "API rate limit exceeded"
    assert last["trigger"] == "scheduled"


async def test_get_last_pipeline_run_returns_most_recent(store):
    """Returns the most recent run, not an older one."""
    run1 = await store.start_pipeline_run(["old"], trigger="scheduled")
    await store.complete_pipeline_run(run1)

    run2 = await store.start_pipeline_run(["new"], trigger="manual")
    await store.complete_pipeline_run(run2, article_count=10)

    last = await store.get_last_pipeline_run()
    assert last["topics"] == ["new"]
    assert last["article_count"] == 10


async def test_get_last_pipeline_run_empty(store):
    """Returns None when no pipeline runs exist."""
    last = await store.get_last_pipeline_run()
    assert last is None


async def test_is_pipeline_running_true(store):
    """Detects active pipeline run."""
    await store.start_pipeline_run(["test"], trigger="manual")
    assert await store.is_pipeline_running() is True


async def test_is_pipeline_running_false_when_completed(store):
    """Completed runs are not counted as running."""
    run_id = await store.start_pipeline_run(["test"], trigger="manual")
    await store.complete_pipeline_run(run_id)
    assert await store.is_pipeline_running() is False


async def test_is_pipeline_running_false_when_failed(store):
    """Failed runs are not counted as running."""
    run_id = await store.start_pipeline_run(["test"], trigger="manual")
    await store.fail_pipeline_run(run_id, "crash")
    assert await store.is_pipeline_running() is False


async def test_stale_running_cleanup(store):
    """Runs older than 3h are auto-marked as failed."""
    # Insert a stale run directly (4 hours old — exceeds 3h threshold)
    await store.db.execute(
        "INSERT INTO pipeline_runs (started_at, status, topics, trigger) "
        "VALUES (datetime('now', '-4 hours'), 'running', '[]', 'manual')"
    )
    await store.db.commit()

    # is_pipeline_running cleans up stale runs
    assert await store.is_pipeline_running() is False

    last = await store.get_last_pipeline_run()
    assert last["status"] == "failed"
    assert "Stale" in last["error"]


async def test_running_within_threshold_not_stale(store):
    """Runs under 3h old should NOT be marked as stale."""
    await store.db.execute(
        "INSERT INTO pipeline_runs (started_at, status, topics, trigger) "
        "VALUES (datetime('now', '-2 hours'), 'running', '[]', 'manual')"
    )
    await store.db.commit()

    # 2h-old run should still be considered running
    assert await store.is_pipeline_running() is True
