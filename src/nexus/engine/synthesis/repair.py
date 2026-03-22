"""Thread repair and historical backfill utilities."""

from __future__ import annotations

import json
import logging
from datetime import date, datetime

from nexus.engine.knowledge.pages import generate_thread_deepdive
from nexus.engine.projection.service import (
    backfill_thread_snapshots,
    capture_status_transition_snapshots,
    hydrate_synthesis_threads,
    save_projection_page,
)
from nexus.engine.synthesis.knowledge import TopicSynthesis
from nexus.engine.synthesis.threads import find_merge_candidates
from nexus.llm.client import LLMClient

logger = logging.getLogger(__name__)


def _day_start(raw_date: str | None) -> str | None:
    if not raw_date:
        return None
    return f"{raw_date}T00:00:00"


def _day_end(raw_date: str | None) -> str | None:
    if not raw_date:
        return None
    return f"{raw_date}T23:59:59"


def _max_timestamp(*values: str | None) -> str | None:
    valid = [value for value in values if value]
    return max(valid) if valid else None


async def _normalize_legacy_merged_threads(store) -> int:
    cursor = await store.db.execute(
        "UPDATE threads SET status = 'merged', updated_at = ? "
        "WHERE status = 'resolved' AND merged_into_id IS NOT NULL",
        (datetime.now().isoformat(),),
    )
    await store.db.commit()
    return cursor.rowcount


async def _recalculate_thread_timestamps(store) -> int:
    cursor = await store.db.execute(
        "SELECT id, status, created_at, updated_at FROM threads WHERE status != 'merged'"
    )
    rows = await cursor.fetchall()
    updated = 0
    for row in rows:
        thread_id, status, created_at, updated_at = row
        bounds_cursor = await store.db.execute(
            "SELECT MIN(ev.date), MAX(ev.date) "
            "FROM thread_events te JOIN events ev ON te.event_id = ev.id "
            "WHERE te.thread_id = ?",
            (thread_id,),
        )
        bounds = await bounds_cursor.fetchone()
        first_event_date = bounds[0] if bounds else None
        latest_event_date = bounds[1] if bounds else None

        created_candidate = min(
            value for value in (created_at, _day_start(first_event_date)) if value
        ) if (created_at or first_event_date) else created_at
        latest_event_ts = _day_end(latest_event_date)
        if status in ("stale", "resolved"):
            updated_candidate = _max_timestamp(updated_at, latest_event_ts)
        else:
            updated_candidate = latest_event_ts or updated_at

        if created_candidate == created_at and updated_candidate == updated_at:
            continue

        await store.db.execute(
            "UPDATE threads SET created_at = ?, updated_at = ? WHERE id = ?",
            (created_candidate, updated_candidate, thread_id),
        )
        updated += 1

    if updated:
        await store.db.commit()
    return updated


async def _hydrate_all_syntheses(store) -> int:
    cursor = await store.db.execute(
        "SELECT topic_slug, date, data_json FROM syntheses ORDER BY topic_slug, date"
    )
    rows = await cursor.fetchall()
    refreshed = 0
    for topic_slug, raw_date, data_json in rows:
        try:
            raw = json.loads(data_json)
        except (json.JSONDecodeError, TypeError):
            continue
        run_date = date.fromisoformat(raw_date)
        synthesis = await hydrate_synthesis_threads(
            store,
            TopicSynthesis(**raw),
            topic_slug=topic_slug,
            as_of=run_date,
        )
        await store.replace_synthesis(synthesis.model_dump(mode="json"), topic_slug, run_date)
        refreshed += 1
    return refreshed


async def _refresh_projection_pages(store) -> int:
    cursor = await store.db.execute(
        "SELECT DISTINCT topic_slug FROM projections ORDER BY topic_slug"
    )
    topic_slugs = [row[0] for row in await cursor.fetchall()]
    refreshed = 0
    for topic_slug in topic_slugs:
        projection = await store.get_latest_projection(topic_slug)
        if projection is None:
            continue
        await save_projection_page(store, projection)
        refreshed += 1
    return refreshed


async def _refresh_thread_pages(store, llm: LLMClient | None) -> dict:
    cursor = await store.db.execute(
        "SELECT slug FROM pages WHERE slug LIKE 'thread:%' ORDER BY slug"
    )
    page_slugs = [row[0] for row in await cursor.fetchall()]
    deleted = 0
    regenerated = 0

    for page_slug in page_slugs:
        thread_slug = page_slug.split(":", 1)[1]
        thread = await store.get_thread(thread_slug)
        if thread is None or thread["status"] == "merged":
            await store.db.execute("DELETE FROM pages WHERE slug = ?", (page_slug,))
            deleted += 1
            continue
        if llm is None:
            continue

        events = await store.get_events_for_thread(thread["id"])
        convergence = await store.get_convergence_for_thread(thread["id"])
        divergence = await store.get_divergence_for_thread(thread["id"])
        payload = await generate_thread_deepdive(
            llm,
            thread,
            [event.model_dump(mode="json") for event in events],
            convergence,
            divergence,
        )
        await store.save_page(
            payload["slug"],
            payload["title"],
            payload["page_type"],
            payload["content_md"],
            payload["topic_slug"],
            payload["ttl_days"],
            payload["prompt_hash"],
        )
        regenerated += 1

    if deleted:
        await store.db.commit()
    return {"deleted": deleted, "regenerated": regenerated}


async def repair_thread_hygiene(
    store,
    llm: LLMClient | None = None,
    *,
    run_date: date | None = None,
) -> dict:
    """Repair duplicate thread history and rebuild thread-dependent artifacts."""
    repair_date = run_date or date.today()
    legacy_merged = await _normalize_legacy_merged_threads(store)

    merge_rounds = 0
    merged_pairs = 0
    while True:
        threads = await store.get_all_threads()
        if len(threads) < 2:
            break
        pairs = await find_merge_candidates(threads, llm)
        if not pairs:
            break
        merge_rounds += 1
        for keep_id, absorb_id in pairs:
            await store.merge_threads(keep_id, absorb_id)
        merged_pairs += len(pairs)

    orphaned = await store.purge_empty_threads(dry_run=False)
    timestamps_recalculated = await _recalculate_thread_timestamps(store)
    stale_count = await store.mark_stale_threads(reference_date=repair_date)

    await store.db.execute("DELETE FROM thread_snapshots")
    await store.db.execute("DELETE FROM causal_links")
    await store.db.commit()

    rebuild_stats = await backfill_thread_snapshots(store)
    transition_snapshots = await capture_status_transition_snapshots(store, repair_date)
    syntheses_refreshed = await _hydrate_all_syntheses(store)
    projection_pages = await _refresh_projection_pages(store)
    thread_pages = await _refresh_thread_pages(store, llm)

    return {
        "legacy_merged": legacy_merged,
        "merge_rounds": merge_rounds,
        "merged_pairs": merged_pairs,
        "purged_empty_threads": orphaned["purged"],
        "timestamps_recalculated": timestamps_recalculated,
        "stale_marked": stale_count,
        "snapshots_rebuilt": rebuild_stats["snapshots"],
        "causal_links_rebuilt": rebuild_stats["causal_links"],
        "transition_snapshots": transition_snapshots,
        "syntheses_refreshed": syntheses_refreshed,
        "projection_pages_refreshed": projection_pages,
        "thread_pages_deleted": thread_pages["deleted"],
        "thread_pages_regenerated": thread_pages["regenerated"],
    }
