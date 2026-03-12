#!/usr/bin/env python3
"""One-time backfill: populate thread_events from stored synthesis JSON.

Usage:
    python scripts/backfill_thread_events.py [--db-path PATH]

Parses syntheses.data_json to recover which events belong to which threads,
matches them by (summary, date, topic_slug), and populates thread_events.
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pathlib import Path as _Path
from nexus.engine.knowledge.store import KnowledgeStore

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


async def backfill(db_path: str) -> None:
    store = KnowledgeStore(_Path(db_path))
    await store.initialize()

    try:
        # Get all syntheses
        cursor = await store.db.execute(
            "SELECT topic_slug, date, data_json FROM syntheses ORDER BY date"
        )
        syntheses = await cursor.fetchall()
        logger.info(f"Found {len(syntheses)} synthesis records")

        # Get all threads
        cursor = await store.db.execute("SELECT id, slug, headline FROM threads")
        threads_by_slug = {}
        threads_by_headline = {}
        for row in await cursor.fetchall():
            threads_by_slug[row[1]] = row[0]
            threads_by_headline[row[2]] = row[0]

        total_links = 0
        for topic_slug, syn_date, data_json in syntheses:
            try:
                data = json.loads(data_json)
            except (json.JSONDecodeError, TypeError):
                continue

            syn_threads = data.get("threads", [])
            for t in syn_threads:
                headline = t.get("headline", "")
                events = t.get("events", [])
                if not events:
                    continue

                # Find thread ID by headline or slug
                tid = threads_by_headline.get(headline)
                if not tid:
                    # Try slug-based lookup
                    slug = t.get("slug")
                    if slug:
                        tid = threads_by_slug.get(slug)
                if not tid:
                    logger.debug(f"  No thread found for: {headline[:60]}")
                    continue

                # Match events to DB rows
                event_ids = []
                for ev in events:
                    summary = ev.get("summary", "")
                    ev_date = ev.get("date", "")
                    if summary and ev_date:
                        eid = await store.find_event_id(summary, ev_date, topic_slug)
                        if eid:
                            event_ids.append(eid)

                if event_ids:
                    await store.link_thread_events(tid, event_ids)
                    total_links += len(event_ids)
                    logger.info(
                        f"  Linked {len(event_ids)} events to thread "
                        f"'{headline[:50]}' (topic={topic_slug})"
                    )

        # Report final state
        cursor = await store.db.execute("SELECT COUNT(*) FROM thread_events")
        count = (await cursor.fetchone())[0]
        logger.info(f"Done. {total_links} links created. thread_events now has {count} rows.")

    finally:
        await store.close()


def main():
    parser = argparse.ArgumentParser(description="Backfill thread_events table")
    parser.add_argument(
        "--db-path", default="data/knowledge.db",
        help="Path to knowledge.db (default: data/knowledge.db)",
    )
    args = parser.parse_args()
    asyncio.run(backfill(args.db_path))


if __name__ == "__main__":
    main()
