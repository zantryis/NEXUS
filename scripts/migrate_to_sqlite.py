#!/usr/bin/env python3
"""One-time migration: import existing YAML data into SQLite knowledge store.

Usage:
    python scripts/migrate_to_sqlite.py [--data-dir DATA_DIR]

Reads events and summaries from data/knowledge/{topic}/*.yaml
and imports them into data/knowledge.db. YAML files are kept as backup.
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nexus.engine.knowledge.events import Event, load_events
from nexus.engine.knowledge.compression import Summary, load_summaries
from nexus.engine.knowledge.store import KnowledgeStore

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


async def migrate(data_dir: Path) -> dict:
    """Import all YAML knowledge data into SQLite."""
    knowledge_dir = data_dir / "knowledge"
    if not knowledge_dir.exists():
        logger.warning(f"No knowledge directory found at {knowledge_dir}")
        return {"topics": 0, "events": 0, "summaries": 0}

    db_path = data_dir / "knowledge.db"
    store = KnowledgeStore(db_path)
    await store.initialize()

    stats = {"topics": 0, "events": 0, "weekly_summaries": 0, "monthly_summaries": 0}

    try:
        for topic_dir in sorted(knowledge_dir.iterdir()):
            if not topic_dir.is_dir():
                continue

            slug = topic_dir.name
            logger.info(f"Migrating topic: {slug}")
            stats["topics"] += 1

            # Import events
            events_path = topic_dir / "events.yaml"
            if events_path.exists():
                events = load_events(events_path)
                if events:
                    count = await store.import_events_from_yaml(events, slug)
                    stats["events"] += count
                    logger.info(f"  {count} events imported")

            # Import weekly summaries
            weekly_path = topic_dir / "weekly_summaries.yaml"
            if weekly_path.exists():
                summaries = load_summaries(weekly_path)
                if summaries:
                    count = await store.import_summaries_from_yaml(summaries, slug, "weekly")
                    stats["weekly_summaries"] += count
                    logger.info(f"  {count} weekly summaries imported")

            # Import monthly summaries
            monthly_path = topic_dir / "monthly_summaries.yaml"
            if monthly_path.exists():
                summaries = load_summaries(monthly_path)
                if summaries:
                    count = await store.import_summaries_from_yaml(summaries, slug, "monthly")
                    stats["monthly_summaries"] += count
                    logger.info(f"  {count} monthly summaries imported")

    finally:
        await store.close()

    logger.info(
        f"Migration complete: {stats['topics']} topics, "
        f"{stats['events']} events, "
        f"{stats['weekly_summaries']} weekly summaries, "
        f"{stats['monthly_summaries']} monthly summaries"
    )
    logger.info(f"Database: {db_path}")
    return stats


def main():
    parser = argparse.ArgumentParser(description="Migrate YAML knowledge data to SQLite")
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data"),
        help="Path to data directory (default: data/)",
    )
    args = parser.parse_args()

    asyncio.run(migrate(args.data_dir))


if __name__ == "__main__":
    main()
