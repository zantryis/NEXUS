#!/usr/bin/env python3
"""One-time script to populate entity thumbnails from Wikipedia."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nexus.engine.knowledge.store import KnowledgeStore
from nexus.web.thumbnails import populate_thumbnails


async def main():
    db_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/knowledge.db")
    print(f"Populating thumbnails from: {db_path}")

    store = KnowledgeStore(db_path)
    await store.initialize()

    try:
        stats = await populate_thumbnails(store)
        print(f"Done: {stats}")
    finally:
        await store.close()


if __name__ == "__main__":
    asyncio.run(main())
