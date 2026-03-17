"""Backfill entity-entity relationships from all historical events.

Processes events in chronological order so invalidation works correctly.
Skips events that already have relationships extracted.
"""

import asyncio
import os
import sys

from dotenv import load_dotenv

load_dotenv()

from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from datetime import date

from nexus.config.models import ModelsConfig
from nexus.engine.knowledge.relationships import (
    extract_relationships_from_event,
    invalidate_contradicted_relationships,
)
from nexus.engine.knowledge.store import KnowledgeStore
from nexus.llm.client import LLMClient


async def main():
    db_path = Path("data/knowledge.db")
    if not db_path.exists():
        print(f"ERROR: {db_path} not found")
        return

    store = KnowledgeStore(db_path)
    await store.initialize()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY not set")
        return

    llm = LLMClient(ModelsConfig(), api_key=api_key)

    # Get events already backfilled
    cursor = await store.db.execute(
        "SELECT DISTINCT source_event_id FROM entity_relationships"
    )
    already_done = {row[0] for row in await cursor.fetchall()}
    print(f"Already backfilled: {len(already_done)} events")

    # Get all events with >=2 entities, ordered chronologically
    cursor = await store.db.execute("""
        SELECT e.id, e.date, e.summary, e.topic_slug,
               GROUP_CONCAT(ent.canonical_name, '||') as entity_names
        FROM events e
        JOIN event_entities ee ON e.id = ee.event_id
        JOIN entities ent ON ee.entity_id = ent.id
        GROUP BY e.id
        HAVING COUNT(ee.entity_id) >= 2
        ORDER BY e.date ASC, e.id ASC
    """)
    all_events = await cursor.fetchall()
    print(f"Total events with >=2 entities: {len(all_events)}")

    to_process = [(eid, edate, summary, topic, names)
                  for eid, edate, summary, topic, names in all_events
                  if eid not in already_done]
    print(f"Events to process: {len(to_process)}")

    if not to_process:
        print("Nothing to do.")
        await store.close()
        return

    total_extracted = 0
    total_invalidated = 0
    errors = 0

    for i, (event_id, event_date, summary, topic_slug, entity_names_str) in enumerate(to_process):
        entity_names = entity_names_str.split("||") if entity_names_str else []

        # Build a minimal Event object
        from nexus.engine.knowledge.events import Event
        event = Event(
            date=date.fromisoformat(event_date) if isinstance(event_date, str) else event_date,
            summary=summary,
            significance=5,
            relation_to_prior="",
            entities=entity_names,
            sources=[],
            event_id=event_id,
        )

        # Get existing active relationships for these entities
        existing_rels = []
        for name in entity_names[:5]:
            ent = await store.find_entity(name)
            if ent:
                rels = await store.get_active_relationships_for_entity(ent["id"])
                existing_rels.extend(rels)

        try:
            extracted = await extract_relationships_from_event(
                llm, event, existing_relationships=existing_rels,
            )
        except Exception as exc:
            errors += 1
            if errors <= 5:
                print(f"  ERROR [{event_date}] {summary[:50]}...: {exc}")
            continue

        if extracted:
            inv = await invalidate_contradicted_relationships(
                store, extracted, event.date,
            )
            total_invalidated += inv

            for rel in extracted:
                src = await store.find_entity(rel.source_entity)
                tgt = await store.find_entity(rel.target_entity)
                if src and tgt:
                    await store.save_entity_relationship({
                        "source_entity_id": src["id"],
                        "target_entity_id": tgt["id"],
                        "relation_type": rel.relation_type,
                        "evidence_text": rel.evidence_text,
                        "source_event_id": event_id,
                        "strength": rel.strength,
                        "valid_from": rel.valid_from.isoformat(),
                    })
                    total_extracted += 1

        # Progress every 20 events
        if (i + 1) % 20 == 0 or i == len(to_process) - 1:
            print(
                f"  [{i+1}/{len(to_process)}] +{total_extracted} rels, "
                f"{total_invalidated} invalidated, {errors} errors"
            )

    # Final stats
    print(f"\n=== BACKFILL COMPLETE ===")
    print(f"Extracted: {total_extracted}")
    print(f"Invalidated: {total_invalidated}")
    print(f"Errors: {errors}")

    cursor = await store.db.execute("SELECT COUNT(*) FROM entity_relationships")
    total = (await cursor.fetchone())[0]
    print(f"DB total relationships: {total}")

    cursor = await store.db.execute(
        "SELECT COUNT(*) FROM entity_relationships WHERE valid_until IS NOT NULL"
    )
    inv_total = (await cursor.fetchone())[0]
    print(f"DB invalidated: {inv_total}")

    cursor = await store.db.execute("""
        SELECT relation_type, COUNT(*)
        FROM entity_relationships
        GROUP BY relation_type
        ORDER BY COUNT(*) DESC
    """)
    print("\nRelationship types:")
    for row in await cursor.fetchall():
        print(f"  {row[0]}: {row[1]}")

    cursor = await store.db.execute("""
        SELECT MIN(valid_from), MAX(valid_from)
        FROM entity_relationships
    """)
    row = await cursor.fetchone()
    print(f"\nTemporal span: {row[0]} to {row[1]}")

    await store.close()


if __name__ == "__main__":
    asyncio.run(main())
