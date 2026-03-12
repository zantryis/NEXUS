#!/usr/bin/env python3
"""Interactive thread consolidation script.

Identifies threads with high entity overlap and proposes merges.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nexus.engine.knowledge.store import KnowledgeStore


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


async def main():
    db_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/knowledge.db")
    threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 0.4
    dry_run = "--dry-run" in sys.argv

    print(f"DB: {db_path}")
    print(f"Threshold: {threshold}")
    print(f"Mode: {'DRY RUN' if dry_run else 'INTERACTIVE'}\n")

    store = KnowledgeStore(db_path)
    await store.initialize()

    try:
        threads = await store.get_active_threads()
        print(f"Found {len(threads)} active/emerging threads\n")

        # Build entity sets
        entity_sets = {}
        for t in threads:
            entities = {e.lower() for e in (t.get("key_entities") or [])}
            entity_sets[t["id"]] = entities

        # Find merge candidates
        merges = []
        seen = set()
        for i, t1 in enumerate(threads):
            if t1["id"] in seen:
                continue
            for t2 in threads[i + 1:]:
                if t2["id"] in seen:
                    continue
                overlap = jaccard(entity_sets[t1["id"]], entity_sets[t2["id"]])
                if overlap >= threshold:
                    # Keep the higher-significance thread
                    if t1["significance"] >= t2["significance"]:
                        keep, absorb = t1, t2
                    else:
                        keep, absorb = t2, t1
                    merges.append((keep, absorb, overlap))
                    seen.add(absorb["id"])

        if not merges:
            print("No merge candidates found.")
            return

        print(f"Found {len(merges)} merge candidates:\n")
        for i, (keep, absorb, overlap) in enumerate(merges):
            print(f"  [{i + 1}] KEEP: \"{keep['headline']}\" (sig {keep['significance']})")
            print(f"       ABSORB: \"{absorb['headline']}\" (sig {absorb['significance']})")
            print(f"       Overlap: {overlap:.2f}")
            print()

        if dry_run:
            print("Dry run — no changes made.")
            return

        confirm = input(f"Proceed with {len(merges)} merges? (y/N): ").strip().lower()
        if confirm != "y":
            print("Aborted.")
            return

        for keep, absorb, overlap in merges:
            print(f"Merging \"{absorb['headline']}\" → \"{keep['headline']}\"...")
            await store.merge_threads(keep["id"], absorb["id"])

        print(f"\nDone. Merged {len(merges)} thread pairs.")

    finally:
        await store.close()


if __name__ == "__main__":
    asyncio.run(main())
