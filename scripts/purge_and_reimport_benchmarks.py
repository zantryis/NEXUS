"""Purge stale benchmark + kalshi_aligned data and reimport from report JSONs.

Usage:
    python scripts/purge_and_reimport_benchmarks.py [--dry-run]
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


async def main(dry_run: bool = False):
    from nexus.engine.knowledge.store import KnowledgeStore

    db_path = Path("data/knowledge.db")
    if not db_path.exists():
        logger.error("data/knowledge.db not found in current directory")
        return

    store = KnowledgeStore(db_path)
    await store.initialize()

    # ── Step 1: Purge stale data ──
    logger.info("=== Phase 1: Purge stale forecast data ===")

    for target in ("kalshi_benchmark", "kalshi_aligned"):
        result = await store.purge_forecast_runs(
            target_variable=target, dry_run=dry_run,
        )
        action = "Would delete" if dry_run else "Deleted"
        runs = result.get("runs_deleted") or result.get("runs_found", 0)
        questions = result.get("questions_deleted") or result.get("questions_found", 0)
        resolutions = result.get("resolutions_deleted", 0)
        mappings = result.get("mappings_deleted", 0)
        scenarios = result.get("scenarios_deleted", 0)
        logger.info(
            "%s %s: %d runs, %d questions, %d resolutions, %d mappings, %d scenarios",
            action, target, runs, questions, resolutions, mappings, scenarios,
        )

    if dry_run:
        logger.info("Dry run complete. Re-run without --dry-run to execute.")
        await store.close()
        return

    # ── Step 2: Reimport from report JSONs ──
    logger.info("=== Phase 2: Reimport benchmarks from report JSONs ===")

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "build_kalshi_fixture",
        Path(__file__).parent / "build_kalshi_fixture.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ingest_benchmark_results = mod.ingest_benchmark_results

    reports = sorted(Path("data/benchmarks").glob("kalshi_engine_comparison*.json"))
    if not reports:
        logger.warning("No report JSONs found in data/benchmarks/")
        await store.close()
        return

    for report_path in reports:
        label = report_path.stem.replace("kalshi_engine_comparison", "").strip("_") or "v1"
        logger.info("Importing %s (label=%s)", report_path.name, label)
        await ingest_benchmark_results(store, report_path, run_label=label)

    # ── Step 3: Verify ──
    logger.info("=== Phase 3: Verification ===")
    rows = await store.db.execute_fetchall(
        "SELECT target_variable, COUNT(*) FROM forecast_questions GROUP BY target_variable"
    )
    for row in rows:
        logger.info("  %s: %d questions", row[0], row[1])

    # Check resolution dates aren't all 2026-01-01
    bench_dates = await store.db.execute_fetchall(
        "SELECT resolution_date, COUNT(*) FROM forecast_questions "
        "WHERE target_variable = 'kalshi_benchmark' "
        "GROUP BY resolution_date ORDER BY COUNT(*) DESC LIMIT 10"
    )
    logger.info("  Benchmark resolution_date distribution (top 10):")
    for row in bench_dates:
        logger.info("    %s: %d", row[0], row[1])

    # Check kalshi_implied is populated
    null_implied = await store.db.execute_fetchall(
        "SELECT COUNT(*) FROM forecast_questions "
        "WHERE target_variable = 'kalshi_benchmark' "
        "AND (target_metadata_json NOT LIKE '%kalshi_implied%' "
        "     OR target_metadata_json LIKE '%\"kalshi_implied\": null%')"
    )
    n_null = null_implied[0][0] if null_implied else 0
    logger.info("  Benchmark questions missing kalshi_implied: %d", n_null)

    await store.close()
    logger.info("Done.")


if __name__ == "__main__":
    dry = "--dry-run" in sys.argv
    asyncio.run(main(dry_run=dry))
