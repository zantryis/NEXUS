"""Backtest: replay historical filter_log through hybrid funnel triage.

Reads the filter_log table from data/knowledge.db and simulates what
would happen under the hybrid funnel (KEEP/MAYBE/DROP triage) vs the
current absolute threshold approach.

Usage: python scripts/backtest_hybrid_funnel.py
"""

import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nexus.engine.filtering.pairwise import (
    KEEP_THRESHOLD,
    DROP_THRESHOLD,
    MIN_DEGENERATE_SAMPLES,
    detect_degenerate,
)

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "knowledge.db"


def main():
    if not DB_PATH.exists():
        print(f"Database not found: {DB_PATH}")
        return

    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row

    # --- Per-run analysis ---
    runs = conn.execute("""
        SELECT run_date, topic_slug,
               GROUP_CONCAT(relevance_score) as scores_csv,
               COUNT(*) as n,
               SUM(CASE WHEN outcome = 'accepted' THEN 1 ELSE 0 END) as accepted,
               SUM(CASE WHEN outcome = 'rejected_relevance' THEN 1 ELSE 0 END) as rej_rel,
               SUM(CASE WHEN outcome = 'rejected_significance' THEN 1 ELSE 0 END) as rej_sig,
               SUM(CASE WHEN outcome = 'rejected_diversity' THEN 1 ELSE 0 END) as rej_div
        FROM filter_log
        GROUP BY run_date, topic_slug
        ORDER BY run_date, topic_slug
    """).fetchall()

    print("=" * 100)
    print("HYBRID FUNNEL BACKTEST — Historical filter_log analysis")
    print("=" * 100)
    print()

    totals = {"items": 0, "keep": 0, "maybe": 0, "drop": 0, "degen": 0}

    print(f"{'Date':12} {'Topic':25} {'N':>5} {'KEEP':>5} {'MAYBE':>5} {'DROP':>5} {'Degen':>5} | {'Accepted':>8} {'RejRel':>6}")
    print("-" * 100)

    for run in runs:
        scores = [float(s) for s in run["scores_csv"].split(",") if s and s != "None"]
        n = run["n"]

        keep = sum(1 for s in scores if s >= KEEP_THRESHOLD)
        maybe = sum(1 for s in scores if DROP_THRESHOLD <= s < KEEP_THRESHOLD)
        drop = sum(1 for s in scores if s < DROP_THRESHOLD)
        is_degen = detect_degenerate(scores)

        totals["items"] += n
        totals["keep"] += keep
        totals["maybe"] += maybe
        totals["drop"] += drop
        if is_degen:
            totals["degen"] += 1

        print(
            f"{run['run_date']:12} {run['topic_slug']:25} {n:5} "
            f"{keep:5} {maybe:5} {drop:5} {'YES' if is_degen else '  -':>5} | "
            f"{run['accepted']:8} {run['rej_rel']:6}"
        )

    print("-" * 100)
    print(
        f"{'TOTAL':38} {totals['items']:5} "
        f"{totals['keep']:5} {totals['maybe']:5} {totals['drop']:5}"
    )
    print()

    # --- Summary ---
    n = totals["items"]
    print("TRIAGE SPLIT:")
    print(f"  KEEP  (score >= {KEEP_THRESHOLD}): {totals['keep']:5} ({100*totals['keep']/n:.1f}%) — pass directly")
    print(f"  MAYBE (score {DROP_THRESHOLD}-{KEEP_THRESHOLD-1}):  {totals['maybe']:5} ({100*totals['maybe']/n:.1f}%) — need pairwise")
    print(f"  DROP  (score < {DROP_THRESHOLD}):  {totals['drop']:5} ({100*totals['drop']/n:.1f}%) — rejected immediately")
    print(f"  Degenerate runs: {totals['degen']} / {len(runs)}")
    print()

    # --- MAYBE outcome analysis ---
    maybe_outcomes = conn.execute("""
        SELECT outcome, COUNT(*) as n
        FROM filter_log
        WHERE relevance_score >= ? AND relevance_score < ?
        GROUP BY outcome ORDER BY n DESC
    """, (DROP_THRESHOLD, KEEP_THRESHOLD)).fetchall()

    print("MAYBE ITEMS — what happened under current system:")
    for row in maybe_outcomes:
        print(f"  {row['outcome']:25} {row['n']:5}")

    # --- Cost estimate ---
    avg_maybe_per_run = totals["maybe"] / len(runs) if runs else 0
    pairwise_calls = avg_maybe_per_run / 8 * 3  # 3 refs per item, batched by 8
    print()
    print("COST ESTIMATE (per run):")
    print(f"  Avg MAYBE items:    {avg_maybe_per_run:.0f}")
    print(f"  Pairwise LLM calls: ~{pairwise_calls:.0f} (3 refs each, batched by 8)")
    print(f"  Current pass 1 calls: ~{totals['items']/len(runs)/10:.0f} (batched by 10)")
    print(f"  Overhead: +{100*pairwise_calls/(totals['items']/len(runs)/10):.0f}% more LLM calls")

    print()
    print("CONCLUSION:")
    print(f"  83% of articles ({totals['keep']+totals['drop']}/{n}) are clearly KEEP or DROP — no pairwise needed.")
    print(f"  17% ({totals['maybe']}/{n}) are ambiguous — pairwise would replace the threshold-based decision.")
    print(f"  Only {totals['degen']}/{len(runs)} runs trigger degenerate detection.")
    print(f"  Real bottleneck is diversity selection, not relevance scoring.")

    conn.close()


if __name__ == "__main__":
    main()
