"""Build a mid-range Kalshi benchmark fixture with candlestick history.

Discovers settled markets, syncs candlestick price history, builds a dataset
at 7/14/30-day cutoffs, filters to mid-range (0.10-0.90), then runs all
engines with multi-sample elicitation.

Usage:
    python scripts/build_kalshi_fixture.py [FLAGS]

Flags:
    --discover-only   Stop after discovering settled markets (Step 1)
    --sync-only       Stop after syncing candlestick history (Step 2)
    --skip-engines    Build dataset but don't run engines (Steps 1-3 only)
    --engines-only    Skip data collection, load saved fixture, run engines only
                      (requires prior run to have saved data/fixtures/)
    --ingest          Ingest existing benchmark results into the forecast DB
                      (reads from data/benchmarks/kalshi_engine_comparison_v2_*.json)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from datetime import date, datetime, time as dtime, timedelta, timezone
from pathlib import Path
from statistics import median

# Load .env before anything else
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nexus.config.models import KalshiBenchmarkConfig
from nexus.engine.projection.kalshi import (
    KalshiClient,
    KalshiLedger,
    _coerce_float,
    derive_series_ticker,
)
from nexus.engine.projection.kalshi_benchmark import (
    BenchmarkEngine,
    BenchmarkQuestion,
    BenchmarkReport,
    MarketBaselineEngine,
    SettledMarket,
    build_benchmark_dataset,
    discover_settled_markets,
    load_benchmark_dataset,
    run_benchmark,
    save_benchmark_dataset,
    save_benchmark_report,
)

logger = logging.getLogger(__name__)

BENCH_DIR = Path("data/benchmarks")
FIXTURE_DIR = Path("data/fixtures")
LEDGER_PATH = BENCH_DIR / "kalshi.sqlite"


# ── Multi-sample wrapper ──────────────────────────────────────────────


class MultiSampleWrapper:
    """Wraps a BenchmarkEngine, calls predict_probability N times, returns median."""

    def __init__(self, engine: BenchmarkEngine, n: int = 3):
        self.engine = engine
        self.n = n
        self.engine_name = f"{engine.engine_name}_x{n}"

    async def predict_probability(
        self,
        question: str,
        *,
        llm=None,
        store=None,
        market_prob: float | None = None,
        as_of: date | None = None,
    ) -> float:
        results = await asyncio.gather(*[
            self.engine.predict_probability(
                question, llm=llm, store=store,
                market_prob=market_prob, as_of=as_of,
            )
            for _ in range(self.n)
        ])
        return sorted(results)[self.n // 2]  # median


# ── Candlestick sync ──────────────────────────────────────────────────


async def sync_candlesticks_for_markets(
    ledger: KalshiLedger,
    client: KalshiClient,
    markets: list[SettledMarket],
    *,
    delay: float = 0.3,
) -> dict:
    """Fetch candlestick history for each settled market and store snapshots.

    Returns stats dict with counts.
    """
    total_synced = 0
    total_snapshots = 0
    total_errors = 0
    start_date = date(2024, 1, 1)
    end_date = date.today()

    for i, market in enumerate(markets):
        series = derive_series_ticker(market.event_ticker)
        if not series:
            total_errors += 1
            continue

        try:
            candles = await client.fetch_candlesticks(
                series_ticker=series,
                ticker=market.ticker,
                start=start_date,
                end=end_date,
            )
        except Exception as exc:
            logger.debug("Candlestick fetch failed for %s: %s", market.ticker, exc)
            total_errors += 1
            candles = []

        if candles:
            for candle in candles:
                end_ts = candle.get("end_period_ts") or candle.get("end_ts") or candle.get("ts")
                if end_ts is None:
                    continue
                captured_at = datetime.fromtimestamp(int(end_ts), tz=timezone.utc)

                # Candlestick structure: price/yes_bid/yes_ask are nested dicts
                # with close_dollars, high_dollars, etc.
                price_obj = candle.get("price") or {}
                close_str = (
                    price_obj.get("close_dollars") if isinstance(price_obj, dict)
                    else candle.get("close")
                )
                implied = None
                if close_str is not None:
                    try:
                        implied = round(float(close_str), 4)
                    except (TypeError, ValueError):
                        pass

                def _nested_close(field_name: str):
                    obj = candle.get(field_name)
                    if isinstance(obj, dict):
                        return _coerce_float(obj.get("close_dollars"))
                    return _coerce_float(obj)

                snapshot = {
                    "captured_at": captured_at.isoformat(),
                    "implied_probability": implied,
                    "last_price": _coerce_float(close_str),
                    "yes_bid": _nested_close("yes_bid"),
                    "yes_ask": _nested_close("yes_ask"),
                    "no_bid": _nested_close("no_bid"),
                    "no_ask": _nested_close("no_ask"),
                    "volume": _coerce_float(candle.get("volume_fp") or candle.get("volume")),
                    "open_interest": _coerce_float(candle.get("open_interest_fp") or candle.get("open_interest")),
                    "status": "candle",
                    "raw": candle,
                }
                await ledger.insert_snapshot(market.ticker, snapshot)
                total_snapshots += 1

            total_synced += 1

        if (i + 1) % 25 == 0:
            logger.info("Candlestick sync: %d/%d markets (snapshots=%d, errors=%d)",
                        i + 1, len(markets), total_snapshots, total_errors)

        await asyncio.sleep(delay)

    return {
        "markets_with_candles": total_synced,
        "total_snapshots": total_snapshots,
        "errors": total_errors,
    }


# ── Engine loading ────────────────────────────────────────────────────


def load_all_engines(*, multi_sample: int = 3) -> dict[str, BenchmarkEngine]:
    """Load all benchmark engines, optionally wrapped with multi-sample."""
    from nexus.engine.projection.actor_engine import ActorBenchmarkEngine
    from nexus.engine.projection.debate_engine import DebateBenchmarkEngine
    from nexus.engine.projection.graphrag_engine import GraphRAGBenchmarkEngine
    from nexus.engine.projection.naked_engine import NakedBenchmarkEngine
    from nexus.engine.projection.perspective_engine import PerspectiveBenchmarkEngine
    from nexus.engine.projection.structural_engine import StructuralBenchmarkEngine

    base_engines: dict[str, BenchmarkEngine] = {
        "naked": NakedBenchmarkEngine(),
        "structural": StructuralBenchmarkEngine(),
        "actor": ActorBenchmarkEngine(),
        "graphrag": GraphRAGBenchmarkEngine(),
        "perspective": PerspectiveBenchmarkEngine(),
        "debate": DebateBenchmarkEngine(),
    }

    if multi_sample <= 1:
        return base_engines

    return {
        name: MultiSampleWrapper(engine, n=multi_sample)
        for name, engine in base_engines.items()
    }


# ── Engine run helper ─────────────────────────────────────────────────


def _print_results(label: str, report):
    print(f"\n{'='*60}")
    print(f"RESULTS — {label}")
    print(f"{'='*60}")
    print(f"{'Engine':<25} {'Mean Brier':>12} {'N':>6}")
    print(f"{'-'*25} {'-'*12} {'-'*6}")
    for name in sorted(report.engine_results.keys()):
        r = report.engine_results[name]
        brier = r.get("mean_brier")
        n = r.get("questions_answered", 0)
        brier_str = f"{brier:.4f}" if brier is not None else "N/A"
        print(f"{name:<25} {brier_str:>12} {n:>6}")


async def _run_engines(mid_range, llm, store, *, multi_sample=3, concurrency=3):
    """Run all engines on the mid-range dataset (both independent and anchored)."""
    print(f"\n{'='*60}")
    print(f"Running engines on {len(mid_range)} mid-range questions...")

    engines = load_all_engines(multi_sample=multi_sample)
    engine_names = list(engines.keys())
    calls_estimate = len(mid_range) * multi_sample * 19
    print(f"  Engines: {engine_names}")
    print(f"  Multi-sample: {multi_sample}x (median)")
    print(f"  Estimated LLM calls: ~{calls_estimate}")
    print(f"  Concurrency: {concurrency}")

    # Run 1: Independent (no market probability hint)
    print(f"\n  --- Run 1: Independent (pure prediction) ---")
    t0 = time.monotonic()
    report_independent = await run_benchmark(
        mid_range, engines, llm=llm, store=store,
        concurrency=concurrency, independent=True,
    )
    elapsed = time.monotonic() - t0
    print(f"  Completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # Run 2: Market-anchored
    print(f"\n  --- Run 2: Market-anchored ---")
    t0 = time.monotonic()
    report_anchored = await run_benchmark(
        mid_range, engines, llm=llm, store=store,
        concurrency=concurrency, independent=False,
    )
    elapsed = time.monotonic() - t0
    print(f"  Completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # Save reports
    print(f"\n{'='*60}")
    print("Saving reports...")

    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    save_benchmark_report(
        report_independent,
        BENCH_DIR / "kalshi_engine_comparison_v2_independent.json",
    )
    save_benchmark_report(
        report_anchored,
        BENCH_DIR / "kalshi_engine_comparison_v2_anchored.json",
    )

    _print_results("Independent (pure prediction)", report_independent)
    _print_results("Market-anchored", report_anchored)


# ── Ingest results into forecast DB ──────────────────────────────────


def _log_loss(prob: float, outcome: bool) -> float:
    import math
    p = max(0.001, min(0.999, prob))
    return -math.log(p if outcome else 1 - p)


async def ingest_benchmark_results(store, report_path: Path, *, run_label: str):
    """Load a benchmark report JSON and persist each prediction to the forecast DB.

    Creates one ForecastRun per engine, with ForecastQuestions tagged
    target_variable="kalshi_benchmark" and immediately resolved.
    """
    from nexus.engine.projection.models import (
        ForecastQuestion,
        ForecastResolution,
        ForecastRun,
    )

    report = json.loads(report_path.read_text())
    per_question = report.get("per_question", [])
    engines = [k for k in report.get("engine_results", {}).keys() if k != "market"]

    if not per_question:
        print(f"  No per-question data in {report_path}")
        return

    # Load fixture for settlement_date lookup (report JSONs don't include it)
    fixture_path = Path(__file__).parent.parent / "data" / "fixtures" / "kalshi_benchmark_full.json"
    settlement_lookup: dict[tuple[str, str], str] = {}
    settlement_by_ticker: dict[str, str] = {}  # ticker-only fallback
    if fixture_path.exists():
        for item in json.loads(fixture_path.read_text()):
            key = (item["ticker"], item.get("cutoff_date", ""))
            settlement_lookup[key] = item["settlement_date"]
            settlement_by_ticker[item["ticker"]] = item["settlement_date"]

    ingested = 0
    for engine_name in engines:
        questions: list[ForecastQuestion] = []
        resolutions: list[tuple[ForecastQuestion, bool, float, float]] = []

        for q in per_question:
            prob = q.get(f"{engine_name}_prob")
            brier = q.get(f"{engine_name}_brier")
            if prob is None:
                continue

            outcome = q.get("outcome", False)
            market_prob = q.get("market_prob")
            ticker_key = q.get("ticker", "")
            cutoff_key = q.get("cutoff_date", "")
            settlement = (
                q.get("settlement_date")
                or settlement_lookup.get((ticker_key, cutoff_key))
                or settlement_by_ticker.get(ticker_key)
                or "2026-01-01"
            )
            cutoff = q.get("cutoff_date", "2025-12-01")

            # Map horizon to allowed Literal[3, 7, 14]
            if isinstance(settlement, str) and isinstance(cutoff, str):
                delta = (date.fromisoformat(settlement) - date.fromisoformat(cutoff)).days
            else:
                delta = 14
            horizon = 14 if delta > 14 else (7 if delta > 3 else 3)

            fq = ForecastQuestion(
                question=q.get("question", ""),
                probability=max(0.05, min(0.95, prob)),
                base_rate=market_prob if market_prob is not None else 0.5,
                target_variable="kalshi_benchmark",
                resolution_criteria=f"Kalshi market {q.get('ticker', '')} settled",
                resolution_date=date.fromisoformat(settlement) if isinstance(settlement, str) else settlement,
                horizon_days=horizon,
                signpost="Kalshi benchmark (backtested)",
                target_metadata={
                    "kalshi_ticker": q.get("ticker", ""),
                    "kalshi_implied": market_prob,
                    "market_prob": market_prob,
                    "market_brier": q.get("market_brier"),
                    "run_label": run_label,
                    "cutoff_date": cutoff,
                },
                external_ref=q.get("ticker", ""),
            )
            questions.append(fq)
            resolutions.append((fq, outcome, brier or 0.25, _log_loss(prob, outcome)))

        if not questions:
            continue

        run = ForecastRun(
            topic_slug="kalshi-benchmark",
            topic_name="Kalshi Benchmark",
            engine=engine_name,
            generated_for=date.today(),
            summary=f"Kalshi benchmark ({run_label}): {len(questions)} predictions",
            questions=questions,
            metadata={
                "kalshi_benchmark": True,
                "run_label": run_label,
                "source_file": str(report_path),
            },
        )
        await store.save_forecast_run(run)

        # Immediately resolve each question
        for fq, outcome, brier, ll in resolutions:
            if fq.question_id:
                await store.set_forecast_resolution(ForecastResolution(
                    forecast_question_id=fq.question_id,
                    outcome_status="resolved",
                    resolved_bool=outcome,
                    brier_score=brier,
                    log_loss=ll,
                    notes=f"Kalshi benchmark auto-resolved ({run_label})",
                    resolved_at=fq.resolution_date,
                ))
                ingested += 1

    print(f"  Ingested {ingested} predictions from {len(engines)} engines ({run_label})")


# ── Main orchestration ────────────────────────────────────────────────


async def main(
    *,
    discover_only: bool = False,
    sync_only: bool = False,
    skip_engines: bool = False,
    engines_only: bool = False,
    ingest_only: bool = False,
    max_markets: int = 500,
    multi_sample: int = 3,
    concurrency: int = 3,
):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # ── Fast path: load saved fixture and skip to engines ──
    if engines_only:
        midrange_path = FIXTURE_DIR / "kalshi_benchmark_midrange.json"
        if not midrange_path.exists():
            print(f"ERROR: {midrange_path} not found. Run without --engines-only first.")
            return

        mid_range = load_benchmark_dataset(midrange_path)
        print(f"Loaded {len(mid_range)} mid-range questions from {midrange_path}")

        # Jump straight to engine runs (Step 4)
        from nexus.config.models import ModelsConfig
        from nexus.engine.knowledge.store import KnowledgeStore
        from nexus.llm.client import LLMClient

        store = KnowledgeStore(Path("data/knowledge.db"))
        await store.initialize()

        llm = LLMClient(
            ModelsConfig(),
            api_key=os.getenv("GEMINI_API_KEY"),
            deepseek_api_key=os.getenv("DEEPSEEK_API_KEY"),
        )

        try:
            await _run_engines(mid_range, llm, store, multi_sample=multi_sample, concurrency=concurrency)
        finally:
            await store.close()
        return

    # ── Fast path: ingest existing results into forecast DB ──
    if ingest_only:
        from nexus.engine.knowledge.store import KnowledgeStore

        ind_path = BENCH_DIR / "kalshi_engine_comparison_v2_independent.json"
        anc_path = BENCH_DIR / "kalshi_engine_comparison_v2_anchored.json"

        if not ind_path.exists() and not anc_path.exists():
            print(f"ERROR: No benchmark results found in {BENCH_DIR}. Run engines first.")
            return

        store = KnowledgeStore(Path("data/knowledge.db"))
        await store.initialize()
        try:
            if ind_path.exists():
                print(f"Ingesting independent results...")
                await ingest_benchmark_results(store, ind_path, run_label="independent")
            if anc_path.exists():
                print(f"Ingesting anchored results...")
                await ingest_benchmark_results(store, anc_path, run_label="anchored")
            print("Done. Results visible on /predictions dashboard.")
        finally:
            await store.close()
        return

    config = KalshiBenchmarkConfig()
    client = KalshiClient(config)
    ledger = KalshiLedger(LEDGER_PATH)
    await ledger.initialize()

    try:
        # ── Step 1: Discover settled markets ──
        print(f"\n{'='*60}")
        print("Step 1: Discovering settled Kalshi markets...")
        t0 = time.monotonic()

        settled = await discover_settled_markets(
            client,
            max_markets=max_markets,
            exclude_categories={"Sports"},
        )
        elapsed = time.monotonic() - t0
        print(f"  Found {len(settled)} settled markets in {elapsed:.1f}s")

        # Store market metadata in ledger
        for market in settled:
            await ledger.upsert_market({
                "ticker": market.ticker,
                "event_ticker": market.event_ticker,
                "series_ticker": derive_series_ticker(market.event_ticker),
                "title": market.title,
                "status": "settled",
            })

        # Category breakdown
        from collections import Counter
        cats = Counter(m.category for m in settled)
        print("  Categories:", dict(cats.most_common(10)))

        # Outcome breakdown
        yes_count = sum(1 for m in settled if m.outcome)
        print(f"  Outcomes: YES={yes_count}, NO={len(settled) - yes_count}")

        if discover_only:
            print("\n--discover-only: stopping here.")
            return

        # ── Step 2: Sync candlestick history ──
        print(f"\n{'='*60}")
        print(f"Step 2: Syncing candlestick history for {len(settled)} markets...")
        print(f"  Estimated time: {len(settled) * 0.35:.0f}s ({len(settled) * 0.35 / 60:.1f} min)")
        t0 = time.monotonic()

        sync_stats = await sync_candlesticks_for_markets(ledger, client, settled)
        elapsed = time.monotonic() - t0
        print(f"  Synced in {elapsed:.1f}s: {sync_stats}")

        counts = await ledger.counts()
        print(f"  Ledger totals: {counts}")

        if sync_only:
            print("\n--sync-only: stopping here.")
            return

        # ── Step 3: Build dataset at multiple cutoffs ──
        print(f"\n{'='*60}")
        print("Step 3: Building benchmark dataset at 7/14/30-day cutoffs...")

        dataset = await build_benchmark_dataset(
            ledger, settled, cutoff_days=[7, 14, 30],
        )
        print(f"  Total questions (all cutoffs): {len(dataset)}")

        # Filter to mid-range
        mid_range = [q for q in dataset if 0.10 < q.market_prob_at_cutoff < 0.90]
        print(f"  Mid-range (0.10-0.90): {len(mid_range)}")

        # Full dataset stats
        extreme = [q for q in dataset if q.market_prob_at_cutoff <= 0.05 or q.market_prob_at_cutoff >= 0.95]
        print(f"  Extreme (≤0.05 or ≥0.95): {len(extreme)}")
        print(f"  Other: {len(dataset) - len(mid_range) - len(extreme)}")

        # Save full dataset
        FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
        save_benchmark_dataset(dataset, FIXTURE_DIR / "kalshi_benchmark_full.json")
        save_benchmark_dataset(mid_range, FIXTURE_DIR / "kalshi_benchmark_midrange.json")
        print(f"  Saved: {FIXTURE_DIR / 'kalshi_benchmark_full.json'}")
        print(f"  Saved: {FIXTURE_DIR / 'kalshi_benchmark_midrange.json'}")

        if skip_engines or not mid_range:
            if not mid_range:
                print("\n  WARNING: No mid-range questions found. Cannot run engines.")
            print("\nStopping before engine runs.")
            return

        # ── Step 4: Run all engines ──
        from nexus.config.models import ModelsConfig
        from nexus.engine.knowledge.store import KnowledgeStore
        from nexus.llm.client import LLMClient

        store = KnowledgeStore(Path("data/knowledge.db"))
        await store.initialize()

        llm = LLMClient(
            ModelsConfig(),
            api_key=os.getenv("GEMINI_API_KEY"),
            deepseek_api_key=os.getenv("DEEPSEEK_API_KEY"),
        )

        try:
            await _run_engines(mid_range, llm, store, multi_sample=multi_sample, concurrency=concurrency)
        finally:
            await store.close()

    finally:
        await ledger.close()


if __name__ == "__main__":
    flags = set(sys.argv[1:])
    asyncio.run(main(
        discover_only="--discover-only" in flags,
        sync_only="--sync-only" in flags,
        skip_engines="--skip-engines" in flags,
        engines_only="--engines-only" in flags,
        ingest_only="--ingest" in flags,
    ))
