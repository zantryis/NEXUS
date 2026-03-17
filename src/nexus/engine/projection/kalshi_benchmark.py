"""Kalshi benchmark — settled market discovery, dataset building, benchmark harness."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)


# ── Data Models ───────────────────────────────────────────────────────


@dataclass
class SettledMarket:
    """A Kalshi market that has settled with a known outcome."""

    ticker: str
    event_ticker: str
    title: str  # full title with deadline/expiry context
    outcome: bool  # YES=True, NO=False
    settlement_date: date
    category: str  # from event_ticker or series_ticker


@dataclass
class BenchmarkQuestion:
    """A single benchmark question with known outcome and market price at cutoff."""

    ticker: str
    question: str  # full market title
    outcome: bool  # ground truth
    settlement_date: date
    cutoff_date: date  # when we "make" our prediction
    market_prob_at_cutoff: float  # implied probability at cutoff
    category: str


@dataclass
class BenchmarkReport:
    """Results of running benchmark engines on a dataset."""

    total_questions: int
    engine_results: dict[str, dict]  # engine_name → {mean_brier, brier_scores, ...}
    per_question: list[dict] = field(default_factory=list)


@runtime_checkable
class BenchmarkEngine(Protocol):
    """Simplified protocol for benchmark comparisons."""

    engine_name: str

    async def predict_probability(
        self,
        question: str,
        *,
        llm=None,
        store=None,
        market_prob: float | None = None,
        as_of: date | None = None,
    ) -> float: ...


# ── Market Baseline Engine ────────────────────────────────────────────


class MarketBaselineEngine:
    """Baseline engine: just returns the market's own implied probability."""

    engine_name = "market"

    async def predict_probability(
        self,
        question: str,
        *,
        llm=None,
        store=None,
        market_prob: float | None = None,
        as_of: date | None = None,
    ) -> float:
        return market_prob if market_prob is not None else 0.50


# ── Discovery ─────────────────────────────────────────────────────────


async def discover_settled_markets(
    client,
    *,
    days_back: int = 90,
    max_markets: int = 0,
    exclude_categories: set[str] | None = None,
    max_pages: int = 100,
) -> list[SettledMarket]:
    """Discover recently-settled Kalshi markets via the events API.

    Paginates through settled events, filters to binary markets with
    yes/no results, and returns structured SettledMarket objects.

    Args:
        max_markets: Stop after collecting this many markets (0 = unlimited).
        exclude_categories: Category strings to skip (e.g. sports, crypto).
        max_pages: Safety limit on API pagination.
    """
    if exclude_categories is None:
        exclude_categories = set()
    exclude_lower = {c.lower() for c in exclude_categories}

    markets: list[SettledMarket] = []
    cursor: str | None = None
    pages = 0

    while pages < max_pages:
        pages += 1
        response = await client.list_events(status="settled", limit=200, cursor=cursor)
        events = response.get("events", [])

        for event in events:
            event_ticker = event.get("event_ticker", "")
            category = event.get("category", "")

            if category.lower() in exclude_lower:
                continue

            for market in event.get("markets", []):
                result = str(market.get("result", "")).lower()
                if result not in ("yes", "no"):
                    continue  # skip non-binary or missing results

                status = str(market.get("status", "")).lower()
                if status not in ("settled", "finalized"):
                    continue

                # Parse settlement date from close_time
                close_time = market.get("close_time", "")
                if close_time:
                    try:
                        settlement_dt = datetime.fromisoformat(
                            close_time.replace("Z", "+00:00")
                        )
                        settlement_date = settlement_dt.date()
                    except (ValueError, TypeError):
                        settlement_date = date.today()
                else:
                    settlement_date = date.today()

                markets.append(
                    SettledMarket(
                        ticker=market.get("ticker", ""),
                        event_ticker=event_ticker,
                        title=market.get("title", ""),
                        outcome=(result == "yes"),
                        settlement_date=settlement_date,
                        category=category,
                    )
                )

                if max_markets and len(markets) >= max_markets:
                    return markets

        next_cursor = response.get("cursor")
        if not next_cursor or not events:
            break
        cursor = next_cursor

    return markets


# ── Dataset Building ──────────────────────────────────────────────────


def _parse_implied_prob(market: dict) -> float | None:
    """Extract implied probability from market metadata payload."""
    # Try explicit implied_probability first
    for key in ("implied_probability", "last_price_dollars", "last_price"):
        val = market.get(key)
        if val is not None and val != "":
            try:
                p = float(val)
                # Kalshi prices in dollars are 0-1; in cents they're 0-100
                return round(p / 100.0, 4) if p > 1.0 else round(p, 4)
            except (TypeError, ValueError):
                continue
    return None


async def build_benchmark_from_metadata(
    ledger,
    settled_markets: list[SettledMarket],
    *,
    min_prob: float = 0.0,
    max_prob: float = 1.0,
) -> list[BenchmarkQuestion]:
    """Build benchmark questions using market metadata (last traded price).

    Use this when historical candlestick snapshots aren't available.
    Each settled market produces one question using its last_price as
    the market probability baseline.

    Args:
        min_prob: Exclude markets with implied probability below this.
        max_prob: Exclude markets with implied probability above this.
    """
    questions: list[BenchmarkQuestion] = []

    for market in settled_markets:
        meta = await ledger.get_market_metadata(market.ticker)
        if not meta:
            continue

        prob = _parse_implied_prob(meta)
        if prob is None:
            continue
        if prob < min_prob or prob > max_prob:
            continue

        questions.append(
            BenchmarkQuestion(
                ticker=market.ticker,
                question=market.title,
                outcome=market.outcome,
                settlement_date=market.settlement_date,
                cutoff_date=market.settlement_date,  # single point
                market_prob_at_cutoff=prob,
                category=market.category,
            )
        )

    return questions


async def build_benchmark_dataset(
    ledger,
    settled_markets: list[SettledMarket],
    *,
    cutoff_days: list[int] | None = None,
) -> list[BenchmarkQuestion]:
    """Build benchmark questions from settled markets at multiple cutoff horizons.

    For each market × cutoff, find the nearest snapshot ≤ cutoff date.
    Markets without a snapshot before the cutoff are skipped.
    """
    if cutoff_days is None:
        cutoff_days = [7, 14, 30]

    questions: list[BenchmarkQuestion] = []

    for market in settled_markets:
        for days in cutoff_days:
            cutoff_date = market.settlement_date - timedelta(days=days)
            cutoff_dt = datetime.combine(
                cutoff_date, time.max, tzinfo=timezone.utc
            )

            snapshot = await ledger.get_nearest_snapshot(
                market.ticker, captured_at=cutoff_dt
            )
            if not snapshot or snapshot.get("implied_probability") is None:
                continue

            questions.append(
                BenchmarkQuestion(
                    ticker=market.ticker,
                    question=market.title,
                    outcome=market.outcome,
                    settlement_date=market.settlement_date,
                    cutoff_date=cutoff_date,
                    market_prob_at_cutoff=float(snapshot["implied_probability"]),
                    category=market.category,
                )
            )

    return questions


# ── Benchmark Harness ─────────────────────────────────────────────────


def _brier(probability: float, outcome: bool) -> float:
    """Compute Brier score: (probability - outcome)^2."""
    target = 1.0 if outcome else 0.0
    p = max(0.05, min(0.95, float(probability)))
    return round((p - target) ** 2, 4)


async def _run_engine_on_question(
    engine_name: str,
    engine: BenchmarkEngine | None,
    q: BenchmarkQuestion,
    *,
    llm=None,
    store=None,
    pass_market_prob: bool = True,
) -> tuple[str, float]:
    """Run a single engine on a single question. Returns (engine_name, probability)."""
    if engine_name == "market" or engine is None:
        return engine_name, q.market_prob_at_cutoff
    try:
        prob = await engine.predict_probability(
            q.question,
            llm=llm,
            store=store,
            market_prob=q.market_prob_at_cutoff if pass_market_prob else None,
            as_of=q.cutoff_date,
        )
        return engine_name, prob
    except Exception as exc:
        logger.warning("Engine %s failed on %s: %s", engine_name, q.ticker, exc)
        return engine_name, q.market_prob_at_cutoff


async def run_benchmark(
    dataset: list[BenchmarkQuestion],
    engines: dict[str, BenchmarkEngine] | None = None,
    *,
    llm=None,
    store=None,
    concurrency: int = 5,
    progress_every: int = 10,
    independent: bool = False,
) -> BenchmarkReport:
    """Run all engines on all benchmark questions, compute Brier scores.

    Always includes a 'market' baseline engine that returns market_prob_at_cutoff.
    Engines for each question run concurrently; questions are batched via semaphore.

    If independent=True, market_prob is NOT passed to engines — tests pure prediction.
    """
    import asyncio

    if engines is None:
        engines = {}

    all_engines: dict[str, BenchmarkEngine | None] = {"market": None}
    all_engines.update(engines)

    engine_briers: dict[str, list[float]] = {name: [] for name in all_engines}
    per_question: list[dict] = []
    sem = asyncio.Semaphore(concurrency)

    async def process_question(q: BenchmarkQuestion) -> dict:
        async with sem:
            tasks = [
                _run_engine_on_question(
                    name, eng, q, llm=llm, store=store,
                    pass_market_prob=not independent,
                )
                for name, eng in all_engines.items()
            ]
            results = await asyncio.gather(*tasks)

        row: dict = {
            "ticker": q.ticker,
            "question": q.question,
            "outcome": q.outcome,
            "cutoff_date": q.cutoff_date.isoformat(),
            "market_prob": q.market_prob_at_cutoff,
        }
        for engine_name, prob in results:
            brier = _brier(prob, q.outcome)
            row[f"{engine_name}_prob"] = prob
            row[f"{engine_name}_brier"] = brier
        return row

    # Run all questions with concurrency
    all_rows = await asyncio.gather(
        *[process_question(q) for q in dataset]
    )

    # Collect results (maintain dataset order)
    for row in all_rows:
        per_question.append(row)
        for name in all_engines:
            brier = row.get(f"{name}_brier")
            if brier is not None:
                engine_briers[name].append(brier)

    # Progress logging
    if per_question and progress_every:
        logger.info("Benchmark complete: %d questions scored.", len(per_question))

    engine_results: dict[str, dict] = {}
    for name, briers in engine_briers.items():
        if briers:
            engine_results[name] = {
                "mean_brier": round(sum(briers) / len(briers), 4),
                "brier_scores": briers,
                "questions_answered": len(briers),
            }
        else:
            engine_results[name] = {
                "mean_brier": None,
                "brier_scores": [],
                "questions_answered": 0,
            }

    return BenchmarkReport(
        total_questions=len(dataset),
        engine_results=engine_results,
        per_question=per_question,
    )


# ── Serialization ─────────────────────────────────────────────────────


def save_benchmark_dataset(
    questions: list[BenchmarkQuestion], path: Path
) -> None:
    """Save benchmark dataset to JSON."""
    data = [
        {
            "ticker": q.ticker,
            "question": q.question,
            "outcome": q.outcome,
            "settlement_date": q.settlement_date.isoformat(),
            "cutoff_date": q.cutoff_date.isoformat(),
            "market_prob_at_cutoff": q.market_prob_at_cutoff,
            "category": q.category,
        }
        for q in questions
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def load_benchmark_dataset(path: Path) -> list[BenchmarkQuestion]:
    """Load benchmark dataset from JSON."""
    data = json.loads(path.read_text())
    return [
        BenchmarkQuestion(
            ticker=d["ticker"],
            question=d["question"],
            outcome=d["outcome"],
            settlement_date=date.fromisoformat(d["settlement_date"]),
            cutoff_date=date.fromisoformat(d["cutoff_date"]),
            market_prob_at_cutoff=d["market_prob_at_cutoff"],
            category=d["category"],
        )
        for d in data
    ]


def save_benchmark_report(report: BenchmarkReport, path: Path) -> None:
    """Save benchmark report to JSON."""
    data = {
        "total_questions": report.total_questions,
        "engine_results": report.engine_results,
        "per_question": report.per_question,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))
