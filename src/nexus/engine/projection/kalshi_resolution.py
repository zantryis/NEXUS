"""Kalshi market resolution — ground truth scoring without LLM judges.

Markets settle to YES or NO. Brier score = (our_probability - outcome)^2.
No LLM judging needed. Pure arithmetic.
"""

from __future__ import annotations

import logging
from datetime import date

from nexus.engine.knowledge.store import KnowledgeStore
from nexus.engine.projection.models import ForecastResolution

logger = logging.getLogger(__name__)


def _brier_score(probability: float, outcome: bool) -> float:
    """Compute Brier score: (probability - outcome)^2."""
    target = 1.0 if outcome else 0.0
    return round((probability - target) ** 2, 4)


async def resolve_kalshi_forecasts(
    store: KnowledgeStore,
    client,
    *,
    as_of: date | None = None,
) -> dict:
    """Resolve Kalshi-aligned forecasts against market settlements.

    For each open forecast with target_variable="kalshi_aligned":
      1. Extract ticker from target_metadata
      2. Fetch market status via client.fetch_market(ticker)
      3. If settled: compute brier_score = (our_prob - outcome)^2
      4. Save ForecastResolution to store

    Returns: {resolved, still_open, errors, brier_scores, mean_brier}
    """
    # Get ALL unresolved Kalshi-aligned forecasts (don't filter by resolution_date,
    # since our resolution_date is artificial — Kalshi markets settle on their own schedule)
    all_questions = await store.get_forecast_questions_between(
        start=date(2020, 1, 1), end=as_of or date.today(),
    )
    kalshi_questions = [
        q for q in all_questions
        if q.get("target_variable") == "kalshi_aligned"
        and q.get("outcome_status") != "resolved"
    ]

    resolved = 0
    still_open = 0
    errors = 0
    brier_scores: list[float] = []
    details: list[dict] = []

    for q in kalshi_questions:
        question_id = q["forecast_question_id"]
        probability = float(q["probability"])

        # Extract ticker from target_metadata (store returns parsed dict)
        metadata = q.get("target_metadata") or {}
        ticker = metadata.get("kalshi_ticker")
        if not ticker:
            logger.warning(
                "No kalshi_ticker in metadata for question %s (%s) — skipping resolution",
                question_id,
                q.get("question", "")[:80],
            )
            errors += 1
            continue

        try:
            market = await client.fetch_market(ticker)
        except Exception as exc:
            logger.warning("Failed to fetch market %s: %s", ticker, exc)
            errors += 1
            continue

        status = str(market.get("status", "")).lower()
        if status not in {"settled", "finalized", "closed"}:
            still_open += 1
            continue

        # Determine outcome
        result_str = str(market.get("result", "")).lower()
        if result_str in {"yes", "true", "1"}:
            outcome = True
        elif result_str in {"no", "false", "0"}:
            outcome = False
        else:
            logger.warning("Unexpected market result for %s: %s", ticker, result_str)
            errors += 1
            continue

        brier = _brier_score(probability, outcome)
        brier_scores.append(brier)

        resolution = ForecastResolution(
            forecast_question_id=question_id,
            outcome_status="resolved",
            resolved_bool=outcome,
            actual_value=1.0 if outcome else 0.0,
            brier_score=brier,
            notes=f"Kalshi market {ticker} settled: {result_str}",
            resolved_at=as_of or date.today(),
            external_ref=ticker,
        )
        await store.set_forecast_resolution(resolution)
        resolved += 1

        details.append({
            "ticker": ticker,
            "our_probability": probability,
            "outcome": outcome,
            "brier_score": brier,
        })

    mean_brier = round(sum(brier_scores) / len(brier_scores), 4) if brier_scores else None

    return {
        "resolved": resolved,
        "still_open": still_open,
        "errors": errors,
        "brier_scores": brier_scores,
        "mean_brier": mean_brier,
        "details": details,
    }


async def kalshi_scoring_report(
    store: KnowledgeStore,
    *,
    start: date,
    end: date,
) -> dict:
    """Aggregate scoring report: our Brier vs market Brier.

    Queries resolved Kalshi-aligned forecast questions and computes:
    - Our mean Brier score
    - Market mean Brier score (using base_rate = market implied)
    - Per-question breakdown
    """
    questions = await store.get_forecast_questions_between(
        start=start, end=end, engine=None,
    )

    kalshi_resolved = [
        q for q in questions
        if q.get("target_variable") == "kalshi_aligned"
        and q.get("outcome_status") == "resolved"
        and q.get("resolved_bool") is not None
    ]

    if not kalshi_resolved:
        return {
            "total_resolved": 0,
            "our_mean_brier": None,
            "market_mean_brier": None,
            "per_question": [],
        }

    our_briers: list[float] = []
    market_briers: list[float] = []
    per_question: list[dict] = []

    for q in kalshi_resolved:
        our_prob = float(q["probability"])
        outcome = bool(q["resolved_bool"])
        our_brier = _brier_score(our_prob, outcome)
        our_briers.append(our_brier)

        # Market probability from target_metadata or base_rate
        market_prob = float(q.get("base_rate", 0.5))
        meta = q.get("target_metadata") or {}
        if "kalshi_implied" in meta:
            market_prob = float(meta["kalshi_implied"])

        market_brier = _brier_score(market_prob, outcome)
        market_briers.append(market_brier)

        per_question.append({
            "question": q.get("question", ""),
            "our_probability": our_prob,
            "market_probability": market_prob,
            "outcome": outcome,
            "our_brier": our_brier,
            "market_brier": market_brier,
            "generated_for": q.get("generated_for"),
        })

    return {
        "total_resolved": len(kalshi_resolved),
        "our_mean_brier": round(sum(our_briers) / len(our_briers), 4),
        "market_mean_brier": round(sum(market_briers) / len(market_briers), 4),
        "per_question": per_question,
    }
