"""Kalshi market matching, aligned forecasting, and divergence tracking.

Flips the mapping direction: instead of mapping our questions to Kalshi
markets (which produced incoherent comparisons), we scan Kalshi markets,
match them to our entity graph, and predict OUR probability for the exact
Kalshi question. Same question, different probability = directly comparable.

Zero LLM calls for scanning/matching (pure keyword + entity overlap).
One LLM call per matched market for graph-informed probability generation.
"""

from __future__ import annotations

import logging
import re
from datetime import date, timedelta

from nexus.engine.projection.models import ForecastQuestion

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Market scanning and entity matching
# ---------------------------------------------------------------------------


def _extract_implied_probability(market: dict) -> float | None:
    """Extract implied probability from a Kalshi market payload.

    Kalshi v2 API uses *_dollars suffixed fields (e.g., last_price_dollars,
    yes_bid_dollars). Values are already in 0.00-1.00 dollar range.
    """
    for field in (
        "implied_probability",
        "last_price_dollars", "last_price",
        "yes_bid_dollars", "yes_bid",
        "yes_ask_dollars", "yes_ask",
    ):
        val = market.get(field)
        if val is not None and val != "" and val != 0:
            fval = float(val)
            # Kalshi cents format (e.g., 32 for 32%)
            if fval > 1.0:
                fval = fval / 100.0
            if 0.0 < fval < 1.0:
                return round(fval, 4)
    return None


def _score_market_against_entities(
    market_text: str,
    entity_names: list[str],
    topic_name: str,
) -> tuple[int, list[str]]:
    """Score a market's text against our entities. Returns (score, matched_entities)."""
    matched = []
    score = 0

    for entity in entity_names:
        if re.search(r"\b" + re.escape(entity) + r"\b", market_text, re.IGNORECASE):
            matched.append(entity)
            score += 2  # Entity match is strong signal

    # Also check topic name words (word-boundary safe)
    topic_words = [w for w in topic_name.lower().replace("-", " ").split() if len(w) > 3]
    for word in topic_words:
        if re.search(r"\b" + re.escape(word) + r"\b", market_text, re.IGNORECASE):
            score += 1

    return score, matched


async def scan_kalshi_markets(
    client,
    *,
    entity_names: list[str],
    topic_name: str,
    max_markets: int = 10,
) -> list[dict]:
    """Scan open Kalshi events, score against our entities.

    Returns list of matched markets sorted by match_score descending.
    Zero LLM calls -- pure keyword/entity matching.
    """
    try:
        response = await client.list_events(status="open", limit=200)
    except Exception as exc:
        logger.warning("Kalshi market scan failed: %s", exc)
        return []

    events = response.get("events", [])
    candidates: list[dict] = []

    for event in events:
        event_title = event.get("title", "")
        event_ticker = event.get("event_ticker", "")

        for market in event.get("markets", []):
            if market.get("status", "").lower() not in ("open", "active", ""):
                continue

            market_title = market.get("title", "")
            market_subtitle = market.get("subtitle", "")
            combined_text = f"{event_title} {market_title} {market_subtitle}"

            score, matched = _score_market_against_entities(
                combined_text, entity_names, topic_name,
            )

            if score == 0:
                continue

            implied = _extract_implied_probability(market)

            candidates.append({
                "ticker": market.get("ticker", ""),
                "title": market_title,
                "event_title": event_title,
                "event_ticker": event_ticker,
                "subtitle": market_subtitle,
                "implied_probability": implied,
                "match_score": score,
                "matched_entities": matched,
                "volume": market.get("volume_fp") or market.get("volume"),
                "open_interest": market.get("open_interest_fp") or market.get("open_interest"),
            })

    # Sort by match score descending, then by volume
    candidates.sort(
        key=lambda m: (m["match_score"], m.get("volume") or 0),
        reverse=True,
    )

    return candidates[:max_markets]


# ---------------------------------------------------------------------------
# Aligned forecast generation
# ---------------------------------------------------------------------------


async def generate_aligned_forecasts(
    llm,
    store,
    matched_markets: list[dict],
    *,
    topic_slug: str,
    run_date: date,
    engine: str = "actor",
    ledger=None,
) -> list[ForecastQuestion]:
    """Generate OUR probability for each matched Kalshi market question.

    engine="actor": per-actor reasoning (original path).
    engine="structural": 3-call reasoning-first prediction (base rate + contrarian + supervisor).
    Falls back to market implied probability when LLM unavailable.
    """
    from nexus.engine.projection.forecasting import _clip_probability

    questions: list[ForecastQuestion] = []

    for market in matched_markets:
        ticker = market["ticker"]
        event_title = market.get("event_title", "")
        contract_title = market.get("title", "")
        _subtitle = market.get("subtitle", "")  # noqa: F841
        # Build a descriptive question: prefer "Event: Contract" for multi-option markets
        if contract_title and event_title and contract_title.lower() != event_title.lower():
            contract_clean = contract_title.rstrip("?")
            market_title = f"{event_title}: {contract_clean}"
        else:
            market_title = event_title or contract_title
        raw_implied = market.get("implied_probability") or 0.5
        implied = max(0.05, min(0.95, raw_implied))
        matched_entities = market.get("matched_entities", [])

        horizon_days = 14
        resolution_date = run_date + timedelta(days=horizon_days)

        our_probability = implied  # Default: trust market
        verdict = None
        confidence = None
        evidence_entities: list[str] = []

        if llm is not None:
            try:
                if engine == "structural":
                    from nexus.engine.projection.evidence import assemble_evidence_package
                    from nexus.engine.projection.structural_engine import predict_structural

                    evidence = await assemble_evidence_package(store, market_title, as_of=run_date)
                    assessment = await predict_structural(llm, evidence)
                    our_probability = assessment.implied_probability
                    verdict = assessment.verdict
                    confidence = assessment.confidence
                    evidence_entities = [
                        e.get("name", "") for e in evidence.entities if e.get("name")
                    ]
                elif engine == "actor":
                    from nexus.engine.projection.actor_engine import predict
                    prediction = await predict(
                        store, llm, market_title,
                        run_date=run_date,
                        market_prob=implied,
                        max_actors=4,
                    )
                    our_probability = prediction.calibrated_probability
                    verdict = prediction.verdict
                    confidence = prediction.confidence
                    evidence_entities = [a.actor for a in prediction.actors]
                elif engine in ("naked", "graphrag", "perspective", "debate"):
                    # BenchmarkEngine protocol: predict_probability() → float
                    if engine == "naked":
                        from nexus.engine.projection.naked_engine import NakedBenchmarkEngine
                        eng = NakedBenchmarkEngine()
                    elif engine == "graphrag":
                        from nexus.engine.projection.graphrag_engine import GraphRAGBenchmarkEngine
                        eng = GraphRAGBenchmarkEngine()
                    elif engine == "perspective":
                        from nexus.engine.projection.perspective_engine import PerspectiveBenchmarkEngine
                        eng = PerspectiveBenchmarkEngine()
                    else:
                        from nexus.engine.projection.debate_engine import DebateBenchmarkEngine
                        eng = DebateBenchmarkEngine()
                    our_probability = await eng.predict_probability(
                        market_title, llm=llm, store=store,
                        market_prob=implied, as_of=run_date,
                    )
                    # Derive verdict for KG-using engines (not naked/strawman)
                    if engine not in ("naked",):
                        from nexus.engine.projection.swarm import derive_verdict
                        verdict, confidence = derive_verdict(our_probability)
            except (ImportError, Exception) as exc:
                logger.warning(
                    "Kalshi aligned forecast failed for %s: %s", ticker, exc,
                )

        our_probability = _clip_probability(our_probability)

        metadata = {
            "kalshi_ticker": ticker,
            "kalshi_implied": implied,
            "matched_entities": matched_entities,
        }
        if verdict is not None:
            metadata["verdict"] = verdict
            metadata["confidence"] = confidence
        if evidence_entities:
            metadata["evidence_entities"] = evidence_entities

        # Snapshot market price in ledger if available
        if ledger is not None:
            try:
                from datetime import datetime, timezone
                from nexus.engine.projection.kalshi import _snapshot_from_market_payload
                snapshot = _snapshot_from_market_payload(
                    market, captured_at=datetime.now(tz=timezone.utc),
                )
                await ledger.insert_snapshot(ticker, snapshot)
            except Exception as exc:
                logger.debug("Ledger snapshot failed for %s: %s", ticker, exc)

        questions.append(ForecastQuestion(
            question=market_title,
            forecast_type="binary",
            target_variable="kalshi_aligned",
            probability=our_probability,
            base_rate=implied,
            resolution_criteria=f"Kalshi market {ticker} resolution",
            resolution_date=resolution_date,
            horizon_days=horizon_days,
            signpost=f"Kalshi market: {market['title']}",
            external_ref=ticker,
            target_metadata=metadata,
            signals_cited=[
                f"kalshi:implied={implied:.3f}",
                f"kalshi:our={our_probability:.3f}",
                f"kalshi:gap={abs(our_probability - implied):.3f}",
            ],
        ))

    return questions


# ---------------------------------------------------------------------------
# Divergence tracking
# ---------------------------------------------------------------------------


def compute_divergences(forecasts: list[ForecastQuestion]) -> list[dict]:
    """Compute divergence between our probability and Kalshi implied.

    Returns list sorted by absolute gap descending.
    """
    if not forecasts:
        return []

    divergences: list[dict] = []

    for q in forecasts:
        kalshi_implied = (q.target_metadata or {}).get("kalshi_implied")
        if kalshi_implied is None:
            continue

        gap = q.probability - kalshi_implied
        divergences.append({
            "ticker": q.external_ref or "",
            "question": q.question,
            "our_probability": round(q.probability, 3),
            "kalshi_probability": round(kalshi_implied, 3),
            "gap_pp": round(abs(gap) * 100, 1),
            "direction": "above" if gap > 0 else "below",
        })

    divergences.sort(key=lambda d: d["gap_pp"], reverse=True)
    return divergences


# ---------------------------------------------------------------------------
# Briefing section renderer
# ---------------------------------------------------------------------------


def render_divergence_section(divergences: list[dict]) -> str:
    """Render a markdown section for market divergence signals.

    Included in the briefing context for the LLM to incorporate.
    """
    if not divergences:
        return ""

    lines = ["\n## Market Divergence Signals"]

    for d in divergences:
        gap = d["gap_pp"]
        _ticker = d["ticker"]  # noqa: F841
        our = d["our_probability"]
        market = d["kalshi_probability"]
        question = d["question"]

        if gap >= 10:
            label = "investigate"
        elif gap >= 5:
            label = "notable"
        else:
            label = "aligned"

        lines.append(
            f"- **{question}**: Our {our:.0%} vs Kalshi {market:.0%} "
            f"(gap: {gap:.0f}pp) -- {label}"
        )

    return "\n".join(lines)
