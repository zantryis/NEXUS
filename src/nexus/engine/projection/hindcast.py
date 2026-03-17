"""Hindcast benchmark — backtest prediction engines against known KG outcomes.

Pick significant events that already happened, generate retrospective prediction
questions, run engines with leakage-safe cutoff dates, and score against ground truth.
Negative cases (entity+window where nothing happened) prevent trivially-YES benchmarks.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import statistics
from dataclasses import dataclass, field
from datetime import date, timedelta
from math import log

from nexus.engine.knowledge.events import Event
from nexus.engine.knowledge.store import KnowledgeStore

logger = logging.getLogger(__name__)


# ── Data Models ──────────────────────────────────────────────────────


@dataclass
class HindcastCase:
    """One hindcast question with known ground truth."""

    case_id: str  # "{topic_slug}:{cutoff}:{entity}"
    topic_slug: str
    question: str
    cutoff_date: date  # engines see data up to here
    resolution_date: date  # cutoff + horizon
    horizon_days: int
    outcome: bool  # ground truth
    source_event_id: int | None  # event that inspired the question (positive)
    source_entity: str
    case_type: str  # "positive" | "negative"


@dataclass
class BacktestReport:
    """Aggregated results of a hindcast benchmark run."""

    total_cases: int
    positive_cases: int
    negative_cases: int
    engines: list[str]
    engine_results: dict[str, dict] = field(default_factory=dict)
    calibration: dict[str, list[dict]] = field(default_factory=dict)
    per_case: list[dict] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


# ── Scoring helpers ──────────────────────────────────────────────────


def _brier(probability: float, outcome: bool) -> float:
    target = 1.0 if outcome else 0.0
    p = max(0.05, min(0.95, float(probability)))
    return round((p - target) ** 2, 4)


def _log_loss(probability: float, outcome: bool) -> float:
    p = max(0.05, min(0.95, float(probability)))
    return round(-log(p if outcome else (1.0 - p)), 4)


# ── Event selection ──────────────────────────────────────────────────


async def select_hindcast_events(
    store: KnowledgeStore,
    topic_slug: str,
    *,
    start: date,
    end: date,
    min_significance: int = 7,
) -> list[Event]:
    """Select high-significance events suitable for hindcast questions."""
    events = await store.get_events(topic_slug, since=start, until=end)
    return [e for e in events if e.significance >= min_significance]


# ── Question generation ──────────────────────────────────────────────

_QUESTION_GEN_SYSTEM = (
    "You convert a real event into a prediction question that could have been "
    "asked before the event happened. Return JSON only."
)

_QUESTION_GEN_PROMPT = """\
Given this real event that occurred on {event_date}:
"{summary}"
Entities involved: {entities}
Topic: {topic_name}

Write a binary YES/NO prediction question that someone could have asked \
{horizon_days} days before this event. The question should:
- Be specific and time-bounded (include a deadline around {event_date})
- Reference specific entities by name
- Be naturally askable by a forecaster who doesn't know the outcome

Return JSON only: {{"question": "...", "primary_entity": "..."}}
"""


async def generate_hindcast_questions(
    llm,
    events: list[Event],
    topic_slug: str,
    topic_name: str,
    *,
    horizon_days: int = 7,
) -> list[HindcastCase]:
    """Generate positive hindcast cases from significant events.

    LLM path: converts event → natural prediction question (1 cheap call per event).
    Template fallback: generates from entity names when llm=None.
    """
    cases: list[HindcastCase] = []

    for event in events:
        cutoff = event.date - timedelta(days=horizon_days)
        entity = event.entities[0] if event.entities else topic_name
        question: str | None = None

        if llm is not None:
            try:
                raw = await llm.complete(
                    "knowledge_summary",
                    _QUESTION_GEN_SYSTEM,
                    _QUESTION_GEN_PROMPT.format(
                        event_date=event.date.isoformat(),
                        summary=event.summary,
                        entities=", ".join(event.entities) or "unknown",
                        topic_name=topic_name,
                        horizon_days=horizon_days,
                    ),
                    json_response=True,
                )
                data = json.loads(raw)
                question = data.get("question")
                entity = data.get("primary_entity", entity)
            except Exception as exc:
                logger.debug("Question generation LLM failed: %s", exc)

        if not question:
            question = (
                f"Will {entity} be involved in significant {topic_name} "
                f"developments by {event.date.isoformat()}?"
            )

        cases.append(HindcastCase(
            case_id=f"{topic_slug}:{cutoff.isoformat()}:{entity}",
            topic_slug=topic_slug,
            question=question,
            cutoff_date=cutoff,
            resolution_date=event.date,
            horizon_days=horizon_days,
            outcome=True,
            source_event_id=event.event_id,
            source_entity=entity,
            case_type="positive",
        ))

    return cases


# ── Negative sampling ────────────────────────────────────────────────


async def sample_negative_cases(
    store: KnowledgeStore,
    topic_slug: str,
    topic_name: str,
    positive_cases: list[HindcastCase],
    *,
    horizon_days: int = 7,
    min_significance: int = 7,
) -> list[HindcastCase]:
    """Generate negative hindcast cases — entities active before cutoff but quiet during window.

    Zero LLM calls. Caps negatives to match positive count (~50% positive rate).
    """
    if not positive_cases:
        return []

    negatives: list[HindcastCase] = []
    max_negatives = len(positive_cases)

    # Collect cutoff windows and positive entities per window
    positive_entities_by_cutoff: dict[date, set[str]] = {}
    for case in positive_cases:
        positive_entities_by_cutoff.setdefault(case.cutoff_date, set()).add(case.source_entity)

    for cutoff in positive_entities_by_cutoff:
        if len(negatives) >= max_negatives:
            break

        resolution = cutoff + timedelta(days=horizon_days)
        excluded = positive_entities_by_cutoff[cutoff]

        # Find entities active before cutoff (had events in prior 30 days)
        pre_cutoff_events = await store.get_events(
            topic_slug, since=cutoff - timedelta(days=30), until=cutoff,
        )
        active_entities: dict[str, int] = {}
        for ev in pre_cutoff_events:
            for ent_name in ev.entities:
                if ent_name not in excluded:
                    active_entities[ent_name] = active_entities.get(ent_name, 0) + 1

        # Check which entities had NO significant events in the forecast window
        window_events = await store.get_events(
            topic_slug, since=cutoff + timedelta(days=1), until=resolution,
        )
        significant_in_window: set[str] = set()
        for ev in window_events:
            if ev.significance >= min_significance:
                significant_in_window.update(ev.entities)

        # Entities active before but quiet during → negative cases
        for entity_name, _count in sorted(active_entities.items(), key=lambda x: -x[1]):
            if entity_name in significant_in_window:
                continue
            if len(negatives) >= max_negatives:
                break

            question = (
                f"Will {entity_name} be involved in significant {topic_name} "
                f"developments by {resolution.isoformat()}?"
            )
            negatives.append(HindcastCase(
                case_id=f"{topic_slug}:{cutoff.isoformat()}:{entity_name}",
                topic_slug=topic_slug,
                question=question,
                cutoff_date=cutoff,
                resolution_date=resolution,
                horizon_days=horizon_days,
                outcome=False,
                source_event_id=None,
                source_entity=entity_name,
                case_type="negative",
            ))

    return negatives


# ── Calibration ──────────────────────────────────────────────────────


def compute_calibration(results: list[dict]) -> list[dict]:
    """Compute calibration curve from prediction results.

    Groups predictions into decile buckets and computes actual hit rate per bucket.
    Each result dict must have 'probability' and 'outcome' keys.
    """
    buckets: dict[float, list[dict]] = {}
    for r in results:
        p = r["probability"]
        bucket = round(int(p * 10) / 10, 1)  # 0.0, 0.1, ..., 0.9
        bucket = min(bucket, 0.9)  # cap at 0.9 bucket
        buckets.setdefault(bucket, []).append(r)

    calibration: list[dict] = []
    for bucket in sorted(buckets):
        items = buckets[bucket]
        predicted_mean = sum(r["probability"] for r in items) / len(items)
        actual_rate = sum(1 for r in items if r["outcome"]) / len(items)
        calibration.append({
            "bucket": bucket,
            "predicted_mean": round(predicted_mean, 4),
            "actual_rate": round(actual_rate, 4),
            "count": len(items),
        })

    return calibration


# ── Main backtest orchestrator ───────────────────────────────────────


async def backtest_forecasts(
    store: KnowledgeStore,
    llm,
    *,
    topics: list[tuple[str, str]],
    start: date,
    end: date,
    engines: list[str] | None = None,
    horizon_days: int = 7,
    min_significance: int = 7,
    max_cases_per_topic: int = 30,
    persist: bool = True,
) -> BacktestReport:
    """Run hindcast benchmark: generate retrospective questions, score engines.

    Args:
        store: Knowledge store with historical events.
        llm: LLM client (or None for template-only questions).
        topics: List of (topic_slug, topic_name) pairs to benchmark.
        start/end: Date range for selecting source events.
        engines: Engine names to evaluate (default: ["structural"]).
        horizon_days: How many days before the event to set the cutoff.
        min_significance: Minimum event significance for positive cases.
        max_cases_per_topic: Cap on cases per topic.
        persist: Whether to save forecast runs/resolutions to the store.

    Returns:
        BacktestReport with per-engine Brier scores and calibration data.
    """
    from nexus.engine.projection.models import (
        ForecastQuestion,
        ForecastResolution,
        ForecastRun,
    )

    if engines is None:
        engines = ["structural"]

    all_cases: list[HindcastCase] = []

    # Phase 1: Generate cases per topic
    for topic_slug, topic_name in topics:
        selected = await select_hindcast_events(
            store, topic_slug, start=start, end=end,
            min_significance=min_significance,
        )
        if not selected:
            logger.info("No significant events for %s in range, skipping.", topic_slug)
            continue

        positives = await generate_hindcast_questions(
            llm, selected, topic_slug, topic_name, horizon_days=horizon_days,
        )
        negatives = await sample_negative_cases(
            store, topic_slug, topic_name, positives, horizon_days=horizon_days,
            min_significance=min_significance,
        )

        combined = positives + negatives
        random.seed(42)  # deterministic shuffle
        random.shuffle(combined)
        all_cases.extend(combined[:max_cases_per_topic])

    if not all_cases:
        return BacktestReport(
            total_cases=0, positive_cases=0, negative_cases=0,
            engines=engines, metadata={"start": start.isoformat(), "end": end.isoformat()},
        )

    # Phase 2: Run engines on each case
    engine_briers: dict[str, list[float]] = {e: [] for e in engines}
    engine_log_losses: dict[str, list[float]] = {e: [] for e in engines}
    per_case: list[dict] = []
    calibration_data: dict[str, list[dict]] = {e: [] for e in engines}

    for case_idx, case in enumerate(all_cases, 1):
        logger.info(
            "Case %d/%d [%s] %s: %s",
            case_idx, len(all_cases), case.case_type,
            case.source_entity, case.question[:80],
        )
        row: dict = {
            "case_id": case.case_id,
            "question": case.question,
            "outcome": case.outcome,
            "case_type": case.case_type,
            "cutoff_date": case.cutoff_date.isoformat(),
            "resolution_date": case.resolution_date.isoformat(),
        }

        for engine_name in engines:
            prob = 0.5  # fallback
            try:
                prob = await _run_engine(
                    engine_name, store, llm, case.question,
                    cutoff=case.cutoff_date,
                )
                logger.info("  %s → %.3f", engine_name, prob)
            except asyncio.TimeoutError:
                logger.warning("Engine %s timed out on %s, using fallback 0.5", engine_name, case.case_id)
            except Exception as exc:
                logger.warning("Engine %s failed on %s: %s", engine_name, case.case_id, exc)

            brier = _brier(prob, case.outcome)
            ll = _log_loss(prob, case.outcome)
            engine_briers[engine_name].append(brier)
            engine_log_losses[engine_name].append(ll)
            calibration_data[engine_name].append({
                "probability": prob,
                "outcome": case.outcome,
            })

            row[f"{engine_name}_prob"] = prob
            row[f"{engine_name}_brier"] = brier

        per_case.append(row)

        # Phase 3: Persist if requested
        if persist:
            for engine_name in engines:
                prob = row.get(f"{engine_name}_prob", 0.5)
                brier = row.get(f"{engine_name}_brier", 0.25)
                fq = ForecastQuestion(
                    question=case.question,
                    probability=prob,
                    base_rate=0.5,
                    target_variable="hindcast",
                    resolution_criteria=f"Hindcast {case.case_type}: {case.case_id}",
                    resolution_date=case.resolution_date,
                    horizon_days=min(case.horizon_days, 14),
                    signpost="hindcast benchmark",
                    target_metadata={
                        "case_type": case.case_type,
                        "source_event_id": case.source_event_id,
                        "cutoff_date": case.cutoff_date.isoformat(),
                    },
                )
                run = ForecastRun(
                    topic_slug=case.topic_slug,
                    topic_name=case.topic_slug,
                    engine=engine_name,
                    generated_for=case.cutoff_date,
                    summary=f"Hindcast: {case.case_type} case for {case.source_entity}",
                    questions=[fq],
                    metadata={"hindcast": True},
                )
                await store.save_forecast_run(run)

                # save_forecast_run sets fq.question_id in-place
                if fq.question_id:
                    await store.set_forecast_resolution(ForecastResolution(
                        forecast_question_id=fq.question_id,
                        outcome_status="resolved",
                        resolved_bool=case.outcome,
                        brier_score=brier,
                        log_loss=_log_loss(prob, case.outcome),
                        notes=f"Hindcast auto-resolved: {case.case_type}",
                        resolved_at=case.resolution_date,
                    ))

    # Phase 4: Aggregate report
    positive_count = sum(1 for c in all_cases if c.outcome)
    engine_results: dict[str, dict] = {}
    for engine_name in engines:
        briers = engine_briers[engine_name]
        if briers:
            engine_results[engine_name] = {
                "mean_brier": round(sum(briers) / len(briers), 4),
                "median_brier": round(statistics.median(briers), 4),
                "n": len(briers),
                "brier_scores": briers,
                "mean_log_loss": round(
                    sum(engine_log_losses[engine_name]) / len(engine_log_losses[engine_name]), 4,
                ),
            }
        else:
            engine_results[engine_name] = {
                "mean_brier": None, "median_brier": None, "n": 0,
                "brier_scores": [], "mean_log_loss": None,
            }

    cal: dict[str, list[dict]] = {}
    for engine_name in engines:
        cal[engine_name] = compute_calibration(calibration_data[engine_name])

    return BacktestReport(
        total_cases=len(all_cases),
        positive_cases=positive_count,
        negative_cases=len(all_cases) - positive_count,
        engines=engines,
        engine_results=engine_results,
        calibration=cal,
        per_case=per_case,
        metadata={
            "start": start.isoformat(),
            "end": end.isoformat(),
            "horizon_days": horizon_days,
            "min_significance": min_significance,
        },
    )


# ── Engine dispatch ──────────────────────────────────────────────────


async def _run_engine(
    engine_name: str,
    store: KnowledgeStore,
    llm,
    question: str,
    *,
    cutoff: date,
    timeout_seconds: int = 120,
) -> float:
    """Run a single engine on a question with leakage-safe cutoff."""
    return await asyncio.wait_for(
        _run_engine_inner(engine_name, store, llm, question, cutoff=cutoff),
        timeout=timeout_seconds,
    )


async def _run_engine_inner(
    engine_name: str,
    store: KnowledgeStore,
    llm,
    question: str,
    *,
    cutoff: date,
) -> float:
    """Inner engine dispatch (wrapped with timeout by _run_engine)."""
    if engine_name == "structural":
        from nexus.engine.projection.evidence import assemble_evidence_package
        from nexus.engine.projection.structural_engine import predict_structural

        evidence = await assemble_evidence_package(store, question, as_of=cutoff)
        assessment = await predict_structural(llm, evidence)
        return assessment.implied_probability

    elif engine_name == "actor":
        from nexus.engine.projection.actor_engine import predict

        prediction = await predict(
            store, llm, question,
            run_date=cutoff, as_of=cutoff,
        )
        return prediction.calibrated_probability

    elif engine_name == "graphrag":
        from nexus.engine.projection.graphrag_engine import GraphRAGBenchmarkEngine

        eng = GraphRAGBenchmarkEngine()
        return await eng.predict_probability(
            question, llm=llm, store=store, as_of=cutoff,
        )

    elif engine_name == "naked":
        from nexus.engine.projection.naked_engine import NakedBenchmarkEngine

        eng = NakedBenchmarkEngine()
        return await eng.predict_probability(
            question, llm=llm, store=store, as_of=cutoff,
        )

    elif engine_name == "perspective":
        from nexus.engine.projection.perspective_engine import PerspectiveBenchmarkEngine

        eng = PerspectiveBenchmarkEngine()
        return await eng.predict_probability(
            question, llm=llm, store=store, as_of=cutoff,
        )

    elif engine_name == "debate":
        from nexus.engine.projection.debate_engine import DebateBenchmarkEngine

        eng = DebateBenchmarkEngine()
        return await eng.predict_probability(
            question, llm=llm, store=store, as_of=cutoff,
        )

    else:
        raise ValueError(f"Unknown engine: {engine_name}")


# ── Serialization ────────────────────────────────────────────────────


def serialize_report(report: BacktestReport) -> dict:
    """Convert BacktestReport to JSON-serializable dict."""
    return {
        "total_cases": report.total_cases,
        "positive_cases": report.positive_cases,
        "negative_cases": report.negative_cases,
        "engines": report.engines,
        "engine_results": report.engine_results,
        "calibration": report.calibration,
        "per_case": report.per_case,
        "metadata": report.metadata,
    }
