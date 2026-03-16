"""Quantified forecast engines and projection rendering adapters."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Protocol

from nexus.engine.knowledge.events import Event
from nexus.engine.projection.graph import rank_graph_evidence
from nexus.engine.projection.models import (
    CrossTopicSignal,
    ForecastQuestion,
    ForecastRun,
    GraphSnapshot,
    TopicProjection,
    confidence_from_probability,
)
from nexus.engine.synthesis.knowledge import NarrativeThread
from nexus.llm.client import LLMClient

logger = logging.getLogger(__name__)

FORECAST_ANALYST_SYSTEM_PROMPT = (
    "You are a quantified forecasting analyst. You will receive candidate forecast specs that already "
    "have machine-resolvable targets and evidence. Choose up to the allowed number of candidates, rewrite "
    "them into clearer human-readable questions, assign a probability between 0.05 and 0.95, and produce "
    "a short summary. Use only the provided candidate keys. Return JSON only in the form "
    "{\"summary\": \"...\", \"questions\": [{\"candidate_key\": \"...\", \"question\": \"...\", "
    "\"probability\": 0.64, \"signpost\": \"...\"}]}"
)

FORECAST_CRITIC_SYSTEM_PROMPT = (
    "You are a forecasting critic. Tighten weak phrasing, reduce unsupported probabilities, and ensure the "
    "question remains aligned with the candidate key. Return the same JSON schema."
)

FAMILY_TARGET_SPECS = {
    "official_statement_event": {
        "keywords": [
            "statement", "briefing", "guidance", "policy", "order", "announcement",
            "department", "ministry", "treasury", "pentagon", "centcom", "command",
        ],
        "base_rate": 0.70,
        "description": "official statement, briefing, or policy move",
        "question": "Will {anchor_text} produce a new official statement, briefing, or policy move in {topic_name} by {resolution_date}?",
    },
    "legal_action_event": {
        "keywords": [
            "court", "lawsuit", "judge", "ruling", "permit", "regulatory", "filing", "appeal",
            "injunction", "administrative",
        ],
        "base_rate": 0.45,
        "description": "court, permit, or regulatory action",
        "question": "Will {anchor_text} trigger a new court, permit, or regulatory action in {topic_name} by {resolution_date}?",
    },
    "partnership_or_product_event": {
        "keywords": [
            "partnership", "pilot", "launch", "agreement", "release", "deploy", "collaboration",
            "contract", "rollout", "commercial",
        ],
        "base_rate": 0.55,
        "description": "partnership, pilot, launch, or product release",
        "question": "Will {anchor_text} lead to a new partnership, pilot, launch, or product-release event in {topic_name} by {resolution_date}?",
    },
    "infrastructure_milestone_event": {
        "keywords": [
            "construction", "plant", "deployment", "project", "facility", "infrastructure",
            "grid", "factory", "corridor", "terminal", "power", "data center",
        ],
        "base_rate": 0.22,
        "description": "deployment, construction, or infrastructure milestone",
        "question": "Will {anchor_text} reach a new deployment, construction, or infrastructure milestone in {topic_name} by {resolution_date}?",
    },
}


@dataclass
class ForecastEngineInput:
    """Normalized topic state passed into forecast engines."""

    topic_slug: str
    topic_name: str
    run_date: date
    threads: list[NarrativeThread] = field(default_factory=list)
    recent_events: list[Event] = field(default_factory=list)
    cross_topic_signals: list[CrossTopicSignal] = field(default_factory=list)
    graph_snapshot: GraphSnapshot | None = None
    metadata: dict = field(default_factory=dict)


class ForecastEngine(Protocol):
    """Common contract for quantified forecast engines."""

    engine_name: str

    async def generate(
        self,
        llm: LLMClient | None,
        payload: ForecastEngineInput,
        *,
        critic_pass: bool = True,
        max_questions: int = 4,
        calibration_data: list[dict] | None = None,
    ) -> ForecastRun:
        """Produce a structured forecast run."""


def _clip_probability(value: float) -> float:
    return round(max(0.05, min(0.95, value)), 3)


def _empirical_adjusted_base_rate(
    hardcoded_rate: float,
    historical_rows: list[dict],
    *,
    min_samples: int = 10,
) -> float:
    """Bayesian shrinkage: blend hardcoded prior with empirical hit rate.

    Below min_samples, returns prior unchanged.
    Otherwise, weight = min(n / 50, 0.7) and blended = weight * empirical + (1 - weight) * prior.
    """
    n = len(historical_rows)
    if n < min_samples:
        return hardcoded_rate
    empirical = sum(1 for r in historical_rows if r["resolved_bool"]) / n
    weight = min(n / 50.0, 0.7)
    return round(weight * empirical + (1.0 - weight) * hardcoded_rate, 4)


def _thread_base_rate(thread: NarrativeThread) -> float:
    return 0.25


def _thread_trajectory_base_rate(thread: NarrativeThread) -> float:
    label = thread.trajectory_label or "steady"
    mapping = {
        "about_to_break": 0.45,
        "accelerating": 0.35,
        "steady": 0.22,
        "decelerating": 0.12,
    }
    rate = mapping.get(label, 0.4)
    if thread.snapshot_count and thread.snapshot_count >= 5:
        rate += 0.01
    return _clip_probability(rate)


def _topic_direction(payload: ForecastEngineInput, variant: str) -> tuple[str, float, float]:
    momenta = [thread.momentum_score or 0.0 for thread in payload.threads if thread.snapshot_count]
    avg_momentum = sum(momenta) / len(momenta) if momenta else 0.0
    expected_direction = "up" if avg_momentum >= 0 else "down"
    if variant == "baseline":
        expected_direction = "up"
    base_rate = 0.52 if expected_direction == "up" else 0.48
    probability = base_rate
    if variant in {"trajectory", "native", "kuzu"}:
        probability = _clip_probability(base_rate + max(-0.08, min(0.08, avg_momentum / 20.0)))
    return expected_direction, base_rate, probability


def _candidate_key(question: ForecastQuestion) -> str:
    if question.target_variable == "thread_new_event_count":
        return f"thread:{question.target_metadata['thread_id']}:{question.horizon_days}"
    if question.target_variable == "cross_topic_follow_on":
        return (
            f"bridge:{question.target_metadata['entity'].lower()}:"
            f"{question.target_metadata['related_topic_slug']}:{question.horizon_days}"
        )
    if question.target_variable == "entity_recurrence":
        return f"entity:{question.target_metadata['entity'].lower()}:{question.horizon_days}"
    if question.target_variable in FAMILY_TARGET_SPECS:
        anchors = ",".join(sorted(entity.lower() for entity in question.target_metadata.get("anchor_entities", [])))
        return f"{question.target_variable}:{anchors}:{question.horizon_days}"
    return f"topic:{question.target_metadata['topic_slug']}:{question.horizon_days}:{question.expected_direction or 'na'}"


def _keyword_hits(text: str, keywords: list[str]) -> int:
    haystack = text.lower()
    return sum(1 for keyword in keywords if keyword in haystack)


def _event_anchor_entities(event: Event, *, max_entities: int = 2) -> list[str]:
    anchors: list[str] = []
    for entity in event.entities:
        normalized = entity.strip()
        if not normalized:
            continue
        if normalized.lower() in {anchor.lower() for anchor in anchors}:
            continue
        anchors.append(normalized)
        if len(anchors) >= max_entities:
            break
    return anchors


def _anchor_text(event: Event) -> str:
    anchors = _event_anchor_entities(event)
    if anchors:
        return " and ".join(anchors)
    return event.summary[:80].rstrip(".")


def _family_probability(
    family: str,
    variant: str,
    *,
    context_momentum: float,
    graph_bonus: float = 0.0,
) -> tuple[float, float]:
    base_rate = FAMILY_TARGET_SPECS[family]["base_rate"]
    probability = base_rate
    if variant in {"trajectory", "native", "kuzu"}:
        probability = _clip_probability(base_rate + max(-0.08, min(0.1, context_momentum / 25.0)))
    if variant == "native":
        probability = _clip_probability(probability + max(-0.02, min(0.04, context_momentum / 50.0)))
    if variant == "kuzu":
        probability = _clip_probability(probability + graph_bonus)
    return base_rate, probability


def _build_candidate_catalog(
    payload: ForecastEngineInput,
    variant: str,
    max_questions: int,
    calibration_data: list[dict] | None = None,
) -> list[ForecastQuestion]:
    candidates: list[tuple[int, ForecastQuestion]] = []
    sorted_threads = sorted(
        [thread for thread in payload.threads if thread.thread_id and thread.snapshot_count],
        key=lambda thread: (thread.momentum_score or 0.0, thread.significance),
        reverse=True,
    )
    thread_by_event_id = {
        event.event_id: thread
        for thread in sorted_threads
        for event in thread.events
        if event.event_id is not None
    }

    graph_evidence = rank_graph_evidence(payload.graph_snapshot) if payload.graph_snapshot else {
        "event_ids": [],
        "thread_ids": [],
        "signal_ids": [],
        "signals_cited": [],
    }
    avg_topic_momentum = (
        sum((thread.momentum_score or 0.0) for thread in sorted_threads) / len(sorted_threads)
        if sorted_threads else 0.0
    )

    def add_candidate(priority: int, candidate: ForecastQuestion) -> None:
        candidates.append((priority, candidate))

    family_best_events: dict[str, tuple[int, Event]] = {}
    for event in payload.recent_events:
        summary = event.summary.lower()
        for family, spec in FAMILY_TARGET_SPECS.items():
            score = _keyword_hits(summary, spec["keywords"])
            if score <= 0:
                continue
            weighted_score = score * 10 + int(event.significance)
            current = family_best_events.get(family)
            if current is None or weighted_score > current[0]:
                family_best_events[family] = (weighted_score, event)

    ranked_families = sorted(
        (
            (score, family, event)
            for family, (score, event) in family_best_events.items()
        ),
        reverse=True,
    )[:2]

    for idx, (_, family, event) in enumerate(ranked_families):
        source_thread = thread_by_event_id.get(event.event_id)
        context_momentum = source_thread.momentum_score if source_thread and source_thread.momentum_score is not None else avg_topic_momentum
        graph_bonus = 0.04 if event.event_id in graph_evidence["event_ids"] else 0.0
        base_rate, probability = _family_probability(
            family,
            variant,
            context_momentum=context_momentum or 0.0,
            graph_bonus=graph_bonus,
        )
        anchor_entities = _event_anchor_entities(event)
        hot_signal = (
            event.significance >= 8
            or (source_thread and (source_thread.trajectory_label == "about_to_break"))
            or (context_momentum or 0.0) >= 5.0
        )
        horizon_days = 3 if hot_signal else 7
        resolution_date = payload.run_date + timedelta(days=horizon_days)
        spec = FAMILY_TARGET_SPECS[family]
        add_candidate(120 - idx * 5, ForecastQuestion(
            question=spec["question"].format(
                anchor_text=_anchor_text(event),
                topic_name=payload.topic_name,
                resolution_date=resolution_date.isoformat(),
            ),
            forecast_type="binary",
            target_variable=family,
            target_metadata={
                "topic_slug": payload.topic_slug,
                "keywords": spec["keywords"],
                "anchor_entities": anchor_entities,
                "source_event_id": event.event_id,
            },
            probability=probability,
            base_rate=base_rate,
            resolution_criteria=(
                f"Resolves true if a new {payload.topic_name} event by {resolution_date.isoformat()} contains "
                f"at least one anchor entity from {anchor_entities or ['the source event']} and matches the "
                f"{spec['description']} keyword set."
            ),
            resolution_date=resolution_date,
            horizon_days=horizon_days,
            signpost=event.summary,
            signals_cited=[
                f"signal-family:{family}",
                f"source-event:{event.summary[:100]}",
            ] + ([f"trajectory:{source_thread.trajectory_label}"] if source_thread and source_thread.trajectory_label else []),
            evidence_event_ids=[event.event_id] if event.event_id else [],
            evidence_thread_ids=[source_thread.thread_id] if source_thread and source_thread.thread_id else [],
            cross_topic_signal_ids=[],
        ))

    for thread in sorted_threads[:2]:
        horizons = [3, 7] if (thread.trajectory_label or "steady") == "about_to_break" else [7]
        for horizon_days in horizons:
            base_rate = _thread_base_rate(thread)
            probability = base_rate
            if variant in {"trajectory", "native", "kuzu"}:
                base_rate = _thread_trajectory_base_rate(thread)
                probability = base_rate
            if variant == "native":
                momentum_adj = max(-0.08, min(0.10, (thread.momentum_score or 0.0) / 30.0))
                probability = _clip_probability(base_rate + momentum_adj)
            elif variant == "kuzu":
                evidence_bonus = 0.04 if thread.thread_id in graph_evidence["thread_ids"] else 0.0
                probability = _clip_probability(base_rate + evidence_bonus + max(-0.08, min(0.12, (thread.momentum_score or 0.0) / 25.0)))

            latest_summary = thread.events[-1].summary if thread.events else f"New reporting tied to {thread.headline}"
            add_candidate(90, ForecastQuestion(
                question=f"Will {thread.headline} add at least one new linked event by {(payload.run_date + timedelta(days=horizon_days)).isoformat()}?",
                forecast_type="binary",
                target_variable="thread_new_event_count",
                target_metadata={
                    "thread_id": thread.thread_id,
                    "topic_slug": payload.topic_slug,
                    "threshold": 1,
                },
                probability=probability,
                base_rate=base_rate,
                resolution_criteria=f"Resolves true if thread {thread.thread_id} gains at least 1 new linked event by the resolution date.",
                resolution_date=payload.run_date + timedelta(days=horizon_days),
                horizon_days=horizon_days,
                signpost=latest_summary,
                signals_cited=[
                    f"trajectory:{thread.trajectory_label or 'steady'}",
                    f"momentum:{round(thread.momentum_score or 0.0, 2)}",
                    f"latest-event:{latest_summary[:100]}",
                ],
                evidence_event_ids=[event.event_id for event in thread.events if event.event_id][:8],
                evidence_thread_ids=[thread.thread_id],
                cross_topic_signal_ids=[],
            ))

    expected_direction, base_rate, probability = _topic_direction(payload, variant)
    if variant == "kuzu" and payload.graph_snapshot:
        bridge_bonus = min(0.08, len(payload.cross_topic_signals) * 0.02)
        probability = _clip_probability(probability + bridge_bonus)
    add_candidate(55, ForecastQuestion(
        question=(
            f"Will activity in {payload.topic_name} trend {expected_direction} over the next 7 days "
            f"relative to the trailing 7-day window?"
        ),
        forecast_type="directional",
        target_variable="topic_event_growth",
        target_metadata={"topic_slug": payload.topic_slug},
        probability=probability,
        base_rate=base_rate,
        resolution_criteria=(
            f"Resolves {expected_direction} if event count in {payload.topic_slug} from "
            f"{(payload.run_date + timedelta(days=1)).isoformat()} to {(payload.run_date + timedelta(days=7)).isoformat()} "
            f"is {'greater' if expected_direction == 'up' else 'less'} than the prior 7-day window."
        ),
        resolution_date=payload.run_date + timedelta(days=7),
        horizon_days=7,
        signpost="A visible change in event tempo across tracked sources.",
        expected_direction=expected_direction,
        signals_cited=[
            f"topic-momentum:{round(sum((thread.momentum_score or 0.0) for thread in sorted_threads), 2)}",
            f"recent-events:{len(payload.recent_events)}",
        ],
        evidence_event_ids=[event.event_id for event in payload.recent_events if event.event_id][-8:],
        evidence_thread_ids=[thread.thread_id for thread in sorted_threads[:3] if thread.thread_id],
        cross_topic_signal_ids=[],
    ))

    for signal in payload.cross_topic_signals[:2]:
        base_rate = 0.46
        probability = base_rate
        if variant in {"native", "kuzu"}:
            signal_strength = min(0.06, len(signal.event_ids) * 0.02)
            probability = _clip_probability(base_rate + signal_strength + (0.04 if signal.signal_id in graph_evidence["signal_ids"] else 0.0))
        add_candidate(70, ForecastQuestion(
            question=(
                f"Will {signal.shared_entity} drive follow-on activity in {signal.related_topic_slug.replace('-', ' ')} "
                f"within the next 7 days?"
            ),
            forecast_type="binary",
            target_variable="cross_topic_follow_on",
            target_metadata={
                "entity": signal.shared_entity,
                "related_topic_slug": signal.related_topic_slug,
            },
            probability=probability,
            base_rate=base_rate,
            resolution_criteria=(
                f"Resolves true if a new event mentioning {signal.shared_entity} appears in "
                f"{signal.related_topic_slug} by the resolution date."
            ),
            resolution_date=payload.run_date + timedelta(days=7),
            horizon_days=7,
            signpost=signal.note,
            signals_cited=[
                f"cross-topic:{signal.shared_entity}",
                signal.note,
            ],
            evidence_event_ids=(signal.event_ids + signal.related_event_ids)[:8],
            evidence_thread_ids=[],
            cross_topic_signal_ids=[signal.signal_id] if signal.signal_id else [],
        ))

    entity_candidates = [
        node["label"]
        for node in (payload.graph_snapshot.nodes if payload.graph_snapshot else [])
        if node["type"] == "entity"
    ]
    if variant == "kuzu" and entity_candidates:
        entity_name = entity_candidates[0]
        add_candidate(50, ForecastQuestion(
            question=f"Will {entity_name} recur in a new {payload.topic_name} event within the next 7 days?",
            forecast_type="binary",
            target_variable="entity_recurrence",
            target_metadata={"entity": entity_name, "topic_slug": payload.topic_slug},
            probability=0.61,
            base_rate=0.5,
            resolution_criteria=(
                f"Resolves true if a new event in {payload.topic_slug} mentions {entity_name} by the resolution date."
            ),
            resolution_date=payload.run_date + timedelta(days=7),
            horizon_days=7,
            signpost=f"Fresh reporting that mentions {entity_name}.",
            signals_cited=[f"graph-entity:{entity_name}"] + graph_evidence["signals_cited"][:1],
            evidence_event_ids=graph_evidence["event_ids"],
            evidence_thread_ids=graph_evidence["thread_ids"],
            cross_topic_signal_ids=graph_evidence["signal_ids"],
        ))

    deduped: dict[str, tuple[int, ForecastQuestion]] = {}
    for priority, candidate in candidates:
        key = _candidate_key(candidate)
        current = deduped.get(key)
        if current is None or priority > current[0]:
            deduped[key] = (priority, candidate)
    return [
        candidate
        for _, candidate in sorted(
            deduped.values(),
            key=lambda item: (item[0], item[1].probability),
            reverse=True,
        )[:max_questions]
    ]


def _render_candidate_prompt(payload: ForecastEngineInput, candidates: list[ForecastQuestion], max_questions: int) -> str:
    lines = [
        f"Topic: {payload.topic_name}",
        f"Run date: {payload.run_date.isoformat()}",
        f"Return at most {max_questions} questions.",
        "",
        "Candidate forecast specs:",
    ]
    for candidate in candidates:
        key = _candidate_key(candidate)
        lines.append(
            f"- key={key} | type={candidate.forecast_type} | probability={candidate.probability} | "
            f"base_rate={candidate.base_rate} | question={candidate.question}"
        )
        lines.append(f"  resolution={candidate.resolution_criteria}")
        lines.append(f"  signpost={candidate.signpost}")
    return "\n".join(lines)


async def _refine_candidates_with_llm(
    llm: LLMClient,
    payload: ForecastEngineInput,
    candidates: list[ForecastQuestion],
    *,
    critic_pass: bool,
    max_questions: int,
) -> tuple[str, list[ForecastQuestion]]:
    prompt = _render_candidate_prompt(payload, candidates, max_questions)
    candidate_map = {_candidate_key(candidate): candidate for candidate in candidates}
    analyst_raw = await llm.complete(
        config_key="knowledge_summary",
        system_prompt=FORECAST_ANALYST_SYSTEM_PROMPT,
        user_prompt=prompt,
        json_response=True,
    )
    candidate_payload = json.loads(analyst_raw)
    if critic_pass:
        critic_raw = await llm.complete(
            config_key="knowledge_summary",
            system_prompt=FORECAST_CRITIC_SYSTEM_PROMPT,
            user_prompt=json.dumps(candidate_payload, indent=2),
            json_response=True,
        )
        candidate_payload = json.loads(critic_raw)

    refined_questions: list[ForecastQuestion] = []
    for item in candidate_payload.get("questions", [])[:max_questions]:
        key = item.get("candidate_key", "")
        base_candidate = candidate_map.get(key)
        if not base_candidate:
            continue
        refined_questions.append(base_candidate.model_copy(update={
            "question": item.get("question", base_candidate.question),
            "probability": _clip_probability(float(item.get("probability", base_candidate.probability))),
            "signpost": item.get("signpost", base_candidate.signpost),
        }))

    summary = candidate_payload.get("summary", "")
    return summary, refined_questions or candidates


class BaselineForecastEngine:
    """Deterministic heuristic baseline."""

    engine_name = "baseline"

    async def generate(
        self,
        llm: LLMClient | None,
        payload: ForecastEngineInput,
        *,
        critic_pass: bool = True,
        max_questions: int = 4,
        calibration_data: list[dict] | None = None,
    ) -> ForecastRun:
        questions = _build_candidate_catalog(payload, "baseline", max_questions, calibration_data=calibration_data)
        return ForecastRun(
            topic_slug=payload.topic_slug,
            topic_name=payload.topic_name,
            engine=self.engine_name,
            generated_for=payload.run_date,
            summary=f"Deterministic baseline forecast for {payload.topic_name}.",
            questions=questions,
            metadata={"baseline": True},
        )


class TrajectoryForecastEngine:
    """Deterministic trajectory-only baseline."""

    engine_name = "trajectory"

    async def generate(
        self,
        llm: LLMClient | None,
        payload: ForecastEngineInput,
        *,
        critic_pass: bool = True,
        max_questions: int = 4,
        calibration_data: list[dict] | None = None,
    ) -> ForecastRun:
        questions = _build_candidate_catalog(payload, "trajectory", max_questions, calibration_data=calibration_data)
        return ForecastRun(
            topic_slug=payload.topic_slug,
            topic_name=payload.topic_name,
            engine=self.engine_name,
            generated_for=payload.run_date,
            summary=f"Trajectory-only forecast for {payload.topic_name}.",
            questions=questions,
            metadata={"trajectory_only": True},
        )


class NativeForecastEngine:
    """Primary quantified engine using deterministic proposals plus optional LLM refinement."""

    engine_name = "native"

    async def generate(
        self,
        llm: LLMClient | None,
        payload: ForecastEngineInput,
        *,
        critic_pass: bool = True,
        max_questions: int = 4,
        calibration_data: list[dict] | None = None,
    ) -> ForecastRun:
        candidates = _build_candidate_catalog(payload, "native", max_questions, calibration_data=calibration_data)
        summary = f"Quantified forecast for {payload.topic_name}."
        questions = candidates
        if llm is not None:
            try:
                summary, questions = await _refine_candidates_with_llm(
                    llm,
                    payload,
                    candidates,
                    critic_pass=critic_pass,
                    max_questions=max_questions,
                )
            except Exception as exc:
                logger.warning("Native forecast refinement failed for %s: %s", payload.topic_slug, exc)
        return ForecastRun(
            topic_slug=payload.topic_slug,
            topic_name=payload.topic_name,
            engine=self.engine_name,
            generated_for=payload.run_date,
            summary=summary,
            questions=questions,
            metadata={"graph_snapshot": bool(payload.graph_snapshot)},
        )


class KuzuForecastEngine:
    """Graph-enhanced retrieval variant for benchmark comparisons."""

    engine_name = "kuzu"

    async def generate(
        self,
        llm: LLMClient | None,
        payload: ForecastEngineInput,
        *,
        critic_pass: bool = True,
        max_questions: int = 4,
        calibration_data: list[dict] | None = None,
    ) -> ForecastRun:
        candidates = _build_candidate_catalog(payload, "kuzu", max_questions, calibration_data=calibration_data)
        summary = f"Graph-enhanced forecast for {payload.topic_name}."
        questions = candidates
        if llm is not None:
            try:
                summary, questions = await _refine_candidates_with_llm(
                    llm,
                    payload,
                    candidates,
                    critic_pass=critic_pass,
                    max_questions=max_questions,
                )
            except Exception as exc:
                logger.warning("Kuzu forecast refinement failed for %s: %s", payload.topic_slug, exc)
        return ForecastRun(
            topic_slug=payload.topic_slug,
            topic_name=payload.topic_name,
            engine=self.engine_name,
            generated_for=payload.run_date,
            summary=summary,
            questions=questions,
            metadata={"graph_enhanced": True},
        )


def get_forecast_engine(engine_name: str) -> ForecastEngine:
    """Resolve a quantified forecast engine by name."""
    normalized = engine_name.strip().lower()
    if normalized == "native":
        return NativeForecastEngine()
    if normalized == "baseline":
        return BaselineForecastEngine()
    if normalized == "trajectory":
        return TrajectoryForecastEngine()
    if normalized == "kuzu":
        return KuzuForecastEngine()
    if normalized == "swarm":
        from nexus.engine.projection.swarm import SwarmForecastEngine
        return SwarmForecastEngine()
    if normalized == "graph":
        from nexus.engine.projection.graph_engine import GraphForecastEngine
        return GraphForecastEngine()
    raise ValueError(f"Unknown forecast engine: {engine_name}")


def projection_from_forecast_run(run: ForecastRun, cross_topic_signals: list[CrossTopicSignal]) -> TopicProjection:
    """Render the user-facing projection artifact from a structured forecast run."""
    return TopicProjection(
        topic_slug=run.topic_slug,
        topic_name=run.topic_name,
        engine=run.engine,
        generated_for=run.generated_for,
        status="ready" if run.questions else "insufficient_history",
        summary=run.summary,
        items=[
            {
                "claim": question.question,
                "confidence": confidence_from_probability(question.probability),
                "horizon_days": question.horizon_days,
                "signpost": question.signpost,
                "signals_cited": question.signals_cited,
                "evidence_event_ids": question.evidence_event_ids,
                "evidence_thread_ids": question.evidence_thread_ids,
                "review_after": question.resolution_date,
                "external_ref": question.external_ref,
            }
            for question in run.questions
        ],
        cross_topic_signals=cross_topic_signals[:5],
        metadata={"forecast_engine": run.engine, "forecast_question_count": len(run.questions), **run.metadata},
    )
