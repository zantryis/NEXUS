"""Actor-based prediction engine.

Real-world outcomes emerge from actors (countries, companies, leaders) interacting.
For each question: identify key actors → assemble per-actor knowledge from the DB →
LLM reasons about each actor's likely behavior (1 call per actor) → LLM synthesizes
into a probability (1 call) → calibrate. Total 3-6 LLM calls per question.

The model's world knowledge provides essential context. Our fresh news data provides
the freshness edge. When they conflict, our data wins.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import date, timedelta

from nexus.engine.knowledge.events import Event
from nexus.engine.knowledge.store import KnowledgeStore
from nexus.engine.projection.forecasting import (
    ForecastEngineInput,
    ForecastEngine,
    _clip_probability,
)
from nexus.engine.projection.models import (
    ForecastQuestion,
    ForecastRun,
)
from nexus.engine.projection.swarm import anchor_blend, derive_verdict, extremize
from nexus.llm.client import LLMClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class ActorKnowledge:
    """Assembled knowledge about a single actor."""

    name: str
    entity_id: int | None = None
    recent_events: list[Event] = field(default_factory=list)
    active_relationships: list[dict] = field(default_factory=list)
    recent_changes: list[dict] = field(default_factory=list)
    thread_context: list[dict] = field(default_factory=list)
    cross_topic_presence: list[dict] = field(default_factory=list)


@dataclass
class ActorAnalysis:
    """Result of reasoning about one actor's influence on an outcome."""

    actor: str
    direction: str  # increases | decreases | neutral
    magnitude: str  # small | moderate | large
    reasoning: str
    key_uncertainty: str
    probability_shift: float  # bounded to [-0.3, 0.3]


@dataclass
class ActorPrediction:
    """Final prediction combining all actor analyses."""

    question: str
    actors: list[ActorAnalysis]
    raw_probability: float
    calibrated_probability: float
    reasoning_chain: str = ""
    key_uncertainties: list[str] = field(default_factory=list)
    signposts: list[str] = field(default_factory=list)
    verdict: str | None = None
    confidence: str | None = None


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

ACTOR_REASONING_SYSTEM = (
    "You are an analyst reasoning about how a specific actor's position and recent "
    "actions affect the probability of a real-world outcome. Use both your world "
    "knowledge and the supplied intelligence. When our monitored intelligence conflicts "
    "with your prior knowledge, treat our data as ground truth — it is more recent."
)

ACTOR_REASONING_PROMPT = """\
Question: {question}

Actor: {actor_name}

The following intelligence about {actor_name} was collected from our automated news \
monitoring system. Some of this information may be more recent than your training data. \
When it conflicts with your prior knowledge, treat our data as ground truth.

Recent events involving {actor_name}:
{events_section}

Active relationships:
{relationships_section}

Recent changes (new or invalidated relationships, last 14 days):
{changes_section}

Given {actor_name}'s position, relationships, and recent actions, how does their \
behavior affect the probability of the outcome described in the question?

Return JSON only:
{{"direction": "increases|decreases|neutral", "magnitude": "small|moderate|large", \
"reasoning": "...", "key_uncertainty": "...", "probability_shift": <float -0.3 to 0.3>}}
"""

SYNTHESIS_SYSTEM = (
    "You are a prediction synthesizer combining multiple actor analyses into a single "
    "probability estimate. Weigh each actor's influence, consider their interactions, "
    "and produce a calibrated probability."
)

SYNTHESIS_PROMPT = """\
Question: {question}

Actor analyses:
{analyses_section}

{market_section}

Combine these actor analyses into a single probability for the question. Consider how \
the actors interact — alliances, opposition, leverage, timing. Be calibrated: 0.50 means \
genuinely uncertain, not a cop-out.

Return JSON only:
{{"probability": <float 0.05-0.95>, "reasoning": "...", \
"key_uncertainties": ["..."], "signposts": ["..."]}}
"""


# ---------------------------------------------------------------------------
# Pipeline functions
# ---------------------------------------------------------------------------


async def identify_actors(
    store: KnowledgeStore,
    question: str,
    *,
    max_actors: int = 5,
    as_of: date | None = None,
) -> list[dict]:
    """Extract key actors from question text, resolve against entity store.

    Zero LLM calls — pure text matching against stored entities.
    Returns list of {name, entity_id, match_source} dicts.
    """
    # Extract capitalized multi-word phrases and known proper nouns
    # This is a lightweight NER approach using our entity store as the dictionary
    all_entities = await store.get_all_entities()
    if not all_entities:
        return []

    scored: list[tuple[int, dict]] = []

    def _word_match(term: str) -> bool:
        return bool(re.search(r"\b" + re.escape(term) + r"\b", question, re.IGNORECASE))

    for entity in all_entities:
        name = entity.get("canonical_name") or entity.get("name", "")
        if not name or len(name) < 2:
            continue
        if _word_match(name):
            scored.append((2, {
                "name": name,
                "entity_id": entity.get("id"),
                "match_source": "canonical",
            }))
            continue
        # Check aliases
        aliases = entity.get("aliases") or []
        if isinstance(aliases, str):
            try:
                aliases = json.loads(aliases)
            except (json.JSONDecodeError, TypeError):
                aliases = [aliases]
        for alias in aliases:
            if isinstance(alias, str) and _word_match(alias):
                scored.append((1, {
                    "name": name,
                    "entity_id": entity.get("id"),
                    "match_source": "alias",
                }))
                break

    # Sort by score descending, deduplicate by entity_id
    scored.sort(key=lambda x: x[0], reverse=True)
    seen_ids: set[int | None] = set()
    result: list[dict] = []
    for _, actor in scored:
        eid = actor.get("entity_id")
        if eid in seen_ids:
            continue
        seen_ids.add(eid)
        result.append(actor)
        if len(result) >= max_actors:
            break

    return result


async def assemble_actor_knowledge(
    store: KnowledgeStore,
    actor: dict,
    *,
    as_of: date | None = None,
    event_days: int = 30,
    max_events: int = 10,
    max_relationships: int = 15,
) -> ActorKnowledge:
    """Assemble knowledge about an actor from the store. Zero LLM calls."""
    entity_id = actor.get("entity_id")
    name = actor["name"]

    if entity_id is None:
        return ActorKnowledge(name=name)

    # Recent events involving this entity
    recent_events = await store.get_events_for_entity(entity_id)
    if as_of:
        cutoff = as_of - timedelta(days=event_days)
        recent_events = [e for e in recent_events if cutoff <= e.date <= as_of]
    recent_events = recent_events[-max_events:]

    # Active relationships
    active_relationships = await store.get_active_relationships_for_entity(
        entity_id, as_of=as_of,
    )
    active_relationships = active_relationships[:max_relationships]

    # Recent relationship changes (new or invalidated)
    recent_changes = await store.get_relationship_timeline(
        entity_id, days=14, reference_date=as_of,
    )

    # Thread context
    thread_context = await store.get_threads_for_entity(entity_id)
    thread_context = thread_context[:3]

    # Cross-topic presence
    cross_topic = await store.get_related_entities(entity_id, limit=5)

    return ActorKnowledge(
        name=name,
        entity_id=entity_id,
        recent_events=recent_events,
        active_relationships=active_relationships,
        recent_changes=recent_changes,
        thread_context=thread_context,
        cross_topic_presence=cross_topic,
    )


def _format_events_section(events: list[Event]) -> str:
    if not events:
        return "(no recent events)"
    lines = []
    for e in events:
        lines.append(f"- [{e.date}] {e.summary}")
    return "\n".join(lines)


def _format_relationships_section(relationships: list[dict]) -> str:
    if not relationships:
        return "(no known relationships)"
    lines = []
    for r in relationships:
        source = r.get("source_name", "?")
        target = r.get("target_name", "?")
        rel_type = r.get("relation_type", "?")
        detail = r.get("detail", "")
        lines.append(f"- {source} --[{rel_type}]--> {target}: {detail}")
    return "\n".join(lines[:15])


def _format_changes_section(changes: list[dict]) -> str:
    if not changes:
        return "(no recent changes)"
    lines = []
    for c in changes:
        source = c.get("source_name", "?")
        target = c.get("target_name", "?")
        rel_type = c.get("relation_type", "?")
        valid_from = c.get("valid_from", "?")
        invalidated = c.get("invalidated_at")
        status = f"invalidated {invalidated}" if invalidated else f"since {valid_from}"
        lines.append(f"- {source} --[{rel_type}]--> {target} ({status})")
    return "\n".join(lines[:10])


async def reason_about_actor(
    llm: LLMClient,
    knowledge: ActorKnowledge,
    question: str,
) -> ActorAnalysis:
    """Reason about one actor's influence on the outcome. 1 LLM call."""
    prompt = ACTOR_REASONING_PROMPT.format(
        question=question,
        actor_name=knowledge.name,
        events_section=_format_events_section(knowledge.recent_events),
        relationships_section=_format_relationships_section(knowledge.active_relationships),
        changes_section=_format_changes_section(knowledge.recent_changes),
    )

    try:
        raw = await llm.complete(
            config_key="knowledge_summary",
            system_prompt=ACTOR_REASONING_SYSTEM,
            user_prompt=prompt,
            json_response=True,
        )
        cleaned = re.sub(r"```json\s*|\s*```", "", raw)
        cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL)
        data = json.loads(cleaned)

        direction = str(data.get("direction", "neutral")).lower()
        if direction not in {"increases", "decreases", "neutral"}:
            direction = "neutral"

        magnitude = str(data.get("magnitude", "small")).lower()
        if magnitude not in {"small", "moderate", "large"}:
            magnitude = "small"

        shift = float(data.get("probability_shift", 0.0))
        shift = max(-0.3, min(0.3, shift))

        return ActorAnalysis(
            actor=knowledge.name,
            direction=direction,
            magnitude=magnitude,
            reasoning=str(data.get("reasoning", "")),
            key_uncertainty=str(data.get("key_uncertainty", "")),
            probability_shift=shift,
        )
    except Exception as exc:
        logger.warning("Actor reasoning failed for %s: %s", knowledge.name, exc)
        return ActorAnalysis(
            actor=knowledge.name,
            direction="neutral",
            magnitude="small",
            reasoning=f"Analysis unavailable: {exc}",
            key_uncertainty="LLM failure",
            probability_shift=0.0,
        )


async def synthesize_prediction(
    llm: LLMClient,
    question: str,
    analyses: list[ActorAnalysis],
    *,
    market_prob: float | None = None,
) -> ActorPrediction:
    """Combine actor analyses into a single calibrated prediction. 1 LLM call."""
    analyses_lines = []
    for a in analyses:
        analyses_lines.append(
            f"- {a.actor}: direction={a.direction}, magnitude={a.magnitude}, "
            f"shift={a.probability_shift:+.2f}\n  Reasoning: {a.reasoning}\n  "
            f"Uncertainty: {a.key_uncertainty}"
        )

    market_section = ""
    if market_prob is not None:
        market_section = (
            f"Market implied probability: {market_prob:.2f}\n"
            f"Consider this as an anchor — the market aggregates many participants' views."
        )

    prompt = SYNTHESIS_PROMPT.format(
        question=question,
        analyses_section="\n".join(analyses_lines) or "(no actor analyses available)",
        market_section=market_section,
    )

    try:
        raw = await llm.complete(
            config_key="knowledge_summary",
            system_prompt=SYNTHESIS_SYSTEM,
            user_prompt=prompt,
            json_response=True,
        )
        cleaned = re.sub(r"```json\s*|\s*```", "", raw)
        cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL)
        data = json.loads(cleaned)

        raw_prob = float(data.get("probability", 0.5))
        raw_prob = max(0.05, min(0.95, raw_prob))

        # Canonical calibration path: compress then anchor.
        # gamma<1 compresses overconfident LLM outputs toward 0.5.
        # anchor_blend then blends with market price (market-anchored).
        calibrated = extremize(raw_prob, gamma=0.8)

        if market_prob is not None:
            calibrated = anchor_blend(calibrated, market_prob, swarm_weight=0.45)

        calibrated = max(0.05, min(0.95, round(calibrated, 3)))
        verdict, confidence = derive_verdict(calibrated)

        return ActorPrediction(
            question=question,
            actors=analyses,
            raw_probability=raw_prob,
            calibrated_probability=calibrated,
            reasoning_chain=str(data.get("reasoning", "")),
            key_uncertainties=data.get("key_uncertainties", []),
            signposts=data.get("signposts", []),
            verdict=verdict,
            confidence=confidence,
        )
    except Exception as exc:
        logger.warning("Synthesis failed for '%s': %s", question[:80], exc)
        # Fallback path (no calibration math): raw shift sum from base.
        # Less accurate than canonical path above — only used when LLM
        # synthesis JSON parsing fails.
        base = market_prob if market_prob is not None else 0.5
        shift_sum = sum(a.probability_shift for a in analyses)
        fallback = max(0.05, min(0.95, round(base + shift_sum, 3)))
        verdict, confidence = derive_verdict(fallback)
        return ActorPrediction(
            question=question,
            actors=analyses,
            raw_probability=fallback,
            calibrated_probability=fallback,
            reasoning_chain=f"Synthesis unavailable: {exc}",
            verdict=verdict,
            confidence=confidence,
        )


async def predict(
    store: KnowledgeStore,
    llm: LLMClient | None,
    question: str,
    *,
    run_date: date,
    market_prob: float | None = None,
    max_actors: int = 4,
    as_of: date | None = None,
) -> ActorPrediction:
    """Top-level entry point: full actor-based prediction pipeline.

    1. identify_actors — 0 LLM calls
    2. assemble_actor_knowledge — 0 LLM calls (store queries only)
    3. reason_about_actor — 1 LLM call per actor
    4. synthesize_prediction — 1 LLM call
    5. calibrate — 0 LLM calls
    """
    cutoff = as_of or run_date
    actors = await identify_actors(store, question, max_actors=max_actors, as_of=cutoff)

    if not actors:
        # No entities found — return baseline prediction
        base = market_prob if market_prob is not None else 0.5
        return ActorPrediction(
            question=question,
            actors=[],
            raw_probability=base,
            calibrated_probability=max(0.05, min(0.95, round(base, 3))),
            reasoning_chain="No relevant actors identified in knowledge base.",
        )

    # Assemble knowledge for each actor
    actor_knowledge_list: list[ActorKnowledge] = []
    for actor in actors:
        knowledge = await assemble_actor_knowledge(store, actor, as_of=cutoff)
        actor_knowledge_list.append(knowledge)

    if llm is None:
        # Deterministic fallback: use heuristics from actor knowledge
        analyses = []
        for knowledge in actor_knowledge_list:
            direction = "neutral"
            shift = 0.0
            if knowledge.recent_events:
                shift = min(0.1, len(knowledge.recent_events) * 0.02)
                direction = "increases"
            analyses.append(ActorAnalysis(
                actor=knowledge.name,
                direction=direction,
                magnitude="small",
                reasoning=f"Heuristic: {len(knowledge.recent_events)} recent events.",
                key_uncertainty="No LLM analysis available.",
                probability_shift=shift,
            ))
        base = market_prob if market_prob is not None else 0.5
        total_shift = sum(a.probability_shift for a in analyses)
        calibrated = max(0.05, min(0.95, round(base + total_shift, 3)))
        return ActorPrediction(
            question=question,
            actors=analyses,
            raw_probability=calibrated,
            calibrated_probability=calibrated,
            reasoning_chain="Deterministic fallback (no LLM).",
        )

    # LLM path: per-actor reasoning then synthesis
    analyses = []
    for knowledge in actor_knowledge_list:
        analysis = await reason_about_actor(llm, knowledge, question)
        analyses.append(analysis)

    prediction = await synthesize_prediction(
        llm, question, analyses, market_prob=market_prob,
    )
    return prediction


# ---------------------------------------------------------------------------
# ForecastEngine protocol implementation
# ---------------------------------------------------------------------------


class ActorForecastEngine:
    """Actor-based forecast engine implementing the ForecastEngine protocol.

    When called with a store, uses the full actor pipeline.
    When called without store (legacy path), generates from thread heuristics.
    """

    engine_name = "actor"

    async def generate(
        self,
        llm: LLMClient | None,
        payload: ForecastEngineInput,
        *,
        critic_pass: bool = True,
        max_questions: int = 4,
        calibration_data: list[dict] | None = None,
        store: KnowledgeStore | None = None,
    ) -> ForecastRun:
        questions: list[ForecastQuestion] = []

        # If we have a store, use the actor pipeline to generate questions
        # from entity-based analysis. Otherwise fall back to thread heuristics.
        if store is not None and payload.recent_events:
            # Build questions from the most significant recent events/entities
            entity_counts: dict[str, int] = {}
            for event in payload.recent_events:
                for entity in event.entities:
                    entity_counts[entity] = entity_counts.get(entity, 0) + 1

            # Top entities as question subjects
            top_entities = sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)[:3]

            for entity_name, count in top_entities:
                q_text = (
                    f"Will {entity_name} be involved in significant new developments "
                    f"related to {payload.topic_name} within 7 days?"
                )
                try:
                    prediction = await predict(
                        store, llm, q_text,
                        run_date=payload.run_date,
                        max_actors=3,
                    )
                    questions.append(ForecastQuestion(
                        question=q_text,
                        forecast_type="binary",
                        target_variable="actor_development",
                        probability=prediction.calibrated_probability,
                        base_rate=prediction.raw_probability,
                        resolution_criteria=(
                            f"Resolves true if significant events involving "
                            f"{entity_name} occur within 7 days."
                        ),
                        resolution_date=payload.run_date + timedelta(days=7),
                        horizon_days=7,
                        signpost=prediction.signposts[0] if prediction.signposts else f"Activity involving {entity_name}",
                        signals_cited=[
                            f"actor:{a.actor}:{a.direction}" for a in prediction.actors
                        ],
                        evidence_event_ids=[
                            e.event_id for e in payload.recent_events
                            if e.event_id and entity_name in e.entities
                        ][:8],
                        evidence_thread_ids=[],
                        target_metadata={
                            "topic_slug": payload.topic_slug,
                            "actor_analysis": True,
                        },
                    ))
                except Exception as exc:
                    logger.warning("Actor forecast failed for %s: %s", entity_name, exc)

                if len(questions) >= max_questions:
                    break

        # Fallback: thread-based heuristic questions (same as stub)
        if not questions:
            questions = self._thread_heuristic_questions(payload, max_questions)

        return ForecastRun(
            topic_slug=payload.topic_slug,
            topic_name=payload.topic_name,
            engine=self.engine_name,
            generated_for=payload.run_date,
            summary=f"{payload.topic_name} forecast generated by actor engine.",
            questions=questions,
            metadata={"actor_based": bool(store)},
        )
    def _thread_heuristic_questions(
        self, payload: ForecastEngineInput, max_questions: int,
    ) -> list[ForecastQuestion]:
        """Deterministic fallback from thread trajectories."""
        questions: list[ForecastQuestion] = []
        threads = sorted(
            payload.threads,
            key=lambda t: (t.momentum_score or 0.0, t.significance),
            reverse=True,
        )
        for thread in threads[:max_questions]:
            label = thread.trajectory_label or "steady"
            momentum = thread.momentum_score or 0.0
            if label == "about_to_break":
                base_prob = 0.75
            elif label == "accelerating":
                base_prob = 0.65
            elif label == "steady":
                base_prob = 0.50
            elif label == "cooling":
                base_prob = 0.35
            else:
                base_prob = 0.45
            probability = _clip_probability(base_prob)
            horizon_days = 3 if label == "about_to_break" else 7
            questions.append(ForecastQuestion(
                question=(
                    f"Will {thread.headline} produce significant new developments "
                    f"within {horizon_days} days?"
                ),
                forecast_type="binary",
                target_variable="thread_development",
                probability=probability,
                base_rate=probability,
                resolution_criteria=(
                    f"Resolves true if significant new events occur for "
                    f"'{thread.headline}' within {horizon_days} days."
                ),
                resolution_date=payload.run_date + timedelta(days=horizon_days),
                horizon_days=horizon_days,
                signpost=(
                    thread.events[-1].summary if thread.events
                    else f"New reporting tied to {thread.headline}"
                ),
                signals_cited=[
                    f"trajectory:{label}",
                    f"momentum:{round(momentum, 2)}",
                ],
                evidence_event_ids=[
                    e.event_id for e in thread.events if e.event_id
                ][:8],
                evidence_thread_ids=[thread.thread_id] if thread.thread_id else [],
                target_metadata={
                    "topic_slug": payload.topic_slug,
                    "thread_id": thread.thread_id,
                },
            ))
        return questions


# ---------------------------------------------------------------------------
# BenchmarkEngine adapter
# ---------------------------------------------------------------------------


class ActorBenchmarkEngine:
    """Adapter satisfying the BenchmarkEngine protocol for Kalshi benchmarks."""

    engine_name = "actor"

    async def predict_probability(
        self,
        question: str,
        *,
        llm=None,
        store=None,
        market_prob: float | None = None,
        as_of: date | None = None,
    ) -> float:
        if llm is None or store is None:
            return 0.50
        pred = await predict(
            store, llm, question,
            run_date=as_of or date.today(),
            market_prob=market_prob,
            as_of=as_of,
        )
        return pred.calibrated_probability
