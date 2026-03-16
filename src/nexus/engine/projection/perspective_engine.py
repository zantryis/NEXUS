"""Multi-perspective prediction engine — MiroFish-inspired debate + mathematical aggregation.

Instead of OASIS's agent simulation on fake social media, we:
1. Generate 3-5 analyst personas with distinct analytical lenses (1 LLM call)
2. Each persona reasons independently about the question (3-5 LLM calls)
3. Aggregate via geometric_mean_of_odds (0 LLM calls) — mathematically optimal

This captures the value of multiple perspectives without the expense or herd behavior
of full agent simulation. Total: 4-6 LLM calls per question.
"""

from __future__ import annotations

import json
import logging
from datetime import date, timedelta

from nexus.engine.projection.forecasting import ForecastEngineInput, _clip_probability
from nexus.engine.projection.models import ForecastQuestion, ForecastRun
from nexus.engine.projection.swarm import anchor_blend, extremize, geometric_mean_of_odds

logger = logging.getLogger(__name__)


# ── Prompts ───────────────────────────────────────────────────────────

PERSONA_GENERATION_SYSTEM = (
    "You generate diverse analyst personas for structured forecasting. "
    "Each persona has a distinct analytical framework that leads to different "
    "conclusions. Return JSON only."
)

PERSONA_GENERATION_PROMPT = (
    "Question: {question}\n\n"
    "Generate {num_personas} analyst personas with different analytical perspectives "
    "for evaluating this question. Each should have a genuinely different approach "
    "(e.g., hawkish vs dovish, quantitative vs qualitative, optimistic vs contrarian, "
    "historical-pattern vs current-momentum).\n\n"
    "Return JSON:\n"
    '{{"personas": [{{"name": "...", "perspective": "one-line description of their analytical lens"}}]}}'
)

PERSONA_REASONING_SYSTEM = (
    "You are {persona_name}, an analyst with this perspective: {persona_perspective}.\n\n"
    "Use your analytical lens to evaluate the question. Be true to your perspective — "
    "a hawkish analyst should weigh conflict signals more heavily, a data-driven analyst "
    "should focus on measurable indicators, etc.\n\n"
    "Be calibrated: 0.50 means genuinely uncertain. Return JSON only."
)

PERSONA_REASONING_PROMPT = (
    "Today's date: {as_of}\n\n"
    "Question: {question}\n\n"
    "{knowledge_section}"
    "As {persona_name} ({persona_perspective}), what is the probability of YES?\n\n"
    'Return JSON: {{"probability": <float 0.05-0.95>, "reasoning": "..."}}'
)

DEFAULT_PERSONAS = [
    {"name": "Evidence-Based Analyst", "perspective": "Weighs hard evidence and data over narratives"},
    {"name": "Contrarian Analyst", "perspective": "Challenges the consensus view, looks for overlooked signals"},
    {"name": "Historical Pattern Analyst", "perspective": "Compares to historical precedents and base rates"},
]


# ── Persona Generation ────────────────────────────────────────────────


async def generate_personas(
    llm,
    question: str,
    *,
    num_personas: int = 3,
) -> list[dict]:
    """Generate diverse analyst personas for a question."""
    try:
        response = await llm.complete(
            config_key="knowledge_summary",
            system_prompt=PERSONA_GENERATION_SYSTEM,
            user_prompt=PERSONA_GENERATION_PROMPT.format(
                question=question, num_personas=num_personas
            ),
            json_response=True,
        )
        data = json.loads(response)
        personas = data.get("personas", [])
        if personas and all("name" in p and "perspective" in p for p in personas):
            return personas[:num_personas]
    except Exception as exc:
        logger.warning("Persona generation failed: %s", exc)

    return DEFAULT_PERSONAS[:num_personas]


# ── Per-Persona Reasoning ─────────────────────────────────────────────


async def reason_as_persona(
    llm,
    persona: dict,
    question: str,
    *,
    knowledge_context: str = "",
    as_of: date | None = None,
) -> dict:
    """Have a persona reason about the question and return a probability."""
    as_of_str = (as_of or date.today()).isoformat()
    persona_name = persona.get("name", "Analyst")
    persona_perspective = persona.get("perspective", "General analysis")

    knowledge_section = ""
    if knowledge_context:
        knowledge_section = (
            "Intelligence context (treat as ground truth when it conflicts with your priors):\n"
            f"{knowledge_context}\n\n"
        )

    system_prompt = PERSONA_REASONING_SYSTEM.format(
        persona_name=persona_name,
        persona_perspective=persona_perspective,
    )
    user_prompt = PERSONA_REASONING_PROMPT.format(
        as_of=as_of_str,
        question=question,
        knowledge_section=knowledge_section,
        persona_name=persona_name,
        persona_perspective=persona_perspective,
    )

    try:
        response = await llm.complete(
            config_key="knowledge_summary",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            json_response=True,
        )
        data = json.loads(response)
        return {
            "persona": persona_name,
            "probability": float(data.get("probability", 0.50)),
            "reasoning": data.get("reasoning", ""),
        }
    except Exception as exc:
        logger.warning("Persona %s reasoning failed: %s", persona_name, exc)
        return {
            "persona": persona_name,
            "probability": 0.50,
            "reasoning": f"Fallback: {exc}",
        }


# ── Main Engine ───────────────────────────────────────────────────────


class PerspectiveBenchmarkEngine:
    """Benchmark engine: multi-perspective reasoning + mathematical aggregation."""

    engine_name = "perspective"

    async def predict_probability(
        self,
        question: str,
        *,
        llm=None,
        store=None,
        market_prob: float | None = None,
        as_of: date | None = None,
    ) -> float:
        if llm is None:
            return 0.50

        as_of = as_of or date.today()

        # Step 1: Generate personas (1 LLM call)
        personas = await generate_personas(llm, question, num_personas=3)

        # Step 2: Build knowledge context if store available
        knowledge_context = ""
        if store is not None:
            try:
                from nexus.engine.projection.graphrag_engine import (
                    extract_entities_from_question,
                    gather_graph_evidence,
                    rank_evidence,
                    _format_event_section,
                    _format_relationship_section,
                )
                entities = await extract_entities_from_question(store, llm, question)
                entity_ids = [e["entity_id"] for e in entities if e.get("entity_id")]
                if entity_ids:
                    evidence = await gather_graph_evidence(store, entity_ids, as_of=as_of)
                    evidence = rank_evidence(evidence, as_of=as_of, max_events=8)
                    knowledge_context = (
                        f"Events:\n{_format_event_section(evidence['events'])}\n\n"
                        f"Relationships:\n{_format_relationship_section(evidence['relationships'])}"
                    )
            except Exception as exc:
                logger.debug("Knowledge context assembly failed: %s", exc)

        # Step 3: Each persona reasons (3-5 LLM calls)
        probabilities: list[float] = []
        for persona in personas:
            result = await reason_as_persona(
                llm, persona, question,
                knowledge_context=knowledge_context,
                as_of=as_of,
            )
            probabilities.append(result["probability"])

        if not probabilities:
            return 0.50

        # Step 4: Aggregate via geometric mean of odds (0 LLM calls)
        raw_prob = geometric_mean_of_odds(probabilities)

        # Step 5: Calibrate
        calibrated = extremize(raw_prob, gamma=0.8)
        if market_prob is not None:
            calibrated = anchor_blend(calibrated, market_prob, swarm_weight=0.45)
        return _clip_probability(calibrated)


class PerspectiveForecastEngine:
    """Adapter for the full ForecastEngine protocol."""

    engine_name = "perspective"

    async def generate(
        self,
        llm,
        payload: ForecastEngineInput,
        *,
        critic_pass: bool = True,
        max_questions: int = 4,
        calibration_data: list[dict] | None = None,
    ) -> ForecastRun:
        benchmark = PerspectiveBenchmarkEngine()
        store = getattr(llm, "_store", None) if llm else None
        questions: list[ForecastQuestion] = []

        for thread in payload.threads[:max_questions]:
            question_text = (
                f"Will the narrative '{thread.headline}' see significant "
                f"developments in the next 14 days?"
            )
            prob = await benchmark.predict_probability(
                question_text,
                llm=llm,
                store=store,
                as_of=payload.run_date,
            )
            questions.append(
                ForecastQuestion(
                    question=question_text,
                    forecast_type="binary",
                    target_variable="thread_development",
                    probability=prob,
                    base_rate=0.50,
                    resolution_criteria=f"Significant events in thread '{thread.headline}'",
                    resolution_date=payload.run_date + timedelta(days=14),
                    horizon_days=14,
                    signpost=f"Watch for events in: {thread.headline}",
                    signals_cited=["engine:perspective"],
                )
            )

        return ForecastRun(
            topic_slug=payload.topic_slug,
            topic_name=payload.topic_name,
            engine="perspective",
            generated_for=payload.run_date,
            summary=f"Multi-perspective engine: {len(questions)} questions.",
            questions=questions,
            metadata={"engine": "perspective"},
        )
