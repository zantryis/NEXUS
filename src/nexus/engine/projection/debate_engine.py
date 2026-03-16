"""Debate engine — tests whether agent interaction improves predictions.

Extends the perspective engine with a debate round where personas see
each other's reasoning and revise their estimates. This tests the core
MiroFish/OASIS hypothesis: does agent-to-agent interaction produce
better calibration than independent reasoning + mathematical aggregation?

Pipeline (11 LLM calls for 5 personas):
1. Generate 5 personas (1 call) — reuse generate_personas()
2. Each persona reasons independently (5 calls) — reuse reason_as_persona()
3. Debate: share all estimates + reasoning, each persona revises (5 calls)
4. Aggregate revised estimates via geometric_mean_of_odds()
5. Calibrate with extremize(gamma=0.8)
"""

from __future__ import annotations

import json
import logging
from datetime import date

from nexus.engine.projection.forecasting import _clip_probability
from nexus.engine.projection.perspective_engine import (
    generate_personas,
    reason_as_persona,
)
from nexus.engine.projection.swarm import anchor_blend, extremize, geometric_mean_of_odds

logger = logging.getLogger(__name__)


DEBATE_REVISION_SYSTEM = (
    "You are {persona_name}, an analyst with this perspective: {persona_perspective}.\n\n"
    "You initially assessed this question and gave your probability estimate. "
    "Now you've seen what other analysts think. Consider their reasoning carefully — "
    "update your estimate if their arguments are compelling, but don't change just to "
    "conform. Stay true to your analytical lens while being open to new information.\n\n"
    "Be calibrated. Return JSON only."
)

DEBATE_REVISION_PROMPT = (
    "Today's date: {as_of}\n\n"
    "Question: {question}\n\n"
    "Your initial estimate: {own_probability:.2f}\n"
    "Your reasoning: {own_reasoning}\n\n"
    "Other analysts' views:\n{other_views}\n\n"
    "After considering all perspectives, what is your revised probability of YES?\n"
    "Explain what (if anything) changed your mind.\n\n"
    'Return JSON: {{"probability": <float 0.05-0.95>, "reasoning": "...", '
    '"changed_because": "..." or null}}'
)


async def debate_revision(
    llm,
    persona: dict,
    question: str,
    own_result: dict,
    all_results: list[dict],
    *,
    as_of: date | None = None,
) -> dict:
    """Have a persona revise their estimate after seeing others' reasoning."""
    as_of_str = (as_of or date.today()).isoformat()
    persona_name = persona.get("name", "Analyst")
    persona_perspective = persona.get("perspective", "General analysis")

    # Format other analysts' views (exclude self)
    other_views = []
    for r in all_results:
        if r["persona"] != persona_name:
            other_views.append(
                f"- {r['persona']} (p={r['probability']:.2f}): {r['reasoning']}"
            )
    other_views_str = "\n".join(other_views) if other_views else "No other views available."

    system_prompt = DEBATE_REVISION_SYSTEM.format(
        persona_name=persona_name,
        persona_perspective=persona_perspective,
    )
    user_prompt = DEBATE_REVISION_PROMPT.format(
        as_of=as_of_str,
        question=question,
        own_probability=own_result["probability"],
        own_reasoning=own_result.get("reasoning", ""),
        other_views=other_views_str,
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
            "initial_probability": own_result["probability"],
            "revised_probability": float(data.get("probability", own_result["probability"])),
            "reasoning": data.get("reasoning", ""),
            "changed_because": data.get("changed_because"),
        }
    except Exception as exc:
        logger.warning("Debate revision for %s failed: %s", persona_name, exc)
        return {
            "persona": persona_name,
            "initial_probability": own_result["probability"],
            "revised_probability": own_result["probability"],
            "reasoning": f"Fallback: {exc}",
            "changed_because": None,
        }


class DebateBenchmarkEngine:
    """Benchmark engine: multi-persona reasoning with debate revision round.

    Compared to PerspectiveBenchmarkEngine (independent reasoning only),
    this adds a debate round where personas see each other's reasoning and
    revise. Tests whether interaction improves calibration.
    """

    engine_name = "debate"

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
        personas = await generate_personas(llm, question, num_personas=5)

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

        # Step 3: Each persona reasons independently (5 LLM calls)
        initial_results: list[dict] = []
        for persona in personas:
            result = await reason_as_persona(
                llm, persona, question,
                knowledge_context=knowledge_context,
                as_of=as_of,
            )
            initial_results.append(result)

        if not initial_results:
            return 0.50

        # Step 4: Debate round — each persona revises (5 LLM calls)
        revised_results: list[dict] = []
        for persona, own_result in zip(personas, initial_results):
            revised = await debate_revision(
                llm, persona, question, own_result, initial_results,
                as_of=as_of,
            )
            revised_results.append(revised)

        # Step 5: Aggregate revised estimates
        revised_probs = [r["revised_probability"] for r in revised_results]
        if not revised_probs:
            revised_probs = [r["probability"] for r in initial_results]

        raw_prob = geometric_mean_of_odds(revised_probs)

        # Step 6: Calibrate
        calibrated = extremize(raw_prob, gamma=0.8)
        if market_prob is not None:
            calibrated = anchor_blend(calibrated, market_prob, swarm_weight=0.45)
        return _clip_probability(calibrated)
