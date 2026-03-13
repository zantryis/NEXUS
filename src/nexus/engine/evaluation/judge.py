"""LLM-as-judge — evaluate TopicSynthesis quality."""

import json
import logging


from nexus.engine.synthesis.knowledge import TopicSynthesis
from nexus.llm.client import LLMClient

logger = logging.getLogger(__name__)

JUDGE_SYSTEM_PROMPT = (
    "You are an expert evaluator of news intelligence synthesis.\n"
    "Score the synthesis on these dimensions using the EXACT anchor definitions below.\n\n"
    "## 1. Completeness (2-10)\n"
    "- **2**: Covers 1-2 stories, misses most major developments\n"
    "- **4**: Covers some developments but has 2+ significant gaps\n"
    "- **6**: Covers most major developments, misses nuance/secondary stories\n"
    "- **8**: Comprehensive — all major + most secondary developments covered\n"
    "- **10**: Complete — no gaps, secondary stories and context included\n\n"
    "## 2. Source Balance (2-10)\n"
    "- **2**: Single affiliation dominates (>80%), other perspectives absent\n"
    "- **4**: 2 affiliations present but heavily skewed (>60% one type)\n"
    "- **6**: Multiple affiliations present, moderate imbalance\n"
    "- **8**: Good representation across affiliations, minor gaps\n"
    "- **10**: Proportional representation across all present affiliation types\n\n"
    "## 3. Convergence Accuracy (2-10)\n"
    "- **2**: 'Confirmed' facts cite only 1 source, or sources are not independent\n"
    "- **4**: Some convergence claims have 2 sources but from same affiliation/country\n"
    "- **6**: Most convergence claims have 2+ independent sources, some weak\n"
    "- **8**: Convergence claims reliably cite independent sources (different outlets/countries)\n"
    "- **10**: All convergence claims have 3+ truly independent sources with clear attribution\n\n"
    "## 4. Divergence Detection (2-10)\n"
    "- **2**: No divergence noted, or 'divergence' is just paraphrasing\n"
    "- **4**: 1 genuine divergence found, others are weak or manufactured\n"
    "- **6**: Several divergences identified, most are genuine framing differences\n"
    "- **8**: Divergences are genuine, attributed to specific outlets, explained clearly\n"
    "- **10**: Nuanced divergence analysis — explains WHY sources differ\n\n"
    "## 5. Entity Coverage (2-10)\n"
    "- **2**: Key actors missing, entities inconsistently named\n"
    "- **4**: Major actors present but secondary actors/orgs absent\n"
    "- **6**: Most key actors tracked, some inconsistency in naming\n"
    "- **8**: All key actors + major orgs tracked, names consistent\n"
    "- **10**: Comprehensive entity tracking including relationships between actors\n\n"
    "Respond with JSON:\n"
    '{"completeness": <int>, "source_balance": <int>, "convergence_accuracy": <int>,\n'
    ' "divergence_detection": <int>, "entity_coverage": <int>, "overall": <float>,\n'
    ' "strengths": ["..."], "weaknesses": ["..."], "suggestions": ["..."]}\n\n'
    "IMPORTANT: Use the FULL range. A naive article dump should score 2-4. "
    "A good human-curated synthesis scores 6-8. Only a near-perfect synthesis deserves 9-10."
)


def _format_synthesis_for_judge(synthesis: TopicSynthesis) -> str:
    """Format a TopicSynthesis into readable text for the judge."""
    parts = [f"# Topic: {synthesis.topic_name}"]
    parts.append(f"Source balance: {synthesis.source_balance}")
    parts.append(f"Languages: {', '.join(synthesis.languages_represented)}")

    for i, thread in enumerate(synthesis.threads):
        parts.append(f"\n## Thread {i+1}: {thread.headline} (significance: {thread.significance})")

        if thread.events:
            parts.append("Events:")
            for e in thread.events:
                sources = ", ".join(
                    f"{s.get('outlet', '?')} ({s.get('affiliation', '?')}/{s.get('country', '?')})"
                    for s in e.sources
                )
                parts.append(f"  - [{e.date}] {e.summary}")
                parts.append(f"    Sources: {sources}")
                parts.append(f"    Entities: {', '.join(e.entities)}")

        if thread.convergence:
            parts.append("Convergence:")
            for c in thread.convergence:
                if isinstance(c, dict):
                    sources = ", ".join(c.get("confirmed_by", []))
                    parts.append(f"  - {c.get('fact', '?')} (confirmed by: {sources})")
                else:
                    parts.append(f"  - {c}")

        if thread.divergence:
            parts.append("Divergence:")
            for d in thread.divergence:
                shared = d.get("shared_event", d.get("claim", "?"))
                parts.append(
                    f"  - {shared}: "
                    f"{d.get('source_a', '?')} vs {d.get('source_b', '?')}"
                )

        if thread.key_entities:
            parts.append(f"Key entities: {', '.join(thread.key_entities)}")

    return "\n".join(parts)


async def judge_synthesis(
    llm: LLMClient,
    synthesis: TopicSynthesis,
    model_override: str | None = None,
) -> dict:
    """Evaluate a single TopicSynthesis via LLM judge. Returns scores dict.

    If model_override is set, temporarily swaps the agent model for this call.
    """
    formatted = _format_synthesis_for_judge(synthesis)

    original_model = None
    if model_override and hasattr(llm, "_config"):
        original_model = llm._config.agent
        llm._config.agent = model_override
    try:
        response = await llm.complete(
            config_key="agent",
            system_prompt=JUDGE_SYSTEM_PROMPT,
            user_prompt=formatted,
            json_response=True,
        )
        scores = json.loads(response)
        # Compute overall if not provided
        if "overall" not in scores:
            dims = ["completeness", "source_balance", "convergence_accuracy",
                     "divergence_detection", "entity_coverage"]
            vals = [scores.get(d, 5) for d in dims]
            scores["overall"] = round(sum(vals) / len(vals), 1)
        return scores
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        logger.warning(f"Judge failed for {synthesis.topic_name}: {e}")
        return {"error": str(e)}
    finally:
        if original_model is not None:
            llm._config.agent = original_model


async def compare_syntheses(
    llm: LLMClient,
    synthesis_a: TopicSynthesis,
    synthesis_b: TopicSynthesis,
    label_a: str = "A",
    label_b: str = "B",
) -> dict:
    """Compare two TopicSynthesis objects on the same rubric."""
    scores_a = await judge_synthesis(llm, synthesis_a)
    scores_b = await judge_synthesis(llm, synthesis_b)

    return {
        label_a: scores_a,
        label_b: scores_b,
        "winner": label_a if scores_a.get("overall", 0) > scores_b.get("overall", 0) else label_b,
    }
