"""LLM-as-judge — evaluate TopicSynthesis quality."""

import json
import logging


from nexus.engine.synthesis.knowledge import TopicSynthesis
from nexus.llm.client import LLMClient

logger = logging.getLogger(__name__)

JUDGE_SYSTEM_PROMPT = (
    "You are an expert evaluator of news intelligence synthesis. "
    "Given a structured knowledge synthesis object, evaluate its quality on these dimensions:\n\n"
    "1. **Completeness** (1-10): Are major developments covered? Are there obvious gaps?\n"
    "2. **Source balance** (1-10): Are affiliations and perspectives proportionally represented?\n"
    "3. **Convergence accuracy** (1-10): Are facts marked as 'confirmed' actually supported by "
    "multiple independent sources?\n"
    "4. **Divergence detection** (1-10): Are genuine framing differences surfaced, "
    "not just paraphrasing?\n"
    "5. **Entity coverage** (1-10): Are key actors and organizations tracked?\n\n"
    "Respond with JSON:\n"
    "{\n"
    '  "completeness": <int>,\n'
    '  "source_balance": <int>,\n'
    '  "convergence_accuracy": <int>,\n'
    '  "divergence_detection": <int>,\n'
    '  "entity_coverage": <int>,\n'
    '  "overall": <float>,\n'
    '  "strengths": ["..."],\n'
    '  "weaknesses": ["..."],\n'
    '  "suggestions": ["..."]\n'
    "}"
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
) -> dict:
    """Evaluate a single TopicSynthesis via LLM judge. Returns scores dict."""
    formatted = _format_synthesis_for_judge(synthesis)

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
