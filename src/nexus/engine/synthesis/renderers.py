"""Artifact renderers — produce output artifacts from TopicSynthesis objects.

Renderers receive structured knowledge (TopicSynthesis) and produce
human-readable artifacts. The synthesis is the core product; renderers
are just presentation layers.
"""

from nexus.config.models import NexusConfig
from nexus.engine.synthesis.knowledge import TopicSynthesis
from nexus.llm.client import LLMClient


EDITORIAL_STANCE = (
    "\n\nEDITORIAL STANCE:\n"
    "You are an opinionated analyst. Take clear positions grounded in:\n"
    "- International law (Geneva Conventions, UDHR, ICJ precedent)\n"
    "- Public health consensus and evidence-based science\n"
    "- Anti-exploitation: labor rights, consumer protection, anti-trafficking\n"
    "- Environmental stewardship per scientific consensus\n"
    "- Common sense: pro-education, anti-corruption, institutional accountability\n\n"
    "You are NOT politically partisan (no left/right bias). You call out violations "
    "of international norms regardless of who commits them. When reporting divergent "
    "framings, state which framing aligns better with established law or scientific "
    "consensus. Be direct — don't hedge when the evidence is clear."
)

BRIEFING_SYSTEM_PROMPT = (
    "You are an expert news analyst generating a concise daily briefing from structured "
    "knowledge synthesis data. Output language: {output_language}. "
    "Style: {style}. Depth: {depth}.\n\n"
    "LENGTH: STRICTLY under 800 words total. This is a highlight briefing — "
    "cover the key points, not every detail. Be punchy and scannable.\n\n"
    "STRUCTURE:\n"
    "- Start with a 2-3 sentence executive summary of today's top stories\n"
    "- ## header per topic, ordered by significance\n"
    "- 1-2 short paragraphs per topic covering only the most important developments\n"
    "- End with a one-line source tally (e.g., '16 sources, 3 languages')\n\n"
    "REQUIREMENTS:\n"
    "- Attribute key claims to sources but don't over-attribute — one or two per paragraph\n"
    "- For diverging framings, note the disagreement in one sentence\n"
    "- For non-{output_language} sources, briefly note origin "
    "(e.g., 'Shargh Daily (Tehran)')\n"
    "- Output clean markdown with ## headers per topic\n"
    "- Prioritize what is NEW today over background context\n"
    "- Do NOT use ### sub-headers — keep it flat and scannable"
)


def _build_synthesis_context(syntheses: list[TopicSynthesis]) -> str:
    """Build prompt context from TopicSynthesis objects."""
    sections = []

    for syn in syntheses:
        parts = [f"### Topic: {syn.topic_name}"]

        if syn.background:
            parts.append("\n**Background context:**")
            for s in syn.background[-3:]:
                parts.append(f"- {s.period_start} to {s.period_end}: {s.text}")

        if syn.threads:
            parts.append("\n**Narrative threads:**")
            for i, thread in enumerate(syn.threads):
                parts.append(f"\n#### Thread {i+1}: {thread.headline} (significance: {thread.significance})")

                if thread.events:
                    parts.append("Events:")
                    for e in thread.events:
                        sources_str = ", ".join(
                            f"{s.get('outlet', '?')} ({s.get('affiliation', '?')}/{s.get('country', '?')})"
                            for s in e.sources
                        )
                        parts.append(f"  - [{e.date}] {e.summary} — {sources_str}")

                if thread.convergence:
                    parts.append("Convergence (confirmed by multiple sources):")
                    for c in thread.convergence:
                        if isinstance(c, dict):
                            sources = ", ".join(c.get("confirmed_by", []))
                            parts.append(f"  ✓ {c.get('fact', '?')} (confirmed by: {sources})")
                        else:
                            parts.append(f"  ✓ {c}")

                if thread.divergence:
                    parts.append("Divergence (different framings):")
                    for d in thread.divergence:
                        shared = d.get("shared_event", d.get("claim", "?"))
                        parts.append(
                            f"  ⟷ {shared}: "
                            f"{d.get('source_a', '?')} says \"{d.get('framing_a', '?')}\" vs "
                            f"{d.get('source_b', '?')} says \"{d.get('framing_b', '?')}\""
                        )

                if thread.key_entities:
                    parts.append(f"Key entities: {', '.join(thread.key_entities)}")

        if syn.source_balance:
            parts.append(f"\nSource balance: {syn.source_balance}")
        if syn.languages_represented:
            parts.append(f"Languages: {', '.join(syn.languages_represented)}")

        sections.append("\n".join(parts))

    return "\n\n---\n\n".join(sections)


async def render_text_briefing(
    llm: LLMClient,
    config: NexusConfig,
    syntheses: list[TopicSynthesis],
) -> str:
    """Render a markdown text briefing from TopicSynthesis objects."""
    context = _build_synthesis_context(syntheses)
    system_prompt = BRIEFING_SYSTEM_PROMPT.format(
        output_language=config.user.output_language,
        style=config.briefing.style,
        depth=config.briefing.depth,
    )
    if config.briefing.style == "editorial":
        system_prompt += EDITORIAL_STANCE

    return await llm.complete(
        config_key="synthesis",
        system_prompt=system_prompt,
        user_prompt=context,
    )
