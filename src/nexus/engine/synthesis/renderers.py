"""Artifact renderers — produce output artifacts from TopicSynthesis objects.

Renderers receive structured knowledge (TopicSynthesis) and produce
human-readable artifacts. The synthesis is the core product; renderers
are just presentation layers.
"""

from nexus.config.models import NexusConfig
from nexus.engine.synthesis.knowledge import TopicSynthesis
from nexus.llm.client import LLMClient


BRIEFING_SYSTEM_PROMPT = (
    "You are an expert news analyst generating a daily briefing from structured "
    "knowledge synthesis data. Output language: {output_language}. "
    "Style: {style}. Depth: {depth}.\n\n"
    "REQUIREMENTS:\n"
    "- Every factual claim MUST be attributed to a named source\n"
    "- Organize by topic, ordered by significance\n"
    "- For each narrative thread, present convergence (what sources agree on) "
    "and divergence (where they disagree) clearly\n"
    "- Reference historical context for narrative continuity\n"
    "- For non-{output_language} sources, provide contextual attribution "
    "(e.g., 'Shargh Daily, a Tehran-based reformist newspaper')\n"
    "- When sources from different affiliations frame events differently, "
    "present BOTH framings without editorial judgment\n"
    "- Output clean markdown with ## headers per topic\n"
    "- Start with a brief executive summary of today's key developments\n"
    "- Note the source balance at the end (e.g., 'This briefing draws from "
    "N sources across M languages')"
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

    return await llm.complete(
        config_key="synthesis",
        system_prompt=system_prompt,
        user_prompt=context,
    )
