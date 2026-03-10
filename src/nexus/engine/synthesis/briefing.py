"""Briefing synthesis — generate daily markdown briefing from knowledge + articles."""

from dataclasses import dataclass

from nexus.config.models import NexusConfig, TopicConfig
from nexus.engine.knowledge.compression import Summary
from nexus.engine.knowledge.events import Event
from nexus.engine.sources.polling import ContentItem
from nexus.llm.client import LLMClient


@dataclass
class TopicContext:
    topic: TopicConfig
    monthly_summaries: list[Summary]
    weekly_summaries: list[Summary]
    recent_events: list[Event]
    top_articles: list[ContentItem]


def build_context(topic_contexts: list[TopicContext]) -> str:
    """Assemble the context window content for the synthesis prompt."""
    sections = []
    for tc in topic_contexts:
        parts = [f"### Topic: {tc.topic.name} (priority: {tc.topic.priority})"]
        parts.append(f"Subtopics: {', '.join(tc.topic.subtopics)}")

        if tc.monthly_summaries:
            parts.append("\n**Monthly background:**")
            for s in tc.monthly_summaries:
                parts.append(f"- {s.period_start} to {s.period_end}: {s.text}")

        if tc.weekly_summaries:
            parts.append("\n**Recent weekly context:**")
            for s in tc.weekly_summaries:
                parts.append(f"- {s.period_start} to {s.period_end}: {s.text}")

        if tc.recent_events:
            parts.append("\n**Today's events:**")
            for e in tc.recent_events:
                sources_str = ", ".join(
                    f"{s.get('outlet', 'Unknown')} ({s.get('language', '?')})"
                    for s in e.sources
                )
                parts.append(f"- [{e.date}] (sig:{e.significance}) {e.summary} — {sources_str}")

        if tc.top_articles:
            parts.append("\n**Full article texts (highest significance):**")
            for a in tc.top_articles[:3]:
                parts.append(f"\n--- {a.title} ({a.source_id}) ---\n{(a.full_text or '')[:2000]}")

        sections.append("\n".join(parts))

    return "\n\n---\n\n".join(sections)


SYSTEM_PROMPT_TEMPLATE = (
    "You are an expert news analyst generating a daily briefing. "
    "Output language: {output_language}. "
    "Style: {style}. Depth: {depth}.\n\n"
    "REQUIREMENTS:\n"
    "- Every factual claim MUST be attributed to a named source\n"
    "- Organize by topic, ordered by priority\n"
    "- Reference historical context from the knowledge layer for narrative continuity\n"
    "- For non-{output_language} sources, provide contextual attribution "
    "(e.g., 'Shargh Daily, a Tehran-based reformist newspaper')\n"
    "- Output clean markdown with ## headers per topic\n"
    "- Start with a brief executive summary of today's key developments"
)


async def generate_briefing(
    llm: LLMClient,
    config: NexusConfig,
    topic_contexts: list[TopicContext],
) -> str:
    """Generate the daily briefing markdown via LLM synthesis."""
    context = build_context(topic_contexts)
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        output_language=config.user.output_language,
        style=config.briefing.style,
        depth=config.briefing.depth,
    )

    return await llm.complete(
        config_key="synthesis",
        system_prompt=system_prompt,
        user_prompt=context,
    )
