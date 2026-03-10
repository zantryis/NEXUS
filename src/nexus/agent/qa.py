"""Q&A agent — answer user questions using the knowledge store."""

import logging

from nexus.config.models import NexusConfig
from nexus.engine.knowledge.store import KnowledgeStore
from nexus.llm.client import LLMClient

logger = logging.getLogger(__name__)

QA_SYSTEM_PROMPT = (
    "You are Nexus, a news intelligence assistant. Answer the user's question "
    "based on the knowledge context provided. Be concise and factual. "
    "Always attribute claims to their sources. "
    "If the context doesn't contain enough information, say so honestly.\n\n"
    "Output language: {output_language}\n"
)


async def answer_question(
    llm: LLMClient,
    store: KnowledgeStore,
    config: NexusConfig,
    question: str,
) -> str:
    """Answer a user question using recent events and threads from the store."""
    # Gather context from store
    context_parts = []

    # Recent events across all topics
    topic_stats = await store.get_topic_stats()
    for ts in topic_stats:
        slug = ts["topic_slug"]
        events = await store.get_recent_events(slug, days=3, limit=10)
        if events:
            context_parts.append(f"## {slug} (recent events)")
            for e in events:
                sources_str = ", ".join(s.get("outlet", "?") for s in e.sources)
                context_parts.append(f"- [{e.date}] {e.summary} (sources: {sources_str})")

    # Active threads
    threads = await store.get_active_threads()
    if threads:
        context_parts.append("\n## Active narrative threads")
        for t in threads:
            context_parts.append(f"- {t['headline']} (significance: {t['significance']})")

    context = "\n".join(context_parts) if context_parts else "No recent data available."

    system_prompt = QA_SYSTEM_PROMPT.format(
        output_language=config.user.output_language,
    )

    user_prompt = f"Knowledge context:\n{context}\n\nUser question: {question}"

    return await llm.complete(
        config_key="agent",
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )
