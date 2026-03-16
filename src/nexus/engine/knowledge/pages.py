"""Cached narrative page generation and staleness management.

Generates LLM-written narrative pages (backstory, entity profiles,
thread deep-dives, weekly recaps) and caches them in the knowledge store.
"""

import hashlib
import json
import logging

from nexus.engine.knowledge.compression import Summary
from nexus.engine.knowledge.store import KnowledgeStore
from nexus.llm.client import LLMClient

logger = logging.getLogger(__name__)

# Page type configurations with TTLs
PAGE_CONFIGS = {
    "backstory": {"ttl_days": 7},
    "entity_profile": {"ttl_days": 3},
    "thread_deepdive": {"ttl_days": 1},
    "weekly_recap": {"ttl_days": 365},  # Effectively immutable
    "projection": {"ttl_days": 1},
}


def compute_prompt_hash(input_data: dict) -> str:
    """Compute a deterministic hash of input data for staleness detection."""
    return hashlib.sha256(
        json.dumps(input_data, sort_keys=True).encode()
    ).hexdigest()


BACKSTORY_SYSTEM_PROMPT = (
    "You are a background briefing writer. Given weekly summaries about a topic, "
    "write a concise background page that helps a reader understand the current state "
    "of affairs. Include key context, recent developments, and important actors.\n\n"
    "Write in markdown format. Be factual, clear, and concise. "
    "Target 500-1000 words."
)

ENTITY_PROFILE_SYSTEM_PROMPT = (
    "You are an entity profile writer. Given information about an entity and its "
    "recent appearances in events, write a concise profile that explains who/what "
    "this entity is, their role in recent events, and their significance.\n\n"
    "Write in markdown format. Be factual and concise. Target 300-600 words."
)

THREAD_DEEPDIVE_SYSTEM_PROMPT = (
    "You are a narrative analyst. Given a narrative thread with its events, "
    "convergence (facts confirmed by multiple sources), and divergence "
    "(conflicting framings), write a deep-dive analysis.\n\n"
    "Write in markdown format. Include source analysis where relevant. "
    "Target 500-1000 words."
)


async def generate_backstory(
    llm: LLMClient,
    topic_name: str,
    topic_slug: str,
    summaries: list[Summary],
) -> dict:
    """Generate a backstory page for a topic."""
    summary_lines = []
    for s in summaries[-10:]:  # Last 10 summaries
        summary_lines.append(
            f"- {s.period_start} to {s.period_end}: {s.text}"
        )

    user_prompt = (
        f"Topic: {topic_name}\n\n"
        f"Recent summaries:\n{chr(10).join(summary_lines) or 'No prior summaries yet.'}\n\n"
        f"Write a background briefing page for this topic."
    )

    input_data = {"topic": topic_name, "summaries": [s.text for s in summaries[-10:]]}
    prompt_hash = compute_prompt_hash(input_data)

    content = await llm.complete(
        config_key="knowledge_summary",
        system_prompt=BACKSTORY_SYSTEM_PROMPT,
        user_prompt=user_prompt,
    )

    return {
        "slug": f"backstory:{topic_slug}",
        "title": f"Background: {topic_name}",
        "page_type": "backstory",
        "content_md": content,
        "topic_slug": topic_slug,
        "ttl_days": PAGE_CONFIGS["backstory"]["ttl_days"],
        "prompt_hash": prompt_hash,
    }


async def generate_entity_profile(
    llm: LLMClient,
    entity: dict,
    events_data: list[dict],
) -> dict:
    """Generate a profile page for an entity."""
    aliases_str = ", ".join(entity.get("aliases", []))
    events_str = "\n".join(
        f"- [{e.get('date', '?')}] {e.get('summary', '')}"
        for e in events_data[:20]
    )

    user_prompt = (
        f"Entity: {entity['canonical_name']}\n"
        f"Type: {entity.get('entity_type', 'unknown')}\n"
        f"Aliases: {aliases_str or 'None'}\n\n"
        f"Recent events involving this entity:\n{events_str or 'No events yet.'}\n\n"
        f"Write a profile for this entity."
    )

    input_data = {
        "entity": entity["canonical_name"],
        "type": entity.get("entity_type"),
        "events": [e.get("summary", "") for e in events_data[:20]],
    }
    prompt_hash = compute_prompt_hash(input_data)

    content = await llm.complete(
        config_key="knowledge_summary",
        system_prompt=ENTITY_PROFILE_SYSTEM_PROMPT,
        user_prompt=user_prompt,
    )

    return {
        "slug": f"entity:{entity['id']}",
        "title": f"Entity Profile: {entity['canonical_name']}",
        "page_type": "entity_profile",
        "content_md": content,
        "topic_slug": None,
        "ttl_days": PAGE_CONFIGS["entity_profile"]["ttl_days"],
        "prompt_hash": prompt_hash,
    }


async def generate_thread_deepdive(
    llm: LLMClient,
    thread: dict,
    events_data: list[dict],
    convergence: list[dict],
    divergence: list[dict],
) -> dict:
    """Generate a deep-dive page for a narrative thread."""
    events_str = "\n".join(
        f"- [{e.get('date', '?')}] {e.get('summary', '')}"
        for e in events_data[:30]
    )
    conv_str = "\n".join(
        f"- {c.get('fact', str(c))} (confirmed by: {', '.join(c.get('confirmed_by', []))})"
        for c in convergence
    ) if convergence else "None"
    div_str = "\n".join(
        f"- {d.get('shared_event', '')}: {d.get('source_a', '')} says \"{d.get('framing_a', '')}\", "
        f"{d.get('source_b', '')} says \"{d.get('framing_b', '')}\""
        for d in divergence
    ) if divergence else "None"

    user_prompt = (
        f"Thread: {thread['headline']}\n"
        f"Status: {thread.get('status', 'unknown')}\n"
        f"Significance: {thread.get('significance', 5)}/10\n\n"
        f"Events:\n{events_str}\n\n"
        f"Convergence (facts confirmed by multiple sources):\n{conv_str}\n\n"
        f"Divergence (conflicting framings):\n{div_str}\n\n"
        f"Write a deep-dive analysis of this narrative thread."
    )

    input_data = {
        "thread": thread["headline"],
        "events": [e.get("summary", "") for e in events_data[:30]],
        "convergence_count": len(convergence),
        "divergence_count": len(divergence),
    }
    prompt_hash = compute_prompt_hash(input_data)

    content = await llm.complete(
        config_key="knowledge_summary",
        system_prompt=THREAD_DEEPDIVE_SYSTEM_PROMPT,
        user_prompt=user_prompt,
    )

    return {
        "slug": f"thread:{thread['slug']}",
        "title": f"Deep Dive: {thread['headline']}",
        "page_type": "thread_deepdive",
        "content_md": content,
        "topic_slug": None,
        "ttl_days": PAGE_CONFIGS["thread_deepdive"]["ttl_days"],
        "prompt_hash": prompt_hash,
    }


async def refresh_stale_pages(
    store: KnowledgeStore,
    llm: LLMClient,
    topic_slugs: list[str],
    topic_names: dict[str, str],
) -> int:
    """Find and regenerate stale pages. Returns count of pages refreshed."""
    refreshed = 0

    # Check backstory pages for each topic
    for slug in topic_slugs:
        backstory_slug = f"backstory:{slug}"
        page = await store.get_page(backstory_slug)

        if page is not None:
            # Check staleness via get_stale_pages
            stale = await store.get_stale_pages()
            stale_slugs = {p["slug"] for p in stale}
            if backstory_slug not in stale_slugs:
                continue  # Still fresh

        # Generate new backstory
        topic_name = topic_names.get(slug, slug)
        summaries = await store.get_summaries(slug, "weekly")
        try:
            result = await generate_backstory(llm, topic_name, slug, summaries)
            await store.save_page(
                result["slug"], result["title"], result["page_type"],
                result["content_md"], result["topic_slug"],
                result["ttl_days"], result["prompt_hash"],
            )
            refreshed += 1
            logger.info(f"Refreshed page: {backstory_slug}")
        except Exception as e:
            logger.warning(f"Failed to generate {backstory_slug}: {e}")

    return refreshed
