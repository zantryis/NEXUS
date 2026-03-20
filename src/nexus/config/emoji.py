"""Assign Unicode emojis to topics via LLM."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nexus.config.models import TopicConfig
    from nexus.llm.client import LLMClient

logger = logging.getLogger(__name__)

_SYSTEM = "You are a helpful assistant that picks a single Unicode emoji to represent a topic."

_USER_TEMPLATE = (
    "For each topic below, pick ONE Unicode emoji that best represents it.\n"
    "Return ONLY a JSON object mapping topic name to emoji, e.g.: "
    '{{"Iran-US Relations": "🌍", "AI/ML Research": "🤖"}}\n\n'
    "Topics:\n{topics}"
)


async def assign_topic_emojis(
    llm: LLMClient,
    topics: list[TopicConfig],
) -> dict[str, str]:
    """Assign emojis to topics that don't have one. Returns {name: emoji}."""
    needs_emoji = [t for t in topics if not t.emoji]
    if not needs_emoji:
        return {}

    topic_list = "\n".join(f"- {t.name}" for t in needs_emoji)
    prompt = _USER_TEMPLATE.format(topics=topic_list)

    try:
        raw = await llm.complete(
            "knowledge_summary", _SYSTEM, prompt, json_response=True,
        )
        result = json.loads(raw)
        if not isinstance(result, dict):
            return {}
        # Validate: each value should be a short string (emoji)
        return {k: v for k, v in result.items() if isinstance(v, str) and len(v) <= 4}
    except Exception:
        logger.debug("Emoji assignment failed, skipping", exc_info=True)
        return {}
