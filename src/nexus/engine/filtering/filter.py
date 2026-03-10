"""Relevance filtering — score content items against topic definitions via LLM."""

import json
import logging
from nexus.engine.sources.polling import ContentItem
from nexus.config.models import TopicConfig
from nexus.llm.client import LLMClient

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = "You are a relevance scorer. Given an article and a topic definition, score how relevant the article is to the topic on a scale of 1-10. Respond with JSON: {\"score\": <int>, \"reason\": \"<brief explanation>\"}"


async def score_relevance(
    llm: LLMClient, item: ContentItem, topic: TopicConfig
) -> tuple[int, str]:
    """Score an item's relevance to a topic. Returns (score, reason)."""
    user_prompt = (
        f"Topic: {topic.name}\n"
        f"Subtopics: {', '.join(topic.subtopics)}\n\n"
        f"Article title: {item.title}\n"
        f"Article text: {(item.full_text or item.snippet)[:2000]}"
    )
    try:
        response = await llm.complete(
            config_key="filtering",
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            json_response=True,
        )
        data = json.loads(response)
        return int(data["score"]), data["reason"]
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        logger.warning(f"Failed to parse relevance score for {item.url}: {e}")
        return 0, f"Failed to parse response: {e}"


async def filter_items(
    llm: LLMClient,
    items: list[ContentItem],
    topic: TopicConfig,
    threshold: int = 5,
) -> list[ContentItem]:
    """Filter items by relevance score against a topic."""
    results = []
    for item in items:
        score, reason = await score_relevance(llm, item, topic)
        item.relevance_score = score
        if score >= threshold:
            results.append(item)
        else:
            logger.debug(f"Filtered out (score={score}): {item.title}")
    return results
