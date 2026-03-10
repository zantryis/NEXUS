"""Relevance filtering — score content items against topic definitions via LLM."""

import json
import logging
from nexus.engine.sources.polling import ContentItem
from nexus.config.models import TopicConfig
from nexus.llm.client import LLMClient

logger = logging.getLogger(__name__)

BATCH_SYSTEM_PROMPT = (
    "You are a relevance scorer. Given a list of articles and a topic definition, "
    "score how relevant each article is to the topic on a scale of 1-10. "
    "Respond with a JSON array of objects: [{\"id\": <int>, \"score\": <int>, \"reason\": \"<brief>\"}] "
    "for each article. Use the article's id number as provided."
)

SINGLE_SYSTEM_PROMPT = (
    "You are a relevance scorer. Given an article and a topic definition, "
    "score how relevant the article is to the topic on a scale of 1-10. "
    'Respond with JSON: {"score": <int>, "reason": "<brief explanation>"}'
)

BATCH_SIZE = 10


async def score_relevance(
    llm: LLMClient, item: ContentItem, topic: TopicConfig
) -> tuple[int, str]:
    """Score a single item's relevance to a topic. Returns (score, reason)."""
    user_prompt = (
        f"Topic: {topic.name}\n"
        f"Subtopics: {', '.join(topic.subtopics)}\n\n"
        f"Article title: {item.title}\n"
        f"Article text: {(item.full_text or item.snippet)[:2000]}"
    )
    try:
        response = await llm.complete(
            config_key="filtering",
            system_prompt=SINGLE_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            json_response=True,
        )
        data = json.loads(response)
        return int(data["score"]), data["reason"]
    except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
        logger.warning(f"Failed to parse relevance score for {item.url}: {e}")
        return 0, f"Failed to parse response: {e}"


async def score_batch(
    llm: LLMClient, items: list[ContentItem], topic: TopicConfig
) -> list[tuple[int, str]]:
    """Score a batch of items in a single LLM call. Returns list of (score, reason)."""
    articles_text = []
    for i, item in enumerate(items):
        text = (item.full_text or item.snippet or "")[:500]
        articles_text.append(f"[Article {i}] Title: {item.title}\nText: {text}\n")

    user_prompt = (
        f"Topic: {topic.name}\n"
        f"Subtopics: {', '.join(topic.subtopics)}\n\n"
        f"Score each article:\n\n" + "\n".join(articles_text)
    )
    try:
        response = await llm.complete(
            config_key="filtering",
            system_prompt=BATCH_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            json_response=True,
        )
        data = json.loads(response)
        if not isinstance(data, list):
            data = [data]

        # Build a map of id -> (score, reason)
        score_map = {}
        for entry in data:
            idx = int(entry["id"])
            score_map[idx] = (int(entry["score"]), entry.get("reason", ""))

        # Return in order, defaulting to 0 for missing
        results = []
        for i in range(len(items)):
            results.append(score_map.get(i, (0, "Missing from batch response")))
        return results

    except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
        logger.warning(f"Batch scoring failed, falling back to individual: {e}")
        # Fallback to individual scoring
        results = []
        for item in items:
            results.append(await score_relevance(llm, item, topic))
        return results


async def filter_items(
    llm: LLMClient,
    items: list[ContentItem],
    topic: TopicConfig,
    threshold: int = 5,
) -> list[ContentItem]:
    """Filter items by relevance score against a topic. Uses batch scoring."""
    results = []

    # Process in batches
    for batch_start in range(0, len(items), BATCH_SIZE):
        batch = items[batch_start:batch_start + BATCH_SIZE]
        scores = await score_batch(llm, batch, topic)

        for item, (score, reason) in zip(batch, scores):
            item.relevance_score = score
            if score >= threshold:
                results.append(item)
            else:
                logger.debug(f"Filtered out (score={score}): {item.title}")

    return results
