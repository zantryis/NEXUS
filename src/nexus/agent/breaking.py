"""Breaking news poller — topic-scoped, single-pass scoring against wire feeds."""

import asyncio
import hashlib
import json
import logging
import re

from nexus.config.models import NexusConfig, TopicConfig
from nexus.engine.knowledge.store import KnowledgeStore
from nexus.engine.sources.polling import ContentItem
from nexus.engine.sources.router import poll_source
from nexus.llm.client import LLMClient

logger = logging.getLogger(__name__)

# Diverse wire feeds: major agencies + social signal
DEFAULT_WIRE_FEEDS = [
    # Major world-news feeds (fast, broad, currently parse reliably here)
    {"type": "rss", "url": "https://rss.nytimes.com/services/xml/rss/nyt/World.xml",
     "id": "wire-nyt", "affiliation": "private", "country": "US", "tier": "A"},
    {"type": "rss", "url": "https://feeds.bbci.co.uk/news/world/rss.xml",
     "id": "wire-bbc", "affiliation": "public", "country": "GB", "tier": "A"},
    {"type": "rss", "url": "https://www.theguardian.com/world/rss",
     "id": "wire-guardian-world", "affiliation": "private", "country": "GB", "tier": "A"},
    {"type": "rss", "url": "https://www.aljazeera.com/xml/rss/all.xml",
     "id": "wire-aljazeera", "affiliation": "private", "country": "QA", "tier": "A"},
    # Social/crowd signal (faster breaking detection)
    {"type": "reddit", "subreddit": "worldnews", "sort": "hot",
     "id": "wire-reddit-worldnews"},
    {"type": "reddit", "subreddit": "geopolitics", "sort": "hot",
     "id": "wire-reddit-geopolitics"},
]

# Breaking-news-specific scoring prompt — stricter than pipeline relevance
BATCH_SYSTEM_PROMPT = (
    "You are a breaking news significance scorer for a personal intelligence briefing. "
    "Given a list of headlines and a topic definition, score how significant each headline "
    "is as a BREAKING development for this specific topic.\n\n"
    "Scoring guide:\n"
    "- 9-10: Major event directly about this topic (war, crisis, breakthrough, major policy change)\n"
    "- 7-8: Significant development clearly related to this specific topic\n"
    "- 4-6: Tangentially related or minor update\n"
    "- 1-3: Not relevant to this topic\n\n"
    "Be STRICT. Most headlines should score 1-5. Only score 7+ for headlines that "
    "someone tracking this specific topic would genuinely want an immediate alert about. "
    "Generic world news that merely mentions a keyword is NOT sufficient for a high score.\n\n"
    'Respond with a JSON array: [{"id": <int>, "score": <int>, "reason": "<brief>"}]'
)

BATCH_SIZE = 15  # Larger batches = fewer LLM calls
MAX_ALERTS_PER_TOPIC = 5  # Cap alerts per topic per cycle


def _hash_headline(headline: str) -> str:
    """Create a dedup hash from a headline."""
    return hashlib.sha256(headline.lower().strip().encode()).hexdigest()[:16]


def _topic_slug(topic: TopicConfig) -> str:
    """Derive slug from topic name."""
    return topic.name.lower().replace(" ", "-").replace("/", "-")


def _topic_keywords(topic: TopicConfig) -> list[str]:
    """Build a lightweight keyword set for heuristic breaking fallback."""
    raw_terms = [topic.name, *topic.subtopics]
    keywords: list[str] = []
    seen: set[str] = set()
    stop = {
        "and", "the", "for", "with", "from", "into", "about", "news", "relations",
        "global", "research", "transition", "technical", "programs", "labs",
    }
    for term in raw_terms:
        parts = re.split(r"[^a-z0-9+]+", term.lower())
        expanded = [term.lower(), *parts]
        for piece in expanded:
            piece = piece.strip()
            if len(piece) < 3 or piece in stop:
                continue
            if piece not in seen:
                seen.add(piece)
                keywords.append(piece)
    return keywords


def _heuristic_score_item(item: ContentItem, topic: TopicConfig) -> tuple[int, str]:
    """Heuristic topic relevance when the hosted LLM proxy is unavailable."""
    text = " ".join(filter(None, [item.title or "", item.snippet or ""]))
    haystack = text.lower()
    keywords = _topic_keywords(topic)
    topic_name = topic.name.lower()

    phrase_hit = topic_name in haystack
    hit_terms = [kw for kw in keywords if kw in haystack]
    hit_count = len(set(hit_terms))

    score = 0
    if phrase_hit:
        score += 4
    score += min(hit_count, 5)
    if any(sub.lower() in haystack for sub in topic.subtopics):
        score += 2

    if score >= 7:
        final = min(9, score)
    elif phrase_hit or hit_count >= 2:
        final = 7
    else:
        final = min(score, 5)

    reason = "heuristic keyword fallback"
    if hit_terms:
        reason += f": {', '.join(sorted(set(hit_terms))[:4])}"
    return final, reason


async def _poll_all_feeds(feed_configs: list[dict]) -> list[ContentItem]:
    """Poll all feeds concurrently and return flattened, URL-deduped items."""
    tasks = [poll_source(cfg) for cfg in feed_configs]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    seen_urls: set[str] = set()
    items: list[ContentItem] = []
    for result in results:
        if isinstance(result, Exception):
            logger.warning(f"Wire feed poll failed: {result}")
            continue
        for item in result:
            if item.url and item.url not in seen_urls:
                seen_urls.add(item.url)
                items.append(item)

    return items


async def _score_batch_for_topic(
    llm: LLMClient,
    items: list[ContentItem],
    topic: TopicConfig,
) -> list[tuple[int, str]]:
    """Score a batch of items for breaking news significance using flash model."""
    articles_text = []
    for i, item in enumerate(items):
        text = (item.snippet or item.title or "")[:300]
        articles_text.append(
            f"[{i}] {item.title}"
            + (f"\n    {text}" if text != item.title else "")
        )

    user_prompt = (
        f"Topic: {topic.name}\n"
        f"Subtopics: {', '.join(topic.subtopics)}\n\n"
        f"Score each headline:\n\n" + "\n".join(articles_text)
    )

    try:
        response = await llm.complete(
            config_key="breaking_news",
            system_prompt=BATCH_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            json_response=True,
        )
        data = json.loads(response)
        if not isinstance(data, list):
            data = [data]

        score_map = {}
        for entry in data:
            idx = int(entry["id"])
            score_map[idx] = (int(entry["score"]), entry.get("reason", ""))

        return [score_map.get(i, (0, "Missing")) for i in range(len(items))]

    except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
        logger.warning(f"Breaking news batch scoring parse failed: {e}; using heuristic fallback")
        return [_heuristic_score_item(item, topic) for item in items]
    except Exception as e:
        logger.warning(f"Breaking news batch scoring failed: {e}; using heuristic fallback")
        return [_heuristic_score_item(item, topic) for item in items]


async def _score_topic(
    llm: LLMClient,
    topic: TopicConfig,
    all_items: list[ContentItem],
    store: KnowledgeStore,
    threshold: int,
) -> list[dict]:
    """Score all items against a single topic. Returns alert dicts."""
    slug = _topic_slug(topic)

    # Batch dedup: get all alerted hashes for this topic in one query
    all_hashes = [_hash_headline(item.title) for item in all_items]
    alerted = await store.get_alerted_hashes(all_hashes, slug)

    candidates = [
        item for item, h in zip(all_items, all_hashes)
        if h not in alerted
    ]

    if not candidates:
        return []

    # Score in batches
    topic_alerts: list[dict] = []
    for batch_start in range(0, len(candidates), BATCH_SIZE):
        batch = candidates[batch_start:batch_start + BATCH_SIZE]
        scores = await _score_batch_for_topic(llm, batch, topic)

        for item, (score, reason) in zip(batch, scores):
            if score >= threshold:
                h = _hash_headline(item.title)
                await store.add_breaking_alert(
                    h, item.title, item.url, score, topic_slug=slug,
                )
                topic_alerts.append({
                    "headline": item.title,
                    "source_url": item.url,
                    "significance_score": score,
                    "headline_hash": h,
                })

    # Cap per topic: keep highest-scoring alerts
    if len(topic_alerts) > MAX_ALERTS_PER_TOPIC:
        topic_alerts.sort(key=lambda a: a["significance_score"], reverse=True)
        topic_alerts = topic_alerts[:MAX_ALERTS_PER_TOPIC]

    if topic_alerts:
        logger.info(
            f"Breaking news [{slug}]: {len(topic_alerts)} alerts "
            f"above threshold {threshold}"
        )

    return topic_alerts


async def check_breaking_news(
    llm: LLMClient,
    config: NexusConfig,
    store: KnowledgeStore,
) -> dict[str, list[dict]]:
    """Poll wire feeds and score headlines against each configured topic.

    Returns {topic_slug: [alert_dicts]} where each alert has:
        headline, source_url, significance_score, headline_hash
    """
    if not config.breaking_news.enabled:
        return {}

    threshold = config.breaking_news.threshold

    # Build feed list
    feeds: list[dict] = []
    if config.breaking_news.default_feeds:
        feeds.extend(DEFAULT_WIRE_FEEDS)
    feeds.extend(config.breaking_news.wire_feeds)

    if not feeds:
        return {}

    # Poll all feeds concurrently
    all_items = await _poll_all_feeds(feeds)
    if not all_items:
        return {}

    logger.info(f"Breaking news: {len(all_items)} headlines from {len(feeds)} feeds")

    # Score ALL topics concurrently
    topic_tasks = [
        _score_topic(llm, topic, all_items, store, threshold)
        for topic in config.topics
    ]
    topic_results = await asyncio.gather(*topic_tasks, return_exceptions=True)

    alerts_by_topic: dict[str, list[dict]] = {}
    for topic, result in zip(config.topics, topic_results):
        if isinstance(result, Exception):
            logger.error(f"Breaking scoring failed for {topic.name}: {result}")
            continue
        if result:
            alerts_by_topic[_topic_slug(topic)] = result

    return alerts_by_topic
