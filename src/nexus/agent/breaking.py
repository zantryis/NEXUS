"""Breaking news poller — check wire feeds for high-significance alerts."""

import hashlib
import json
import logging

import feedparser

from nexus.config.models import NexusConfig
from nexus.engine.knowledge.store import KnowledgeStore
from nexus.llm.client import LLMClient

logger = logging.getLogger(__name__)

# Wire feed URLs for breaking news (major agencies)
WIRE_FEEDS = [
    "https://feeds.reuters.com/reuters/topNews",
    "https://rss.nytimes.com/services/xml/rss/nyt/World.xml",
]

SCORING_SYSTEM_PROMPT = (
    "You are a news significance scorer. Given a list of headlines, "
    "score each from 1-10 for global significance. "
    "Return JSON: {{\"scores\": [{{\"index\": 0, \"score\": 8, \"headline\": \"...\"}}]}}\n"
    "Only include headlines scoring 7 or above."
)


def _hash_headline(headline: str) -> str:
    """Create a dedup hash from a headline."""
    return hashlib.sha256(headline.lower().strip().encode()).hexdigest()[:16]


async def check_breaking_news(
    llm: LLMClient,
    config: NexusConfig,
    store: KnowledgeStore,
    feed_urls: list[str] | None = None,
) -> list[dict]:
    """Poll wire feeds and return new high-significance headlines.

    Returns list of dicts: {headline, source_url, significance_score, headline_hash}
    """
    if not config.breaking_news.enabled:
        return []

    urls = feed_urls or WIRE_FEEDS
    threshold = config.breaking_news.threshold

    # Gather headlines from feeds
    headlines = []
    for url in urls:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:20]:
                title = entry.get("title", "").strip()
                link = entry.get("link", "")
                if title:
                    h = _hash_headline(title)
                    if not await store.is_alerted(h):
                        headlines.append({
                            "title": title, "url": link, "hash": h,
                        })
        except Exception as e:
            logger.warning(f"Feed parse failed for {url}: {e}")

    if not headlines:
        return []

    # Score headlines via LLM
    headline_list = "\n".join(
        f"[{i}] {h['title']}" for i, h in enumerate(headlines)
    )
    try:
        response = await llm.complete(
            config_key="breaking_news",
            system_prompt=SCORING_SYSTEM_PROMPT,
            user_prompt=f"Headlines to score:\n{headline_list}",
            json_response=True,
        )
        data = json.loads(response)
        scores = data.get("scores", [])
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        logger.warning(f"Breaking news scoring failed: {e}")
        return []

    # Filter and record alerts
    alerts = []
    for s in scores:
        idx = s.get("index", -1)
        score = s.get("score", 0)
        if 0 <= idx < len(headlines) and score >= threshold:
            h = headlines[idx]
            await store.add_breaking_alert(h["hash"], h["title"], h["url"], score)
            alerts.append({
                "headline": h["title"],
                "source_url": h["url"],
                "significance_score": score,
                "headline_hash": h["hash"],
            })

    if alerts:
        logger.info(f"Breaking news: {len(alerts)} alerts above threshold {threshold}")

    return alerts
