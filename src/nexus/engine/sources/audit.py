"""Source quality audit — score each feed's articles against a topic.

Polls sources, scores article relevance via LLM, and classifies each source
as keep (≥5.0), review (2.0–5.0), or drop (<2.0).

Usage: python -m nexus audit-sources <topic-slug>
"""

import logging
from collections import defaultdict

from nexus.config.models import TopicConfig
from nexus.engine.filtering.filter import score_batch
from nexus.engine.sources.polling import ContentItem, poll_all_feeds
from nexus.llm.client import LLMClient

logger = logging.getLogger(__name__)

KEEP_THRESHOLD = 5.0
DROP_THRESHOLD = 2.0


async def audit_source(
    llm: LLMClient,
    source_id: str,
    articles: list[ContentItem],
    topic: TopicConfig,
) -> dict:
    """Score a single source's articles against a topic.

    Returns dict with source_id, mean_score, verdict, n_articles, scores.
    """
    if not articles:
        return {
            "source_id": source_id,
            "mean_score": 0.0,
            "verdict": "dead",
            "n_articles": 0,
            "scores": [],
        }

    scores = await score_batch(llm, articles, topic)
    score_values = [s for s, _ in scores]
    mean = sum(score_values) / len(score_values) if score_values else 0.0

    if mean >= KEEP_THRESHOLD:
        verdict = "keep"
    elif mean >= DROP_THRESHOLD:
        verdict = "review"
    else:
        verdict = "drop"

    return {
        "source_id": source_id,
        "mean_score": round(mean, 2),
        "verdict": verdict,
        "n_articles": len(articles),
        "scores": scores,
    }


async def audit_registry(
    llm: LLMClient,
    sources: list[dict],
    topic: TopicConfig,
    max_articles_per_source: int = 10,
) -> list[dict]:
    """Audit all sources in a registry against a topic.

    Polls feeds, groups articles by source, scores each source.
    """
    all_articles = poll_all_feeds(sources)

    # Group by source_id
    by_source: dict[str, list[ContentItem]] = defaultdict(list)
    for article in all_articles:
        by_source[article.source_id].append(article)

    results = []
    source_ids = {s.get("id", s.get("url", "")) for s in sources}

    for source_id in sorted(source_ids):
        articles = by_source.get(source_id, [])[:max_articles_per_source]

        if not articles:
            results.append({
                "source_id": source_id,
                "mean_score": 0.0,
                "verdict": "dead",
                "n_articles": 0,
                "scores": [],
            })
            continue

        result = await audit_source(llm, source_id, articles, topic)
        results.append(result)
        logger.info(
            f"  {result['verdict'].upper():6s} {result['mean_score']:4.1f}  "
            f"{result['n_articles']:3d} articles  {source_id}"
        )

    return results
