"""Reddit source adapter — uses Reddit's RSS interface."""

import logging

from nexus.engine.sources.base import SourceAdapter
from nexus.engine.sources.polling import ContentItem, poll_feed

logger = logging.getLogger(__name__)


class RedditAdapter(SourceAdapter):
    """Adapter for Reddit subreddits via RSS."""

    source_type = "reddit"

    async def poll(self, source_config: dict) -> list[ContentItem]:
        subreddit = source_config.get("subreddit", "")
        sort = source_config.get("sort", "top")
        if not subreddit:
            return []

        url = f"https://www.reddit.com/r/{subreddit}/{sort}/.rss?t=day"
        source_id = source_config.get("id", f"reddit-{subreddit}")

        return poll_feed(
            url,
            source_id,
            source_language=source_config.get("language", "en"),
            source_affiliation=source_config.get("affiliation", "social"),
            source_country=source_config.get("country"),
            source_tier=source_config.get("tier", "C"),
        )
