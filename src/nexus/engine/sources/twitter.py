"""Twitter/Nitter source adapter — tries multiple Nitter instances."""

import logging

from nexus.engine.sources.base import SourceAdapter
from nexus.engine.sources.polling import ContentItem, poll_feed

logger = logging.getLogger(__name__)


class TwitterAdapter(SourceAdapter):
    """Adapter for Twitter via Nitter RSS proxies."""

    source_type = "twitter"

    DEFAULT_INSTANCES = [
        "nitter.net",
        "nitter.privacydev.net",
        "nitter.poast.org",
    ]

    async def poll(self, source_config: dict) -> list[ContentItem]:
        username = source_config.get("username", "")
        if not username:
            return []

        instances = source_config.get("nitter_instances", self.DEFAULT_INSTANCES)
        source_id = source_config.get("id", f"twitter-{username}")

        for instance in instances:
            url = f"https://{instance}/{username}/rss"
            items = poll_feed(
                url,
                source_id,
                source_language=source_config.get("language", "en"),
                source_affiliation=source_config.get("affiliation", "social"),
                source_country=source_config.get("country"),
                source_tier=source_config.get("tier", "C"),
            )
            if items:
                return items
            logger.warning(
                f"Nitter instance {instance} failed for @{username}, trying next..."
            )

        logger.warning(f"All Nitter instances failed for @{username}")
        return []
