"""RSS source adapter — delegates to existing poll_feed."""

import asyncio

from nexus.engine.sources.base import SourceAdapter
from nexus.engine.sources.polling import ContentItem, poll_feed


class RSSAdapter(SourceAdapter):
    """Adapter for RSS/Atom feeds."""

    source_type = "rss"

    async def poll(self, source_config: dict) -> list[ContentItem]:
        return await asyncio.to_thread(
            poll_feed,
            source_config["url"],
            source_config["id"],
            source_language=source_config.get("language"),
            source_affiliation=source_config.get("affiliation"),
            source_country=source_config.get("country"),
            source_tier=source_config.get("tier"),
        )
