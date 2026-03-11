"""Source routing — dispatch polling to the correct adapter by type."""

import logging

from nexus.engine.sources.base import SourceAdapter
from nexus.engine.sources.polling import ContentItem
from nexus.engine.sources.rss import RSSAdapter

logger = logging.getLogger(__name__)

ADAPTERS: dict[str, SourceAdapter] = {}


def register_adapter(adapter: SourceAdapter):
    """Register a source adapter for its declared type."""
    ADAPTERS[adapter.source_type] = adapter


def _ensure_defaults():
    """Lazily register built-in adapters."""
    if "rss" not in ADAPTERS:
        register_adapter(RSSAdapter())
    if "telegram_channel" not in ADAPTERS:
        from nexus.engine.sources.telegram_channel import TelegramChannelAdapter

        register_adapter(TelegramChannelAdapter())
    if "reddit" not in ADAPTERS:
        from nexus.engine.sources.reddit import RedditAdapter

        register_adapter(RedditAdapter())
    if "twitter" not in ADAPTERS:
        from nexus.engine.sources.twitter import TwitterAdapter

        register_adapter(TwitterAdapter())


async def poll_source(source_config: dict) -> list[ContentItem]:
    """Poll a single source through the appropriate adapter."""
    _ensure_defaults()
    source_type = source_config.get("type", "rss")
    adapter = ADAPTERS.get(source_type)
    if not adapter:
        logger.warning(f"No adapter for source type: {source_type}")
        return []
    try:
        return await adapter.poll(source_config)
    except Exception as e:
        logger.warning(
            f"Adapter {source_type} failed for {source_config.get('id', '?')}: {e}"
        )
        return []


async def poll_all_sources(sources: list[dict]) -> list[ContentItem]:
    """Poll all sources through their respective adapters."""
    _ensure_defaults()
    all_items = []
    for source in sources:
        items = await poll_source(source)
        all_items.extend(items)
    return all_items
