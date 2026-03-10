"""Content ingestion — fetch full article text from URLs."""

import logging
import trafilatura
from nexus.engine.sources.polling import ContentItem

logger = logging.getLogger(__name__)


def ingest_item(item: ContentItem) -> ContentItem:
    """Fetch and extract full text for a single content item."""
    html = trafilatura.fetch_url(item.url)
    if not html:
        logger.warning(f"Failed to fetch: {item.url}")
        return item

    text = trafilatura.extract(html)
    if not text:
        logger.warning(f"Failed to extract text: {item.url}")
        return item

    item.full_text = text
    return item


def ingest_items(items: list[ContentItem]) -> list[ContentItem]:
    """Ingest all items, return only those with extracted text."""
    results = []
    for item in items:
        ingested = ingest_item(item)
        if ingested.full_text:
            results.append(ingested)
    return results
