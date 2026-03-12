"""RSS source polling — fetch latest items from registered feeds."""

import logging
from datetime import datetime
from time import mktime
from typing import Optional

import feedparser
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class ContentItem(BaseModel):
    """A piece of content fetched from a source."""
    title: str
    url: str
    source_id: str
    snippet: str = ""
    published: Optional[datetime] = None
    full_text: Optional[str] = None
    language: Optional[str] = None
    relevance_score: Optional[float] = None
    # Source metadata (populated from registry)
    source_language: Optional[str] = None
    source_affiliation: Optional[str] = None
    source_country: Optional[str] = None
    source_tier: Optional[str] = None
    # Post-ingestion metadata
    detected_language: Optional[str] = None
    extraction_status: str = "pending"
    extraction_error: Optional[str] = None


def poll_feed(
    url: str,
    source_id: str,
    source_language: Optional[str] = None,
    source_affiliation: Optional[str] = None,
    source_country: Optional[str] = None,
    source_tier: Optional[str] = None,
) -> list[ContentItem]:
    """Fetch and parse a single RSS feed. Returns empty list on failure."""
    feed = feedparser.parse(url)
    if feed.bozo and not feed.entries:
        logger.warning(f"Failed to parse feed {source_id}: {url}")
        return []

    items = []
    for entry in feed.entries:
        published = None
        if hasattr(entry, "published_parsed") and entry.published_parsed:
            published = datetime.fromtimestamp(mktime(entry.published_parsed))

        items.append(ContentItem(
            title=entry.title,
            url=entry.link,
            source_id=source_id,
            snippet=entry.get("summary", ""),
            published=published,
            language=source_language,
            source_language=source_language,
            source_affiliation=source_affiliation,
            source_country=source_country,
            source_tier=source_tier,
        ))
    return items


def poll_all_feeds(sources: list[dict]) -> list[ContentItem]:
    """Poll all feeds from a list of source dicts."""
    all_items = []
    for source in sources:
        items = poll_feed(
            source["url"],
            source["id"],
            source_language=source.get("language"),
            source_affiliation=source.get("affiliation"),
            source_country=source.get("country"),
            source_tier=source.get("tier"),
        )
        all_items.extend(items)
    return all_items
