"""RSS source polling — fetch latest items from registered feeds."""

import logging
from datetime import datetime
from time import mktime
from typing import Optional

import feedparser
from pydantic import BaseModel, Field

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


def poll_feed(url: str, source_id: str) -> list[ContentItem]:
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
        ))
    return items


def poll_all_feeds(sources: list[dict]) -> list[ContentItem]:
    """Poll all feeds from a list of source dicts with 'url' and 'id' keys."""
    all_items = []
    for source in sources:
        items = poll_feed(source["url"], source["id"])
        all_items.extend(items)
    return all_items
