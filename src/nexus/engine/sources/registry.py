"""Global source registry — curated pool of verified RSS feeds."""

import logging
from pathlib import Path

import feedparser
import yaml
from pydantic import BaseModel, Field

from nexus.config.models import TopicConfig

logger = logging.getLogger(__name__)


class GlobalSource(BaseModel):
    """A source in the global registry.

    Affiliations:
        state    — government-controlled editorial (CGTN, RT, TASS, Anadolu)
        public   — publicly funded, editorially independent (BBC, NHK, DW)
        private  — corporate/private ownership (NYT, Guardian, SCMP)
        nonprofit — NGO or nonprofit (Carbon Brief)
        academic — academic institution (arXiv)
    """
    id: str
    name: str
    url: str
    language: str = "en"
    tier: str = "A"
    tags: list[str] = Field(default_factory=list)
    affiliation: str = "private"
    country: str = ""


def load_global_registry(path: Path) -> list[GlobalSource]:
    """Load the global source registry from YAML."""
    if not path.exists():
        return []
    raw = yaml.safe_load(path.read_text())
    if not raw or "sources" not in raw:
        return []
    return [GlobalSource(**s) for s in raw["sources"]]


def sources_for_topic(
    sources: list[GlobalSource],
    topic: TopicConfig,
    tag_hints: list[str],
) -> list[GlobalSource]:
    """Select sources matching a topic by tag overlap and language filter."""
    tag_set = set(tag_hints)
    matched = []
    for s in sources:
        if s.language not in topic.source_languages:
            continue
        if tag_set & set(s.tags):
            matched.append(s)
    # Sort: tier A first, then by number of tag matches (desc)
    matched.sort(key=lambda s: (s.tier, -len(tag_set & set(s.tags))))
    return matched


def check_feed_health(source: GlobalSource) -> dict:
    """Check if a feed is reachable and returning entries."""
    try:
        feed = feedparser.parse(source.url)
        n = len(feed.entries)
        title = feed.feed.get("title", "?")
        if n > 0:
            return {"id": source.id, "status": "ok", "entries": n, "title": title}
        else:
            return {"id": source.id, "status": "empty", "entries": 0, "title": title}
    except Exception as e:
        return {"id": source.id, "status": "error", "entries": 0, "title": str(e)}


def build_topic_registry(matched: list[GlobalSource]) -> dict:
    """Convert matched global sources to a topic registry format."""
    return {
        "sources": [
            {"id": s.id, "type": "rss", "url": s.url, "tier": s.tier, "language": s.language}
            for s in matched
        ]
    }
