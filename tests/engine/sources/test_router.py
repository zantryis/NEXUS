"""Tests for source routing."""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from nexus.engine.sources.polling import ContentItem
from nexus.engine.sources.router import (
    poll_source,
    poll_all_sources,
    register_adapter,
    ADAPTERS,
)
from nexus.engine.sources.base import SourceAdapter


class DummyAdapter(SourceAdapter):
    source_type = "dummy"

    async def poll(self, source_config: dict) -> list[ContentItem]:
        return [
            ContentItem(
                title="Dummy item",
                url="https://dummy.com/1",
                source_id=source_config.get("id", "dummy"),
            )
        ]


@patch("nexus.engine.sources.rss.poll_feed")
async def test_poll_source_rss(mock_poll_feed):
    """RSS sources route through the RSSAdapter which delegates to poll_feed."""
    mock_poll_feed.return_value = [
        ContentItem(title="RSS Article", url="https://example.com/1", source_id="feed1")
    ]
    source = {
        "type": "rss",
        "url": "https://example.com/feed",
        "id": "feed1",
        "language": "en",
        "affiliation": "independent",
        "country": "US",
        "tier": "A",
    }
    items = await poll_source(source)
    assert len(items) == 1
    assert items[0].title == "RSS Article"
    mock_poll_feed.assert_called_once_with(
        "https://example.com/feed",
        "feed1",
        source_language="en",
        source_affiliation="independent",
        source_country="US",
        source_tier="A",
    )


async def test_poll_source_unknown_type():
    """Unknown source types return empty list without crashing."""
    source = {"type": "carrier_pigeon", "id": "pigeon1"}
    items = await poll_source(source)
    assert items == []


@patch("nexus.engine.sources.rss.poll_feed")
async def test_poll_all_sources(mock_poll_feed):
    """poll_all_sources processes multiple sources of mixed types."""
    mock_poll_feed.return_value = [
        ContentItem(title="RSS Item", url="https://example.com/1", source_id="feed1")
    ]
    sources = [
        {"type": "rss", "url": "https://example.com/feed", "id": "feed1"},
        {"type": "unknown_type", "id": "x"},
    ]
    items = await poll_all_sources(sources)
    # Only the RSS source should produce items
    assert len(items) == 1
    assert items[0].title == "RSS Item"


async def test_register_custom_adapter():
    """Custom adapters can be registered and used."""
    adapter = DummyAdapter()
    register_adapter(adapter)
    try:
        source = {"type": "dummy", "id": "d1"}
        items = await poll_source(source)
        assert len(items) == 1
        assert items[0].title == "Dummy item"
    finally:
        # Clean up
        ADAPTERS.pop("dummy", None)
