"""Tests for configurable breaking news wire feeds."""

import json
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from nexus.config.models import NexusConfig, UserConfig, BreakingNewsConfig
from nexus.agent.breaking import check_breaking_news, WIRE_FEEDS


def _make_config(**bn_kw) -> NexusConfig:
    return NexusConfig(
        user=UserConfig(name="Tristan"),
        breaking_news=BreakingNewsConfig(**bn_kw),
    )


@patch("nexus.agent.breaking.feedparser")
async def test_breaking_uses_default_feeds(mock_fp):
    """When wire_feeds is empty, the hardcoded WIRE_FEEDS are used."""
    mock_fp.parse.return_value = MagicMock(entries=[])
    config = _make_config()  # No wire_feeds
    store = AsyncMock()

    await check_breaking_news(AsyncMock(), config, store)

    # feedparser.parse should be called with the default WIRE_FEEDS
    called_urls = [c.args[0] for c in mock_fp.parse.call_args_list]
    for url in WIRE_FEEDS:
        assert url in called_urls


@patch("nexus.agent.breaking.poll_source")
@patch("nexus.agent.breaking.feedparser")
async def test_breaking_uses_custom_feeds(mock_fp, mock_poll_source):
    """When wire_feeds is set, those are used instead of defaults."""
    from nexus.engine.sources.polling import ContentItem

    mock_poll_source.return_value = [
        ContentItem(
            title="Custom feed headline",
            url="https://custom.com/article",
            source_id="custom-feed",
        )
    ]

    store = AsyncMock()
    store.is_alerted = AsyncMock(return_value=False)

    llm = AsyncMock()
    llm.complete = AsyncMock(return_value=json.dumps({"scores": []}))

    config = _make_config(wire_feeds=[
        {"type": "telegram_channel", "channel": "@reuters", "id": "tg-reuters"},
    ])
    await check_breaking_news(llm, config, store)

    # poll_source should be called with the custom feed config
    mock_poll_source.assert_called()
    # feedparser should NOT be called since we're using custom feeds via router
    mock_fp.parse.assert_not_called()


@patch("nexus.agent.breaking.poll_source")
@patch("nexus.agent.breaking.feedparser")
async def test_breaking_wire_feed_uses_router(mock_fp, mock_poll_source):
    """Custom wire feeds are polled through the source router."""
    from nexus.engine.sources.polling import ContentItem

    mock_poll_source.return_value = [
        ContentItem(
            title="Routed headline",
            url="https://routed.com/1",
            source_id="routed",
        )
    ]

    store = AsyncMock()
    store.is_alerted = AsyncMock(return_value=False)

    llm = AsyncMock()
    llm.complete = AsyncMock(return_value=json.dumps({
        "scores": [{"index": 0, "score": 9, "headline": "Routed headline"}]
    }))

    wire_feed = {"type": "rss", "url": "https://custom-wire.com/rss", "id": "custom-wire"}
    config = _make_config(threshold=7, wire_feeds=[wire_feed])

    alerts = await check_breaking_news(llm, config, store)

    mock_poll_source.assert_called_with(wire_feed)
    assert len(alerts) == 1
    assert alerts[0]["headline"] == "Routed headline"
    store.add_breaking_alert.assert_called_once()
