"""Tests for configurable breaking news wire feeds."""

import json
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from nexus.config.models import NexusConfig, UserConfig, BreakingNewsConfig, TopicConfig
from nexus.agent.breaking import check_breaking_news, DEFAULT_WIRE_FEEDS
from nexus.engine.sources.polling import ContentItem


def _make_config(**bn_kw) -> NexusConfig:
    return NexusConfig(
        user=UserConfig(name="Tristan"),
        breaking_news=BreakingNewsConfig(**bn_kw),
        topics=[TopicConfig(name="Test Topic", subtopics=["testing"])],
    )


@patch("nexus.agent.breaking._poll_all_feeds")
async def test_breaking_uses_default_feeds(mock_poll):
    """When default_feeds=True, DEFAULT_WIRE_FEEDS are included."""
    mock_poll.return_value = []
    config = _make_config(default_feeds=True)
    store = AsyncMock()

    await check_breaking_news(AsyncMock(), config, store)

    call_args = mock_poll.call_args[0][0]
    assert len(call_args) >= len(DEFAULT_WIRE_FEEDS)
    feed_ids = [f.get("id", "") for f in call_args]
    assert "wire-reuters" in feed_ids
    assert "wire-nyt" in feed_ids
    assert "wire-bbc" in feed_ids


@patch("nexus.agent.breaking._poll_all_feeds")
async def test_breaking_no_default_feeds(mock_poll):
    """When default_feeds=False, only custom wire_feeds are used."""
    mock_poll.return_value = []

    custom_feed = {"type": "rss", "url": "https://custom.com/rss", "id": "custom"}
    config = _make_config(default_feeds=False, wire_feeds=[custom_feed])
    store = AsyncMock()

    await check_breaking_news(AsyncMock(), config, store)

    call_args = mock_poll.call_args[0][0]
    assert len(call_args) == 1
    assert call_args[0]["id"] == "custom"


@patch("nexus.agent.breaking._poll_all_feeds")
async def test_breaking_merges_default_and_custom(mock_poll):
    """Default feeds + custom wire_feeds are merged."""
    mock_poll.return_value = []

    custom_feed = {"type": "telegram_channel", "channel": "@reuters", "id": "tg-reuters"}
    config = _make_config(default_feeds=True, wire_feeds=[custom_feed])
    store = AsyncMock()

    await check_breaking_news(AsyncMock(), config, store)

    call_args = mock_poll.call_args[0][0]
    feed_ids = [f.get("id", "") for f in call_args]
    assert "wire-reuters" in feed_ids  # default
    assert "tg-reuters" in feed_ids    # custom


@patch("nexus.agent.breaking._poll_all_feeds")
async def test_breaking_empty_feeds_returns_empty(mock_poll):
    """No feeds configured → empty result."""
    config = _make_config(default_feeds=False, wire_feeds=[])
    result = await check_breaking_news(AsyncMock(), config, AsyncMock())
    assert result == {}
    mock_poll.assert_not_called()
