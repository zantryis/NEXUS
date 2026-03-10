"""Tests for breaking news poller."""

import json
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from nexus.config.models import NexusConfig, UserConfig, BreakingNewsConfig
from nexus.agent.breaking import check_breaking_news, _hash_headline


def _make_config(**bn_kw) -> NexusConfig:
    return NexusConfig(
        user=UserConfig(name="Tristan"),
        breaking_news=BreakingNewsConfig(**bn_kw),
    )


def test_hash_headline():
    h = _hash_headline("Major event happened")
    assert isinstance(h, str)
    assert len(h) == 16
    # Same headline = same hash
    assert _hash_headline("Major event happened") == h
    # Different headline = different hash
    assert _hash_headline("Different headline") != h


async def test_breaking_news_disabled():
    config = _make_config(enabled=False)
    result = await check_breaking_news(AsyncMock(), config, AsyncMock())
    assert result == []


@patch("nexus.agent.breaking.feedparser")
async def test_breaking_news_finds_alerts(mock_fp, tmp_path):
    """When LLM scores headlines above threshold, alerts are returned."""
    # Mock feed
    mock_entry = MagicMock()
    mock_entry.get.side_effect = lambda k, d="": {
        "title": "Major crisis erupts", "link": "https://reuters.com/breaking"
    }.get(k, d)
    mock_fp.parse.return_value = MagicMock(entries=[mock_entry])

    # Mock store
    store = AsyncMock()
    store.is_alerted = AsyncMock(return_value=False)
    store.add_breaking_alert = AsyncMock(return_value=1)

    # Mock LLM
    llm = AsyncMock()
    llm.complete = AsyncMock(return_value=json.dumps({
        "scores": [{"index": 0, "score": 9, "headline": "Major crisis erupts"}]
    }))

    config = _make_config(threshold=7)
    alerts = await check_breaking_news(
        llm, config, store, feed_urls=["https://test.com/feed"]
    )

    assert len(alerts) == 1
    assert alerts[0]["headline"] == "Major crisis erupts"
    assert alerts[0]["significance_score"] == 9
    store.add_breaking_alert.assert_called_once()


@patch("nexus.agent.breaking.feedparser")
async def test_breaking_news_skips_alerted(mock_fp, tmp_path):
    """Already-alerted headlines should be skipped."""
    mock_entry = MagicMock()
    mock_entry.get.side_effect = lambda k, d="": {
        "title": "Old news", "link": "https://test.com/old"
    }.get(k, d)
    mock_fp.parse.return_value = MagicMock(entries=[mock_entry])

    store = AsyncMock()
    store.is_alerted = AsyncMock(return_value=True)  # Already alerted

    config = _make_config()
    alerts = await check_breaking_news(
        AsyncMock(), config, store, feed_urls=["https://test.com/feed"]
    )

    assert len(alerts) == 0


@patch("nexus.agent.breaking.feedparser")
async def test_breaking_news_below_threshold(mock_fp, tmp_path):
    """Headlines below threshold should not generate alerts."""
    mock_entry = MagicMock()
    mock_entry.get.side_effect = lambda k, d="": {
        "title": "Minor update", "link": "https://test.com/minor"
    }.get(k, d)
    mock_fp.parse.return_value = MagicMock(entries=[mock_entry])

    store = AsyncMock()
    store.is_alerted = AsyncMock(return_value=False)

    llm = AsyncMock()
    llm.complete = AsyncMock(return_value=json.dumps({
        "scores": [{"index": 0, "score": 3, "headline": "Minor update"}]
    }))

    config = _make_config(threshold=7)
    alerts = await check_breaking_news(
        llm, config, store, feed_urls=["https://test.com/feed"]
    )

    assert len(alerts) == 0
