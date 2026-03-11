"""Tests for Twitter/Nitter source adapter."""

import pytest
from unittest.mock import patch, call

from nexus.engine.sources.twitter import TwitterAdapter
from nexus.engine.sources.polling import ContentItem


@patch("nexus.engine.sources.twitter.poll_feed")
async def test_twitter_tries_multiple_instances(mock_poll_feed):
    """When first Nitter instance fails, adapter tries the next one."""
    mock_poll_feed.side_effect = [
        [],  # First instance returns nothing
        [ContentItem(title="Tweet", url="https://nitter.two/user/1", source_id="twitter-user")],
    ]
    adapter = TwitterAdapter()
    items = await adapter.poll({
        "username": "user",
        "id": "twitter-user",
        "nitter_instances": ["nitter.one", "nitter.two"],
    })

    assert len(items) == 1
    assert items[0].title == "Tweet"
    assert mock_poll_feed.call_count == 2
    # First call used nitter.one
    assert mock_poll_feed.call_args_list[0][0][0] == "https://nitter.one/user/rss"
    # Second call used nitter.two
    assert mock_poll_feed.call_args_list[1][0][0] == "https://nitter.two/user/rss"


@patch("nexus.engine.sources.twitter.poll_feed")
async def test_twitter_all_instances_fail(mock_poll_feed):
    """When all Nitter instances fail, returns empty list."""
    mock_poll_feed.return_value = []
    adapter = TwitterAdapter()
    items = await adapter.poll({
        "username": "user",
        "id": "twitter-user",
        "nitter_instances": ["nitter.one", "nitter.two"],
    })
    assert items == []
    assert mock_poll_feed.call_count == 2


async def test_twitter_missing_username():
    """Missing username returns empty list."""
    adapter = TwitterAdapter()
    items = await adapter.poll({"id": "twitter-none"})
    assert items == []


@patch("nexus.engine.sources.twitter.poll_feed")
async def test_twitter_custom_instances(mock_poll_feed):
    """Adapter uses config-provided instance list instead of defaults."""
    mock_poll_feed.return_value = [
        ContentItem(title="Tweet", url="https://custom.nitter/user/1", source_id="twitter-user")
    ]
    adapter = TwitterAdapter()
    items = await adapter.poll({
        "username": "user",
        "id": "twitter-user",
        "nitter_instances": ["custom.nitter"],
    })
    assert len(items) == 1
    mock_poll_feed.assert_called_once()
    assert "custom.nitter" in mock_poll_feed.call_args[0][0]
