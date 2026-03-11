"""Tests for Reddit source adapter."""

import pytest
from unittest.mock import patch

from nexus.engine.sources.reddit import RedditAdapter
from nexus.engine.sources.polling import ContentItem


@patch("nexus.engine.sources.reddit.poll_feed")
async def test_reddit_builds_correct_url(mock_poll_feed):
    """Reddit adapter constructs the correct RSS URL from subreddit config."""
    mock_poll_feed.return_value = []
    adapter = RedditAdapter()
    await adapter.poll({"subreddit": "worldnews", "id": "reddit-worldnews"})

    mock_poll_feed.assert_called_once()
    call_args = mock_poll_feed.call_args
    assert call_args[0][0] == "https://www.reddit.com/r/worldnews/top/.rss?t=day"
    assert call_args[0][1] == "reddit-worldnews"


@patch("nexus.engine.sources.reddit.poll_feed")
async def test_reddit_delegates_to_poll_feed(mock_poll_feed):
    """Reddit adapter passes config metadata through to poll_feed."""
    mock_poll_feed.return_value = [
        ContentItem(
            title="Reddit Post",
            url="https://reddit.com/r/worldnews/1",
            source_id="reddit-worldnews",
        )
    ]
    adapter = RedditAdapter()
    items = await adapter.poll({
        "subreddit": "worldnews",
        "id": "reddit-worldnews",
        "language": "en",
        "affiliation": "social",
        "country": "US",
        "tier": "C",
        "sort": "hot",
    })

    assert len(items) == 1
    assert items[0].title == "Reddit Post"
    mock_poll_feed.assert_called_once_with(
        "https://www.reddit.com/r/worldnews/hot/.rss?t=day",
        "reddit-worldnews",
        source_language="en",
        source_affiliation="social",
        source_country="US",
        source_tier="C",
    )


async def test_reddit_missing_subreddit():
    """Missing subreddit returns empty list."""
    adapter = RedditAdapter()
    items = await adapter.poll({"id": "reddit-none"})
    assert items == []
