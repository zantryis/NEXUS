"""Tests for RSS source polling."""

import pytest
from unittest.mock import patch, MagicMock
from nexus.engine.sources.polling import poll_feed, poll_all_feeds, ContentItem


SAMPLE_RSS = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Test Feed</title>
    <item>
      <title>Article One</title>
      <link>https://example.com/article-1</link>
      <pubDate>Mon, 10 Mar 2026 12:00:00 GMT</pubDate>
      <description>First article summary.</description>
    </item>
    <item>
      <title>Article Two</title>
      <link>https://example.com/article-2</link>
      <pubDate>Mon, 10 Mar 2026 10:00:00 GMT</pubDate>
      <description>Second article summary.</description>
    </item>
  </channel>
</rss>
"""


def test_content_item_model():
    item = ContentItem(
        title="Test",
        url="https://example.com",
        source_id="test-feed",
        snippet="A snippet",
    )
    assert item.title == "Test"
    assert item.full_text is None


def test_poll_feed_parses_entries():
    with patch("nexus.engine.sources.polling.feedparser.parse") as mock_parse:
        mock_parse.return_value = MagicMock(
            bozo=False,
            entries=[
                MagicMock(
                    title="Article One",
                    link="https://example.com/1",
                    get=lambda k, d=None: "Summary" if k == "summary" else d,
                    **{"published_parsed": None},
                ),
            ],
        )
        items = poll_feed("https://example.com/feed", "test-feed")
        assert len(items) == 1
        assert items[0].title == "Article One"
        assert items[0].source_id == "test-feed"


def test_poll_feed_handles_bad_feed():
    with patch("nexus.engine.sources.polling.feedparser.parse") as mock_parse:
        mock_parse.return_value = MagicMock(bozo=True, entries=[])
        items = poll_feed("https://bad-feed.com/rss", "bad")
        assert items == []


def test_poll_all_feeds():
    sources = [
        {"url": "https://example.com/feed1", "id": "feed1"},
        {"url": "https://example.com/feed2", "id": "feed2"},
    ]
    with patch("nexus.engine.sources.polling.poll_feed") as mock_poll:
        mock_poll.side_effect = [
            [ContentItem(title="A", url="https://a.com", source_id="feed1")],
            [ContentItem(title="B", url="https://b.com", source_id="feed2")],
        ]
        items = poll_all_feeds(sources)
        assert len(items) == 2
        assert mock_poll.call_count == 2


def test_poll_all_feeds_skips_failed():
    sources = [
        {"url": "https://example.com/good", "id": "good"},
        {"url": "https://example.com/bad", "id": "bad"},
    ]
    with patch("nexus.engine.sources.polling.poll_feed") as mock_poll:
        mock_poll.side_effect = [
            [ContentItem(title="A", url="https://a.com", source_id="good")],
            [],  # bad feed returns empty
        ]
        items = poll_all_feeds(sources)
        assert len(items) == 1
