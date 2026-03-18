"""Tests for RSSAdapter and filter_recent."""

import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from nexus.engine.sources.polling import ContentItem, filter_recent
from nexus.engine.sources.rss import RSSAdapter


# ── filter_recent ──


def _item(title: str, published: datetime | None = None) -> ContentItem:
    return ContentItem(title=title, url=f"https://example.com/{title}", source_id="test", published=published)


def test_filter_recent_keeps_recent_items():
    now = datetime.now(timezone.utc)
    items = [
        _item("fresh", now - timedelta(hours=1)),
        _item("also_fresh", now - timedelta(hours=24)),
    ]
    result = filter_recent(items, max_age_hours=48)
    assert len(result) == 2


def test_filter_recent_drops_old_items():
    now = datetime.now(timezone.utc)
    items = [
        _item("fresh", now - timedelta(hours=1)),
        _item("stale", now - timedelta(hours=72)),
    ]
    result = filter_recent(items, max_age_hours=48)
    assert len(result) == 1
    assert result[0].title == "fresh"


def test_filter_recent_keeps_items_without_date():
    """Items with no published date should be kept (let filter stage decide)."""
    items = [
        _item("no_date", published=None),
        _item("also_no_date", published=None),
    ]
    result = filter_recent(items, max_age_hours=48)
    assert len(result) == 2


def test_filter_recent_handles_naive_datetimes():
    """Naive datetimes (no timezone) should be treated as UTC."""
    now_naive = datetime.now()
    items = [
        _item("naive_fresh", now_naive - timedelta(hours=1)),
        _item("naive_stale", now_naive - timedelta(hours=72)),
    ]
    result = filter_recent(items, max_age_hours=48)
    assert len(result) == 1
    assert result[0].title == "naive_fresh"


def test_filter_recent_boundary_just_inside_cutoff():
    """Item just inside the cutoff boundary should be kept."""
    now = datetime.now(timezone.utc)
    # 1 second inside the window to avoid timing races
    items = [_item("boundary", now - timedelta(hours=48) + timedelta(seconds=1))]
    result = filter_recent(items, max_age_hours=48)
    assert len(result) == 1


def test_filter_recent_empty_list():
    assert filter_recent([], max_age_hours=48) == []


def test_filter_recent_mixed_tz_and_naive():
    """Mix of timezone-aware and naive datetimes."""
    now_utc = datetime.now(timezone.utc)
    now_naive = datetime.now()
    items = [
        _item("tz_aware", now_utc - timedelta(hours=1)),
        _item("naive", now_naive - timedelta(hours=1)),
        _item("no_date"),
        _item("old_tz", now_utc - timedelta(hours=100)),
    ]
    result = filter_recent(items, max_age_hours=48)
    assert len(result) == 3
    titles = {r.title for r in result}
    assert "old_tz" not in titles


def test_filter_recent_custom_age():
    now = datetime.now(timezone.utc)
    items = [
        _item("within_6h", now - timedelta(hours=5)),
        _item("outside_6h", now - timedelta(hours=8)),
    ]
    result = filter_recent(items, max_age_hours=6)
    assert len(result) == 1
    assert result[0].title == "within_6h"


# ── RSSAdapter ──


@pytest.mark.asyncio
async def test_rss_adapter_delegates_to_poll_feed():
    """RSSAdapter.poll should call poll_feed with correct args."""
    adapter = RSSAdapter()
    assert adapter.source_type == "rss"

    source_config = {
        "url": "https://example.com/feed",
        "id": "test-feed",
        "language": "en",
        "affiliation": "public",
        "country": "US",
        "tier": "A",
    }

    with patch("nexus.engine.sources.rss.poll_feed") as mock_poll:
        mock_poll.return_value = [
            ContentItem(title="Test", url="https://example.com/1", source_id="test-feed")
        ]
        result = await adapter.poll(source_config)

    assert len(result) == 1
    mock_poll.assert_called_once_with(
        "https://example.com/feed",
        "test-feed",
        source_language="en",
        source_affiliation="public",
        source_country="US",
        source_tier="A",
    )


@pytest.mark.asyncio
async def test_rss_adapter_handles_missing_optional_fields():
    """RSSAdapter should pass None for missing optional source config fields."""
    adapter = RSSAdapter()
    source_config = {"url": "https://example.com/feed", "id": "minimal"}

    with patch("nexus.engine.sources.rss.poll_feed") as mock_poll:
        mock_poll.return_value = []
        await adapter.poll(source_config)

    mock_poll.assert_called_once_with(
        "https://example.com/feed",
        "minimal",
        source_language=None,
        source_affiliation=None,
        source_country=None,
        source_tier=None,
    )


@pytest.mark.asyncio
async def test_rss_adapter_returns_empty_on_failure():
    """RSSAdapter should return empty list when poll_feed returns empty."""
    adapter = RSSAdapter()
    source_config = {"url": "https://bad.com/feed", "id": "bad"}

    with patch("nexus.engine.sources.rss.poll_feed") as mock_poll:
        mock_poll.return_value = []
        result = await adapter.poll(source_config)

    assert result == []
