"""Tests for URL normalization and deduplication."""

from nexus.engine.ingestion.dedup import normalize_url, dedup_items
from nexus.engine.sources.polling import ContentItem


def test_normalize_strips_utm_params():
    url = "https://example.com/article?utm_source=twitter&utm_medium=social&id=123"
    assert normalize_url(url) == "https://example.com/article?id=123"


def test_normalize_strips_fragment():
    url = "https://example.com/article#section-2"
    assert normalize_url(url) == "https://example.com/article"


def test_normalize_strips_trailing_slash():
    url = "https://example.com/article/"
    assert normalize_url(url) == "https://example.com/article"


def test_normalize_preserves_meaningful_params():
    url = "https://example.com/article?id=123&page=2"
    assert normalize_url(url) == "https://example.com/article?id=123&page=2"


def test_normalize_handles_clean_url():
    url = "https://example.com/article"
    assert normalize_url(url) == "https://example.com/article"


def test_dedup_removes_exact_duplicates():
    items = [
        ContentItem(title="A", url="https://example.com/1", source_id="feed1"),
        ContentItem(title="A copy", url="https://example.com/1", source_id="feed2"),
    ]
    result = dedup_items(items)
    assert len(result) == 1
    assert result[0].source_id == "feed1"  # keeps first seen


def test_dedup_removes_tracking_param_duplicates():
    items = [
        ContentItem(title="A", url="https://example.com/article", source_id="f1"),
        ContentItem(title="A", url="https://example.com/article?utm_source=rss", source_id="f2"),
    ]
    result = dedup_items(items)
    assert len(result) == 1


def test_dedup_keeps_different_articles():
    items = [
        ContentItem(title="A", url="https://example.com/1", source_id="f1"),
        ContentItem(title="B", url="https://example.com/2", source_id="f1"),
    ]
    result = dedup_items(items)
    assert len(result) == 2


def test_dedup_empty_list():
    assert dedup_items([]) == []
