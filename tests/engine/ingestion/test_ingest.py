"""Tests for content ingestion — article text extraction."""

import json
import pytest
from unittest.mock import patch, MagicMock
from nexus.engine.sources.polling import ContentItem
from nexus.engine.ingestion.ingest import (
    ingest_item, ingest_items, async_ingest_items,
    _detect_paywall,
)


@pytest.fixture
def sample_item():
    return ContentItem(
        title="Test Article",
        url="https://example.com/article",
        source_id="test-feed",
        snippet="A test article.",
        source_language="en",
    )


def _mock_trafilatura(fetch_html=None, extract_json=None):
    """Helper to mock trafilatura with JSON extraction support."""
    mock = MagicMock()
    mock.fetch_url.return_value = fetch_html
    mock.extract.return_value = extract_json
    return mock


def test_ingest_item_extracts_text(sample_item):
    extracted = json.dumps({"text": "Full article text here.", "language": "en"})
    with patch("nexus.engine.ingestion.ingest.trafilatura") as mock_traf:
        mock_traf.fetch_url.return_value = "<html><body>Full article text.</body></html>"
        mock_traf.extract.return_value = extracted

        result = ingest_item(sample_item)
        assert result.full_text == "Full article text here."
        assert result.extraction_status == "ok"
        assert result.detected_language == "en"
        mock_traf.fetch_url.assert_called_once_with(sample_item.url)


def test_ingest_item_handles_fetch_failure(sample_item):
    with patch("nexus.engine.ingestion.ingest.trafilatura") as mock_traf:
        mock_traf.fetch_url.return_value = None

        result = ingest_item(sample_item)
        assert result.full_text is None
        assert result.extraction_status == "fetch_failed"


def test_ingest_item_handles_extraction_failure(sample_item):
    with patch("nexus.engine.ingestion.ingest.trafilatura") as mock_traf:
        mock_traf.fetch_url.return_value = "<html></html>"
        mock_traf.extract.return_value = None

        result = ingest_item(sample_item)
        assert result.full_text == ""
        assert result.extraction_status == "extract_failed"


def test_ingest_item_detects_language(sample_item):
    extracted = json.dumps({"text": "مقاله فارسی", "language": "fa"})
    with patch("nexus.engine.ingestion.ingest.trafilatura") as mock_traf:
        mock_traf.fetch_url.return_value = "<html><body>content</body></html>"
        mock_traf.extract.return_value = extracted

        result = ingest_item(sample_item)
        assert result.detected_language == "fa"


def test_ingest_item_falls_back_to_source_language(sample_item):
    """If trafilatura doesn't detect language, use source_language."""
    extracted = json.dumps({"text": "Some text", "language": None})
    with patch("nexus.engine.ingestion.ingest.trafilatura") as mock_traf:
        mock_traf.fetch_url.return_value = "<html><body>content</body></html>"
        mock_traf.extract.return_value = extracted

        result = ingest_item(sample_item)
        assert result.detected_language == "en"  # Falls back to source_language


def test_ingest_item_detects_paywall(sample_item):
    html = "<html><body><p>Subscribe to continue reading</p></body></html>"
    extracted = json.dumps({"text": "Partial...", "language": "en"})
    with patch("nexus.engine.ingestion.ingest.trafilatura") as mock_traf:
        mock_traf.fetch_url.return_value = html
        mock_traf.extract.return_value = extracted

        result = ingest_item(sample_item)
        assert result.extraction_status == "paywall"
        assert result.full_text == "Partial..."


def test_detect_paywall_heuristics():
    assert _detect_paywall("<p>Subscribe to continue reading</p>")
    assert _detect_paywall("<div class='paywall'>Content locked</div>")
    assert _detect_paywall("<p>This article is for subscribers only</p>")
    assert not _detect_paywall("<p>A normal article about technology</p>")
    assert not _detect_paywall("<p>Free to read article content</p>")


def test_ingest_items_filters_empty():
    items = [
        ContentItem(title="A", url="https://a.com", source_id="f1"),
        ContentItem(title="B", url="https://b.com", source_id="f2"),
    ]
    with patch("nexus.engine.ingestion.ingest.ingest_item") as mock_ingest:
        item_with_text = items[0].model_copy()
        item_with_text.full_text = "Some text"
        item_without = items[1].model_copy()
        item_without.full_text = None
        mock_ingest.side_effect = [item_with_text, item_without]

        results = ingest_items(items)
        assert len(results) == 1
        assert results[0].full_text == "Some text"


@pytest.mark.asyncio
async def test_async_ingest_items():
    items = [
        ContentItem(title="A", url="https://a.com/1", source_id="f1"),
        ContentItem(title="B", url="https://b.com/2", source_id="f2"),
    ]
    extracted = json.dumps({"text": "Article text", "language": "en"})
    with patch("nexus.engine.ingestion.ingest.trafilatura") as mock_traf:
        mock_traf.fetch_url.return_value = "<html><body>text</body></html>"
        mock_traf.extract.return_value = extracted

        results = await async_ingest_items(items)
        assert len(results) == 2
        assert all(r.full_text == "Article text" for r in results)


@pytest.mark.asyncio
async def test_async_ingest_items_empty():
    results = await async_ingest_items([])
    assert results == []
