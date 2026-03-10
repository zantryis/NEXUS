"""Tests for content ingestion — article text extraction."""

import pytest
from unittest.mock import patch
from nexus.engine.sources.polling import ContentItem
from nexus.engine.ingestion.ingest import ingest_item, ingest_items


@pytest.fixture
def sample_item():
    return ContentItem(
        title="Test Article",
        url="https://example.com/article",
        source_id="test-feed",
        snippet="A test article.",
    )


def test_ingest_item_extracts_text(sample_item):
    with patch("nexus.engine.ingestion.ingest.trafilatura") as mock_traf:
        mock_traf.fetch_url.return_value = "<html><body>Full article text here.</body></html>"
        mock_traf.extract.return_value = "Full article text here."

        result = ingest_item(sample_item)
        assert result.full_text == "Full article text here."
        mock_traf.fetch_url.assert_called_once_with(sample_item.url)


def test_ingest_item_handles_fetch_failure(sample_item):
    with patch("nexus.engine.ingestion.ingest.trafilatura") as mock_traf:
        mock_traf.fetch_url.return_value = None

        result = ingest_item(sample_item)
        assert result.full_text is None


def test_ingest_item_handles_extraction_failure(sample_item):
    with patch("nexus.engine.ingestion.ingest.trafilatura") as mock_traf:
        mock_traf.fetch_url.return_value = "<html></html>"
        mock_traf.extract.return_value = None

        result = ingest_item(sample_item)
        assert result.full_text is None


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
