"""Tests for the engine pipeline orchestrator."""

import pytest
from datetime import date
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock
from nexus.config.models import NexusConfig, UserConfig, TopicConfig
from nexus.engine.pipeline import run_pipeline


@pytest.fixture
def config():
    return NexusConfig(
        user=UserConfig(name="Tristan", timezone="America/Denver"),
        topics=[
            TopicConfig(name="AI Research", priority="high", subtopics=["agents"]),
        ],
    )


@pytest.fixture
def data_dir(tmp_path):
    return tmp_path / "data"


@pytest.mark.asyncio
async def test_pipeline_produces_briefing(config, data_dir):
    mock_llm = AsyncMock()

    # Mock source polling
    with patch("nexus.engine.pipeline.load_source_registry") as mock_registry, \
         patch("nexus.engine.pipeline.poll_all_feeds") as mock_poll, \
         patch("nexus.engine.pipeline.ingest_items") as mock_ingest, \
         patch("nexus.engine.pipeline.filter_items") as mock_filter, \
         patch("nexus.engine.pipeline.extract_event") as mock_extract, \
         patch("nexus.engine.pipeline.generate_briefing") as mock_synth:

        from nexus.engine.sources.polling import ContentItem
        from nexus.engine.knowledge.events import Event

        mock_registry.return_value = [{"url": "https://feed.com/rss", "id": "test"}]
        item = ContentItem(
            title="AI News", url="https://example.com", source_id="test",
            full_text="Full article text", relevance_score=8,
        )
        mock_poll.return_value = [item]
        mock_ingest.return_value = [item]
        mock_filter.return_value = [item]
        mock_extract.return_value = Event(
            date=date(2026, 3, 9), summary="AI event", significance=8,
        )
        mock_synth.return_value = "# Daily Briefing\n\n## AI Research\n\nContent here."

        briefing_path = await run_pipeline(config, mock_llm, data_dir)

        assert briefing_path.exists()
        assert "AI Research" in briefing_path.read_text()
        mock_poll.assert_called_once()
        mock_synth.assert_called_once()
