"""Tests for the engine pipeline orchestrator."""

import pytest
from datetime import date
from unittest.mock import AsyncMock, patch
from nexus.config.models import NexusConfig, UserConfig, TopicConfig
from nexus.engine.pipeline import run_pipeline, _event_cap_for_topic


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

    with patch("nexus.engine.pipeline.load_source_registry") as mock_registry, \
         patch("nexus.engine.pipeline.poll_all_feeds") as mock_poll, \
         patch("nexus.engine.pipeline.async_ingest_items") as mock_ingest, \
         patch("nexus.engine.pipeline.filter_items") as mock_filter, \
         patch("nexus.engine.pipeline.extract_event") as mock_extract, \
         patch("nexus.engine.pipeline.synthesize_topic") as mock_synth_topic, \
         patch("nexus.engine.pipeline.render_text_briefing") as mock_render, \
         patch("nexus.engine.pipeline.maybe_compress") as mock_compress:

        from nexus.engine.sources.polling import ContentItem
        from nexus.engine.knowledge.events import Event
        from nexus.engine.synthesis.knowledge import TopicSynthesis, NarrativeThread

        mock_registry.return_value = [{"url": "https://feed.com/rss", "id": "test"}]
        item = ContentItem(
            title="AI News", url="https://example.com", source_id="test",
            full_text="Full article text", relevance_score=8,
        )
        mock_poll.return_value = [item]
        mock_ingest.return_value = [item]
        from nexus.engine.filtering.filter import FilterResult
        mock_filter.return_value = FilterResult(accepted=[item], log_entries=[])
        mock_extract.return_value = Event(
            date=date(2026, 3, 9), summary="AI event", significance=8,
        )
        mock_compress.return_value = None
        mock_synth_topic.return_value = TopicSynthesis(
            topic_name="AI Research",
            threads=[NarrativeThread(headline="AI event", significance=8)],
        )
        mock_render.return_value = "# Daily Briefing\n\n## AI Research\n\nContent here."

        briefing_path = await run_pipeline(config, mock_llm, data_dir)

        assert briefing_path.exists()
        assert "AI Research" in briefing_path.read_text()
        mock_poll.assert_called_once()
        mock_render.assert_called_once()


def test_event_cap_for_narrow_topic():
    topic = TopicConfig(name="Iran-US", scope="narrow")
    assert _event_cap_for_topic(topic) == 15


def test_event_cap_for_medium_topic():
    topic = TopicConfig(name="Energy", scope="medium")
    assert _event_cap_for_topic(topic) == 20


def test_event_cap_for_broad_topic():
    topic = TopicConfig(name="AI/ML", scope="broad")
    assert _event_cap_for_topic(topic) == 35


def test_event_cap_default_scope():
    topic = TopicConfig(name="Test")
    assert _event_cap_for_topic(topic) == 20


def test_event_cap_max_events_override():
    topic = TopicConfig(name="Test", scope="narrow", max_events=50)
    assert _event_cap_for_topic(topic) == 50


@pytest.mark.asyncio
async def test_run_pipeline_passes_api_keys_to_audio(config, data_dir):
    """run_pipeline should forward openai and elevenlabs keys to run_audio_pipeline."""
    mock_llm = AsyncMock()

    with patch("nexus.engine.pipeline.load_source_registry") as mock_registry, \
         patch("nexus.engine.pipeline.poll_all_feeds") as mock_poll, \
         patch("nexus.engine.pipeline.async_ingest_items") as mock_ingest, \
         patch("nexus.engine.pipeline.filter_items") as mock_filter, \
         patch("nexus.engine.pipeline.extract_event") as mock_extract, \
         patch("nexus.engine.pipeline.synthesize_topic") as mock_synth_topic, \
         patch("nexus.engine.pipeline.render_text_briefing") as mock_render, \
         patch("nexus.engine.pipeline.maybe_compress") as mock_compress, \
         patch("nexus.engine.pipeline.run_audio_pipeline") as mock_audio:

        from nexus.engine.sources.polling import ContentItem
        from nexus.engine.knowledge.events import Event
        from nexus.engine.synthesis.knowledge import TopicSynthesis, NarrativeThread

        mock_registry.return_value = [{"url": "https://feed.com/rss", "id": "test"}]
        item = ContentItem(
            title="AI News", url="https://example.com", source_id="test",
            full_text="Full article text", relevance_score=8,
        )
        mock_poll.return_value = [item]
        mock_ingest.return_value = [item]
        from nexus.engine.filtering.filter import FilterResult
        mock_filter.return_value = FilterResult(accepted=[item], log_entries=[])
        mock_extract.return_value = Event(
            date=date(2026, 3, 9), summary="AI event", significance=8,
        )
        mock_compress.return_value = None
        mock_synth_topic.return_value = TopicSynthesis(
            topic_name="AI Research",
            threads=[NarrativeThread(headline="AI event", significance=8)],
        )
        mock_render.return_value = "# Daily Briefing\n\n## AI Research\n\nContent here."
        mock_audio.return_value = None

        await run_pipeline(
            config, mock_llm, data_dir,
            gemini_api_key="gem-key",
            openai_api_key="oai-key",
            elevenlabs_api_key="el-key",
        )

        # Verify audio pipeline received all three keys
        call_kwargs = mock_audio.call_args.kwargs
        assert call_kwargs["gemini_api_key"] == "gem-key"
        assert call_kwargs["openai_api_key"] == "oai-key"
        assert call_kwargs["elevenlabs_api_key"] == "el-key"
