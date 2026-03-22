"""Tests for the engine pipeline orchestrator."""

import pytest
from datetime import date
from unittest.mock import AsyncMock, patch
from nexus.config.models import NexusConfig, UserConfig, TopicConfig, SourcesConfig
from nexus.engine.pipeline import run_pipeline, run_backfill, _event_cap_for_topic
from nexus.llm.client import UsageTracker


class _FakeLLMResponse:
    def __init__(self, text: str):
        self.text = text


class FakePipelineLLM:
    """Minimal async LLM stub for pipeline tests."""

    def __init__(self):
        self.usage = UsageTracker()
        self._store = None

    async def set_store(self, store) -> None:
        self._store = store

    async def flush_usage(self) -> None:
        return None

    async def generate(self, *args, **kwargs):
        return _FakeLLMResponse("- Highlight")

    async def complete(self, *args, **kwargs):
        if kwargs.get("json_response"):
            return "[]"
        return "Stub content"


@pytest.fixture
def config():
    cfg = NexusConfig(
        user=UserConfig(name="Tristan", timezone="America/Denver"),
        topics=[
            TopicConfig(name="AI Research", priority="high", subtopics=["agents"]),
        ],
    )
    cfg.breaking_news.enabled = False
    return cfg


@pytest.fixture
def data_dir(tmp_path):
    return tmp_path / "data"


@pytest.mark.asyncio
async def test_pipeline_produces_briefing(config, data_dir):
    mock_llm = FakePipelineLLM()

    with patch("nexus.engine.pipeline.load_source_registry") as mock_registry, \
         patch("nexus.engine.pipeline.poll_all_feeds") as mock_poll, \
         patch("nexus.engine.pipeline.async_ingest_items") as mock_ingest, \
         patch("nexus.engine.pipeline.filter_items") as mock_filter, \
         patch("nexus.engine.pipeline.extract_event") as mock_extract, \
         patch("nexus.engine.pipeline.synthesize_topic") as mock_synth_topic, \
         patch("nexus.engine.pipeline.render_text_briefing") as mock_render, \
         patch("nexus.engine.pipeline.maybe_compress") as mock_compress, \
         patch("nexus.engine.pipeline.run_audio_pipeline", new_callable=AsyncMock) as mock_audio:

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

        briefing_path = await run_pipeline(config, mock_llm, data_dir)

        assert briefing_path.exists()
        assert "AI Research" in briefing_path.read_text()
        mock_poll.assert_called_once()
        mock_render.assert_called_once()


@pytest.mark.asyncio
async def test_pipeline_marks_stale_threads_before_projection(config, data_dir):
    mock_llm = FakePipelineLLM()

    with patch("nexus.engine.pipeline.load_source_registry") as mock_registry, \
         patch("nexus.engine.pipeline.poll_all_feeds") as mock_poll, \
         patch("nexus.engine.pipeline.async_ingest_items") as mock_ingest, \
         patch("nexus.engine.pipeline.filter_items") as mock_filter, \
         patch("nexus.engine.pipeline.extract_event") as mock_extract, \
         patch("nexus.engine.pipeline.synthesize_topic") as mock_synth_topic, \
         patch("nexus.engine.pipeline.render_text_briefing") as mock_render, \
         patch("nexus.engine.pipeline.maybe_compress") as mock_compress, \
         patch("nexus.engine.pipeline.run_audio_pipeline", new_callable=AsyncMock) as mock_audio, \
         patch("nexus.engine.pipeline.run_projection_pass", new_callable=AsyncMock) as mock_projection, \
         patch("nexus.engine.pipeline.KnowledgeStore.mark_stale_threads", new_callable=AsyncMock) as mock_mark_stale:

        from nexus.engine.sources.polling import ContentItem
        from nexus.engine.knowledge.events import Event
        from nexus.engine.synthesis.knowledge import TopicSynthesis, NarrativeThread
        from nexus.engine.filtering.filter import FilterResult

        mock_registry.return_value = [{"url": "https://feed.com/rss", "id": "test"}]
        item = ContentItem(
            title="AI News", url="https://example.com", source_id="test",
            full_text="Full article text", relevance_score=8,
        )
        mock_poll.return_value = [item]
        mock_ingest.return_value = [item]
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
        mock_mark_stale.return_value = 1

        await run_pipeline(config, mock_llm, data_dir)

        mock_mark_stale.assert_awaited_once()
        assert mock_projection.await_count == 1


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
    mock_llm = FakePipelineLLM()

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


# ── Auto-discovery fallback ──


@pytest.mark.asyncio
async def test_run_pipeline_auto_discovers_when_no_sources(data_dir):
    """run_pipeline should call discover_sources when registry is empty and discovery enabled."""
    config = NexusConfig(
        user=UserConfig(name="Test"),
        topics=[TopicConfig(name="Space Exploration", subtopics=["rockets"])],
        sources=SourcesConfig(discover_new_sources=True),
    )
    config.breaking_news.enabled = False
    mock_llm = FakePipelineLLM()

    with patch("nexus.engine.pipeline.load_source_registry") as mock_registry, \
         patch("nexus.engine.sources.discovery.discover_sources", new_callable=AsyncMock) as mock_discover, \
         patch("nexus.engine.pipeline.poll_all_feeds") as mock_poll, \
         patch("nexus.engine.pipeline.async_ingest_items") as mock_ingest, \
         patch("nexus.engine.pipeline.filter_items") as mock_filter, \
         patch("nexus.engine.pipeline.extract_event") as mock_extract, \
         patch("nexus.engine.pipeline.synthesize_topic") as mock_synth_topic, \
         patch("nexus.engine.pipeline.render_text_briefing") as mock_render, \
         patch("nexus.engine.pipeline.maybe_compress"), \
         patch("nexus.engine.pipeline.run_audio_pipeline", new_callable=AsyncMock) as mock_audio:

        from nexus.engine.sources.polling import ContentItem
        from nexus.engine.knowledge.events import Event
        from nexus.engine.synthesis.knowledge import TopicSynthesis, NarrativeThread

        # Empty registry → triggers discovery
        mock_registry.return_value = []

        # Discovery returns feeds
        from dataclasses import dataclass, field
        @dataclass
        class FakeDiscoveryResult:
            feeds: list = field(default_factory=list)
        mock_discover.return_value = FakeDiscoveryResult(
            feeds=[{"url": "https://space.com/rss", "id": "space-rss", "name": "Space.com"}]
        )

        item = ContentItem(
            title="Rocket launch", url="https://example.com", source_id="space-rss",
            full_text="Full text", relevance_score=8,
        )
        mock_poll.return_value = [item]
        mock_ingest.return_value = [item]
        from nexus.engine.filtering.filter import FilterResult
        mock_filter.return_value = FilterResult(accepted=[item], log_entries=[])
        mock_extract.return_value = Event(
            date=date.today(), summary="Rocket launch", significance=8,
        )
        mock_synth_topic.return_value = TopicSynthesis(
            topic_name="Space Exploration",
            threads=[NarrativeThread(headline="Rocket launch", significance=8)],
        )
        mock_render.return_value = "# Briefing\n\nSpace content."
        mock_audio.return_value = None

        briefing_path = await run_pipeline(config, mock_llm, data_dir)
        assert briefing_path.exists()
        mock_discover.assert_called_once()

        # Verify registry was saved to disk
        slug = "space-exploration"
        reg_path = data_dir / "sources" / slug / "registry.yaml"
        assert reg_path.exists()


@pytest.mark.asyncio
async def test_run_pipeline_skips_topic_when_no_sources_and_discovery_disabled(data_dir):
    """run_pipeline should skip topic when registry is empty and discovery disabled."""
    config = NexusConfig(
        user=UserConfig(name="Test"),
        topics=[TopicConfig(name="Space Exploration")],
        sources=SourcesConfig(discover_new_sources=False),
    )
    config.breaking_news.enabled = False
    mock_llm = FakePipelineLLM()

    with patch("nexus.engine.pipeline.load_source_registry") as mock_registry, \
         patch("nexus.engine.pipeline.run_topic_pipeline", new_callable=AsyncMock) as mock_topic, \
         patch("nexus.engine.pipeline.render_text_briefing") as mock_render, \
         patch("nexus.engine.pipeline.run_audio_pipeline", new_callable=AsyncMock) as mock_audio:

        mock_registry.return_value = []
        mock_render.return_value = "# Empty briefing"
        mock_audio.return_value = None

        await run_pipeline(config, mock_llm, data_dir)

        # run_topic_pipeline should NOT be called since sources are empty
        mock_topic.assert_not_called()


# ── max_age_hours pass-through ──


@pytest.mark.asyncio
async def test_run_topic_pipeline_uses_custom_max_age_hours(config, data_dir):
    """run_topic_pipeline should pass max_age_hours to filter_recent."""
    mock_llm = FakePipelineLLM()

    with patch("nexus.engine.pipeline.poll_all_feeds") as mock_poll, \
         patch("nexus.engine.pipeline.filter_recent") as mock_filter_recent, \
         patch("nexus.engine.pipeline.dedup_items") as mock_dedup, \
         patch("nexus.engine.pipeline.async_ingest_items") as mock_ingest, \
         patch("nexus.engine.pipeline.filter_items") as mock_filter, \
         patch("nexus.engine.pipeline.extract_event"), \
         patch("nexus.engine.pipeline.synthesize_topic") as mock_synth, \
         patch("nexus.engine.pipeline.maybe_compress"):

        from nexus.engine.sources.polling import ContentItem
        from nexus.engine.synthesis.knowledge import TopicSynthesis, NarrativeThread
        from nexus.engine.pipeline import run_topic_pipeline
        from nexus.engine.knowledge.store import KnowledgeStore

        item = ContentItem(
            title="Old News", url="https://example.com", source_id="test",
            full_text="Text", relevance_score=8,
        )
        mock_poll.return_value = [item]
        mock_filter_recent.return_value = [item]
        mock_dedup.return_value = [item]
        mock_ingest.return_value = [item]
        from nexus.engine.filtering.filter import FilterResult
        mock_filter.return_value = FilterResult(accepted=[], log_entries=[])
        mock_synth.return_value = TopicSynthesis(
            topic_name="AI Research",
            threads=[NarrativeThread(headline="Stub", significance=5)],
        )

        store = KnowledgeStore(data_dir / "knowledge.db")
        await store.initialize()

        topic = config.topics[0]
        sources = [{"url": "https://feed.com/rss", "id": "test"}]

        # Default: 48h
        await run_topic_pipeline(
            mock_llm, topic, data_dir, sources, store=store,
        )
        mock_filter_recent.assert_called_with(mock_poll.return_value, max_age_hours=48)

        # Custom: 168h (backfill)
        mock_filter_recent.reset_mock()
        await run_topic_pipeline(
            mock_llm, topic, data_dir, sources, store=store,
            max_age_hours=168,
        )
        mock_filter_recent.assert_called_with(mock_poll.return_value, max_age_hours=168)

        await store.close()


@pytest.mark.asyncio
async def test_run_pipeline_forwards_max_age_hours(config, data_dir):
    """run_pipeline should forward max_age_hours to run_topic_pipeline."""
    mock_llm = FakePipelineLLM()

    with patch("nexus.engine.pipeline.load_source_registry") as mock_registry, \
         patch("nexus.engine.pipeline.run_topic_pipeline", new_callable=AsyncMock) as mock_topic, \
         patch("nexus.engine.pipeline.render_text_briefing") as mock_render, \
         patch("nexus.engine.pipeline.run_audio_pipeline", new_callable=AsyncMock) as mock_audio:

        from nexus.engine.synthesis.knowledge import TopicSynthesis, NarrativeThread

        mock_registry.return_value = [{"url": "https://feed.com/rss", "id": "test"}]
        mock_topic.return_value = type("R", (), {
            "synthesis": TopicSynthesis(
                topic_name="AI Research",
                threads=[NarrativeThread(headline="Stub", significance=5)],
            ),
            "articles": [],
            "extracted_event_count": 0,
        })()
        mock_render.return_value = "# Briefing"
        mock_audio.return_value = None

        await run_pipeline(config, mock_llm, data_dir, max_age_hours=168)

        # Verify max_age_hours was passed through
        call_kwargs = mock_topic.call_args.kwargs
        assert call_kwargs["max_age_hours"] == 168


# ── run_backfill ──


@pytest.mark.asyncio
async def test_run_backfill_uses_extended_window(config, data_dir):
    """run_backfill should call run_topic_pipeline with 168h window and no synthesis output."""
    mock_llm = FakePipelineLLM()

    with patch("nexus.engine.pipeline.load_source_registry") as mock_registry, \
         patch("nexus.engine.pipeline.run_topic_pipeline", new_callable=AsyncMock) as mock_topic:

        from nexus.engine.synthesis.knowledge import TopicSynthesis, NarrativeThread

        mock_registry.return_value = [{"url": "https://feed.com/rss", "id": "test"}]
        mock_topic.return_value = type("R", (), {
            "synthesis": TopicSynthesis(
                topic_name="AI Research",
                threads=[NarrativeThread(headline="Stub", significance=5)],
            ),
            "articles": [],
            "extracted_event_count": 7,
        })()

        total = await run_backfill(config, mock_llm, data_dir, max_age_hours=168)

        assert total == 7
        call_kwargs = mock_topic.call_args.kwargs
        assert call_kwargs["max_age_hours"] == 168


@pytest.mark.asyncio
async def test_run_backfill_skips_topics_without_sources(data_dir):
    """run_backfill should skip topics that have no source registry."""
    config = NexusConfig(
        user=UserConfig(name="Test"),
        topics=[TopicConfig(name="No Sources Topic")],
    )
    mock_llm = FakePipelineLLM()

    with patch("nexus.engine.pipeline.load_source_registry") as mock_registry, \
         patch("nexus.engine.pipeline.run_topic_pipeline", new_callable=AsyncMock) as mock_topic:

        mock_registry.return_value = []  # No sources

        total = await run_backfill(config, mock_llm, data_dir)

        assert total == 0
        mock_topic.assert_not_called()


@pytest.mark.asyncio
async def test_run_backfill_continues_on_topic_failure(data_dir):
    """run_backfill should continue to next topic if one fails."""
    config = NexusConfig(
        user=UserConfig(name="Test"),
        topics=[
            TopicConfig(name="Topic A"),
            TopicConfig(name="Topic B"),
        ],
    )
    mock_llm = FakePipelineLLM()

    with patch("nexus.engine.pipeline.load_source_registry") as mock_registry, \
         patch("nexus.engine.pipeline.run_topic_pipeline", new_callable=AsyncMock) as mock_topic:

        from nexus.engine.synthesis.knowledge import TopicSynthesis, NarrativeThread

        mock_registry.return_value = [{"url": "https://feed.com/rss", "id": "test"}]

        # First topic fails, second succeeds
        mock_topic.side_effect = [
            RuntimeError("boom"),
            type("R", (), {
                "synthesis": TopicSynthesis(
                    topic_name="Topic B",
                    threads=[NarrativeThread(headline="Stub", significance=5)],
                ),
                "articles": [],
                "extracted_event_count": 3,
            })(),
        ]

        total = await run_backfill(config, mock_llm, data_dir)

        assert total == 3
        assert mock_topic.call_count == 2
