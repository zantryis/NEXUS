"""Tests for pipeline post-pipeline background tasks (breaking news, Kalshi loop)."""

import pytest
from datetime import date
from unittest.mock import AsyncMock, patch

from nexus.config.models import (
    KalshiBenchmarkConfig,
    NexusConfig,
    TopicConfig,
    UserConfig,
)
from nexus.llm.client import UsageTracker


@pytest.fixture
def config():
    return NexusConfig(
        user=UserConfig(name="Test", timezone="UTC"),
        topics=[TopicConfig(name="AI Research", priority="high")],
    )


def _make_llm_mock():
    mock_llm = AsyncMock()
    mock_llm.usage = UsageTracker()
    return mock_llm


def _pipeline_patches():
    """Context managers for all pipeline dependencies."""
    from nexus.engine.sources.polling import ContentItem
    from nexus.engine.knowledge.events import Event
    from nexus.engine.synthesis.knowledge import TopicSynthesis, NarrativeThread
    from nexus.engine.filtering.filter import FilterResult

    item = ContentItem(
        title="AI News", url="https://example.com", source_id="test",
        full_text="Article", relevance_score=8,
    )
    synth = TopicSynthesis(
        topic_name="AI Research",
        threads=[NarrativeThread(headline="AI event", significance=8)],
    )

    return {
        "registry": [{"url": "https://feed.com/rss", "id": "test"}],
        "items": [item],
        "filter_result": FilterResult(accepted=[item], log_entries=[]),
        "event": Event(date=date(2026, 3, 9), summary="AI", significance=8),
        "synth": synth,
    }


@pytest.mark.asyncio
async def test_post_pipeline_calls_breaking_news(config, tmp_path):
    """Breaking news check should run as a post-pipeline task."""
    mock_llm = _make_llm_mock()
    data = _pipeline_patches()

    with patch("nexus.engine.pipeline.load_source_registry", return_value=data["registry"]), \
         patch("nexus.engine.pipeline.poll_all_feeds", return_value=data["items"]), \
         patch("nexus.engine.pipeline.async_ingest_items", return_value=data["items"]), \
         patch("nexus.engine.pipeline.filter_items", return_value=data["filter_result"]), \
         patch("nexus.engine.pipeline.extract_event", return_value=data["event"]), \
         patch("nexus.engine.pipeline.synthesize_topic", return_value=data["synth"]), \
         patch("nexus.engine.pipeline.render_text_briefing", return_value="# Briefing"), \
         patch("nexus.engine.pipeline.maybe_compress", return_value=None), \
         patch("nexus.engine.pipeline.run_audio_pipeline", return_value=None), \
         patch("nexus.web.thumbnails.populate_thumbnails", new_callable=AsyncMock, return_value={"fetched": 0, "cached": 0, "errors": 0}), \
         patch("nexus.agent.breaking.check_breaking_news", new_callable=AsyncMock) as mock_breaking:

        mock_breaking.return_value = {"ai-research": [{"headline": "Breaking!"}]}

        from nexus.engine.pipeline import run_pipeline
        await run_pipeline(config, mock_llm, tmp_path / "data")

        mock_breaking.assert_called_once()
        args = mock_breaking.call_args[0]
        assert args[0] is mock_llm  # llm
        assert args[1] is config    # config


@pytest.mark.asyncio
async def test_post_pipeline_breaking_failure_non_fatal(config, tmp_path):
    """Breaking news failure should not crash the pipeline."""
    mock_llm = _make_llm_mock()
    data = _pipeline_patches()

    with patch("nexus.engine.pipeline.load_source_registry", return_value=data["registry"]), \
         patch("nexus.engine.pipeline.poll_all_feeds", return_value=data["items"]), \
         patch("nexus.engine.pipeline.async_ingest_items", return_value=data["items"]), \
         patch("nexus.engine.pipeline.filter_items", return_value=data["filter_result"]), \
         patch("nexus.engine.pipeline.extract_event", return_value=data["event"]), \
         patch("nexus.engine.pipeline.synthesize_topic", return_value=data["synth"]), \
         patch("nexus.engine.pipeline.render_text_briefing", return_value="# Briefing"), \
         patch("nexus.engine.pipeline.maybe_compress", return_value=None), \
         patch("nexus.engine.pipeline.run_audio_pipeline", return_value=None), \
         patch("nexus.web.thumbnails.populate_thumbnails", new_callable=AsyncMock, return_value={"fetched": 0, "cached": 0, "errors": 0}), \
         patch("nexus.agent.breaking.check_breaking_news", new_callable=AsyncMock, side_effect=RuntimeError("Wire feeds down")):

        from nexus.engine.pipeline import run_pipeline
        result = await run_pipeline(config, mock_llm, tmp_path / "data")
        assert result.exists()


def test_build_kalshi_client_returns_client():
    """build_kalshi_client should return a KalshiClient instance."""
    from nexus.engine.projection.kalshi import KalshiClient, build_kalshi_client

    cfg = KalshiBenchmarkConfig()
    client = build_kalshi_client(cfg)
    assert isinstance(client, KalshiClient)
    assert client.config is cfg


def test_build_kalshi_client_importable():
    """build_kalshi_client should be importable from kalshi module."""
    from nexus.engine.projection.kalshi import build_kalshi_client
    assert callable(build_kalshi_client)
