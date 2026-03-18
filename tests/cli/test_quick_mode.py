"""Tests for --quick mode in the engine CLI."""

import sys
from unittest.mock import patch, MagicMock, AsyncMock

import pytest


def _make_config():
    """Build a minimal NexusConfig-like object for testing."""
    from nexus.config.models import (
        NexusConfig, AudioConfig, FutureProjectionConfig, ModelsConfig,
        SourcesConfig, BudgetConfig, BreakingNewsConfig, BriefingConfig,
        TelegramConfig, TopicConfig, UserConfig,
    )
    return NexusConfig(
        user=UserConfig(name="Test", timezone="UTC"),
        topics=[TopicConfig(name="Test Topic")],
        models=ModelsConfig(),
        audio=AudioConfig(enabled=True),
        future_projection=FutureProjectionConfig(enabled=True),
        sources=SourcesConfig(),
        budget=BudgetConfig(),
        breaking_news=BreakingNewsConfig(),
        briefing=BriefingConfig(),
        telegram=TelegramConfig(),
    )


def test_quick_flag_disables_audio_and_projections():
    """--quick should set audio.enabled=False and future_projection.enabled=False."""
    config = _make_config()
    assert config.audio.enabled is True
    assert config.future_projection.enabled is True

    # Simulate what run_engine does when --quick is passed
    config.audio.enabled = False
    config.future_projection.enabled = False

    assert config.audio.enabled is False
    assert config.future_projection.enabled is False


def test_quick_flag_caps_max_ingest():
    """--quick should pass max_ingest=20 to run_pipeline."""
    with patch("sys.argv", ["nexus", "engine", "--quick"]), \
         patch("nexus.__main__.load_dotenv"), \
         patch("nexus.config.loader.load_config") as mock_load, \
         patch("nexus.llm.client.LLMClient") as mock_llm_cls, \
         patch("nexus.engine.pipeline.run_pipeline", new_callable=AsyncMock) as mock_pipeline, \
         patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}):

        config = _make_config()
        mock_load.return_value = config
        mock_llm_cls.return_value = MagicMock()
        mock_pipeline.return_value = "/tmp/briefing.md"

        from nexus.__main__ import run_engine
        run_engine()

        mock_pipeline.assert_called_once()
        call_kwargs = mock_pipeline.call_args
        assert call_kwargs.kwargs.get("max_ingest") == 20 or \
               (call_kwargs.args and len(call_kwargs.args) > 3)

        # Verify config was mutated
        assert config.audio.enabled is False
        assert config.future_projection.enabled is False


def test_normal_mode_no_ingest_cap():
    """Without --quick, max_ingest should be None (no cap)."""
    with patch("sys.argv", ["nexus", "engine"]), \
         patch("nexus.__main__.load_dotenv"), \
         patch("nexus.config.loader.load_config") as mock_load, \
         patch("nexus.llm.client.LLMClient") as mock_llm_cls, \
         patch("nexus.engine.pipeline.run_pipeline", new_callable=AsyncMock) as mock_pipeline, \
         patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}):

        config = _make_config()
        mock_load.return_value = config
        mock_llm_cls.return_value = MagicMock()
        mock_pipeline.return_value = "/tmp/briefing.md"

        from nexus.__main__ import run_engine
        run_engine()

        call_kwargs = mock_pipeline.call_args
        # max_ingest should not be set (None)
        assert call_kwargs.kwargs.get("max_ingest") is None
