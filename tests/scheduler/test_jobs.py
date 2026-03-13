"""Tests for scheduled jobs."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

from nexus.config.models import NexusConfig, UserConfig, BreakingNewsConfig
from nexus.scheduler.jobs import (
    daily_pipeline_job, breaking_news_job, schedule_jobs,
)


@pytest.fixture
def config():
    return NexusConfig(
        user=UserConfig(name="Tristan", timezone="America/Denver"),
    )


async def test_daily_pipeline_job_success(config, tmp_path):
    """Daily pipeline job should call run_pipeline."""
    briefing_path = tmp_path / "artifacts" / "briefings" / "2026-03-10.md"
    briefing_path.parent.mkdir(parents=True)
    briefing_path.write_text("Test briefing")

    with patch("nexus.engine.pipeline.run_pipeline", new_callable=AsyncMock) as mock_pipeline:
        mock_pipeline.return_value = briefing_path
        # Need to patch where it's looked up inside the function
        import nexus.scheduler.jobs as jobs_mod
        with patch.object(jobs_mod, "__builtins__", jobs_mod.__builtins__):
            # The function does `from nexus.engine.pipeline import run_pipeline`
            # so we mock at the source
            await daily_pipeline_job(
                config, AsyncMock(), tmp_path, AsyncMock(),
            )
        mock_pipeline.assert_called_once()


async def test_daily_pipeline_job_error(config, tmp_path):
    """Pipeline failure should be caught and logged, not raised."""
    with patch("nexus.engine.pipeline.run_pipeline", new_callable=AsyncMock) as mock_pipeline:
        mock_pipeline.side_effect = Exception("LLM timeout")
        # Should not raise
        await daily_pipeline_job(config, AsyncMock(), tmp_path, AsyncMock())


async def test_breaking_news_job_success(config):
    with patch("nexus.agent.breaking.check_breaking_news", new_callable=AsyncMock) as mock_check:
        mock_check.return_value = {}
        await breaking_news_job(config, AsyncMock(), AsyncMock())
        mock_check.assert_called_once()


async def test_breaking_news_job_disabled():
    config = NexusConfig(
        user=UserConfig(name="Tristan"),
        breaking_news=BreakingNewsConfig(enabled=False),
    )
    with patch("nexus.agent.breaking.check_breaking_news", new_callable=AsyncMock) as mock_check:
        mock_check.return_value = {}
        await breaking_news_job(config, AsyncMock(), AsyncMock())


def test_schedule_jobs_registers():
    """schedule_jobs should add jobs to the scheduler."""
    config = NexusConfig(
        user=UserConfig(name="Tristan", timezone="America/Denver"),
        breaking_news=BreakingNewsConfig(enabled=True, poll_interval_hours=3),
    )
    scheduler = MagicMock()

    schedule_jobs(scheduler, config, MagicMock(), Path("/tmp"), MagicMock())

    # Should have 2 add_job calls: daily pipeline + breaking news
    assert scheduler.add_job.call_count == 2
    call_ids = [c.kwargs["id"] for c in scheduler.add_job.call_args_list]
    assert "daily_pipeline" in call_ids
    assert "breaking_news" in call_ids


def test_schedule_jobs_no_breaking():
    """When breaking_news is disabled, only 1 job should be registered."""
    config = NexusConfig(
        user=UserConfig(name="Tristan"),
        breaking_news=BreakingNewsConfig(enabled=False),
    )
    scheduler = MagicMock()

    schedule_jobs(scheduler, config, MagicMock(), Path("/tmp"), MagicMock())

    assert scheduler.add_job.call_count == 1


def test_schedule_jobs_passes_all_api_keys():
    """schedule_jobs should forward openai and elevenlabs keys to the daily pipeline job."""
    config = NexusConfig(
        user=UserConfig(name="Tristan", timezone="America/Denver"),
        breaking_news=BreakingNewsConfig(enabled=False),
    )
    scheduler = MagicMock()

    schedule_jobs(
        scheduler, config, MagicMock(), Path("/tmp"), MagicMock(),
        gemini_api_key="gem-key",
        openai_api_key="oai-key",
        elevenlabs_api_key="el-key",
    )

    # Check the daily_pipeline job kwargs include all three keys
    daily_call = scheduler.add_job.call_args_list[0]
    kwargs = daily_call.kwargs.get("kwargs", {})
    assert kwargs["gemini_api_key"] == "gem-key"
    assert kwargs["openai_api_key"] == "oai-key"
    assert kwargs["elevenlabs_api_key"] == "el-key"


async def test_daily_pipeline_job_passes_api_keys(config, tmp_path):
    """daily_pipeline_job should forward all API keys to run_pipeline."""
    briefing_path = tmp_path / "artifacts" / "briefings" / "2026-03-10.md"
    briefing_path.parent.mkdir(parents=True)
    briefing_path.write_text("Test briefing")

    with patch("nexus.engine.pipeline.run_pipeline", new_callable=AsyncMock) as mock_pipeline:
        mock_pipeline.return_value = briefing_path
        await daily_pipeline_job(
            config, AsyncMock(), tmp_path, AsyncMock(),
            gemini_api_key="gem-key",
            openai_api_key="oai-key",
            elevenlabs_api_key="el-key",
        )
        call_kwargs = mock_pipeline.call_args.kwargs
        assert call_kwargs["gemini_api_key"] == "gem-key"
        assert call_kwargs["openai_api_key"] == "oai-key"
        assert call_kwargs["elevenlabs_api_key"] == "el-key"


async def test_daily_pipeline_job_forwards_max_ingest(config, tmp_path):
    """daily_pipeline_job should forward max_ingest to run_pipeline."""
    briefing_path = tmp_path / "artifacts" / "briefings" / "2026-03-10.md"
    briefing_path.parent.mkdir(parents=True)
    briefing_path.write_text("Test briefing")

    with patch("nexus.engine.pipeline.run_pipeline", new_callable=AsyncMock) as mock_pipeline:
        mock_pipeline.return_value = briefing_path
        await daily_pipeline_job(
            config, AsyncMock(), tmp_path, AsyncMock(),
            max_ingest=20,
        )
        call_kwargs = mock_pipeline.call_args.kwargs
        assert call_kwargs["max_ingest"] == 20
