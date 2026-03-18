"""Tests for scheduled jobs."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

from nexus.config.models import (
    FutureProjectionConfig, NexusConfig, UserConfig, TopicConfig,
    BreakingNewsConfig, SourcesConfig,
)
from nexus.scheduler.jobs import (
    daily_pipeline_job, daily_prediction_job, breaking_news_job,
    source_rediscovery_job, schedule_jobs,
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

    mock_store = AsyncMock()
    mock_store.is_pipeline_running.return_value = False

    with patch("nexus.engine.pipeline.run_pipeline", new_callable=AsyncMock) as mock_pipeline:
        mock_pipeline.return_value = briefing_path
        # Need to patch where it's looked up inside the function
        import nexus.scheduler.jobs as jobs_mod
        with patch.object(jobs_mod, "__builtins__", jobs_mod.__builtins__):
            # The function does `from nexus.engine.pipeline import run_pipeline`
            # so we mock at the source
            await daily_pipeline_job(
                config, AsyncMock(), tmp_path, mock_store,
            )
        mock_pipeline.assert_called_once()


async def test_daily_pipeline_job_error(config, tmp_path):
    """Pipeline failure should be caught and logged, not raised."""
    mock_store = AsyncMock()
    mock_store.is_pipeline_running.return_value = False

    with patch("nexus.engine.pipeline.run_pipeline", new_callable=AsyncMock) as mock_pipeline:
        mock_pipeline.side_effect = Exception("LLM timeout")
        # Should not raise
        await daily_pipeline_job(config, AsyncMock(), tmp_path, mock_store)


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

    mock_store = AsyncMock()
    mock_store.is_pipeline_running.return_value = False

    with patch("nexus.engine.pipeline.run_pipeline", new_callable=AsyncMock) as mock_pipeline:
        mock_pipeline.return_value = briefing_path
        await daily_pipeline_job(
            config, AsyncMock(), tmp_path, mock_store,
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

    mock_store = AsyncMock()
    mock_store.is_pipeline_running.return_value = False

    with patch("nexus.engine.pipeline.run_pipeline", new_callable=AsyncMock) as mock_pipeline:
        mock_pipeline.return_value = briefing_path
        await daily_pipeline_job(
            config, AsyncMock(), tmp_path, mock_store,
            max_ingest=20,
        )
        call_kwargs = mock_pipeline.call_args.kwargs
        assert call_kwargs["max_ingest"] == 20


def test_schedule_jobs_with_predictions():
    """When future_projection is enabled, prediction job should be registered."""
    config = NexusConfig(
        user=UserConfig(name="Tristan", timezone="America/Denver"),
        breaking_news=BreakingNewsConfig(enabled=False),
        future_projection=FutureProjectionConfig(enabled=True),
    )
    scheduler = MagicMock()

    schedule_jobs(scheduler, config, MagicMock(), Path("/tmp"), MagicMock())

    # Should have 2 jobs: daily pipeline + daily predictions
    assert scheduler.add_job.call_count == 2
    call_ids = [c.kwargs["id"] for c in scheduler.add_job.call_args_list]
    assert "daily_pipeline" in call_ids
    assert "daily_predictions" in call_ids


def test_schedule_jobs_prediction_offset():
    """Prediction job should be offset from pipeline time."""
    config = NexusConfig(
        user=UserConfig(name="Tristan", timezone="America/Denver"),
        future_projection=FutureProjectionConfig(
            enabled=True, prediction_schedule_offset_minutes=45,
        ),
        breaking_news=BreakingNewsConfig(enabled=False),
    )
    config.briefing.schedule = "06:00"
    scheduler = MagicMock()

    schedule_jobs(scheduler, config, MagicMock(), Path("/tmp"), MagicMock())

    # Find the prediction job call
    pred_call = None
    for call in scheduler.add_job.call_args_list:
        if call.kwargs.get("id") == "daily_predictions":
            pred_call = call
            break
    assert pred_call is not None
    # 06:00 + 45min = 06:45
    assert pred_call.kwargs["hour"] == 6
    assert pred_call.kwargs["minute"] == 45


async def test_daily_prediction_job_kg_native(config, tmp_path):
    """Daily prediction job should call KG-native predictions."""
    config.future_projection = FutureProjectionConfig(
        enabled=True, kg_native_enabled=True,
    )
    mock_store = AsyncMock()

    with patch(
        "nexus.engine.projection.service.generate_kg_native_predictions",
        new_callable=AsyncMock,
    ) as mock_kg:
        mock_kg.return_value = {"total_generated": 3, "topics": []}
        await daily_prediction_job(config, AsyncMock(), mock_store, tmp_path)
        mock_kg.assert_called_once()


# ── Source re-discovery ──


async def test_source_rediscovery_merges_new_feeds(tmp_path):
    """Re-discovery should merge new feeds into existing registry without overwriting."""
    import yaml

    config = NexusConfig(
        user=UserConfig(name="Test"),
        topics=[TopicConfig(name="AI Research", subtopics=["agents"])],
        sources=SourcesConfig(discover_new_sources=True, discovery_interval_days=7),
    )
    data_dir = tmp_path / "data"

    # Pre-existing registry with one feed
    slug = "ai-research"
    reg_dir = data_dir / "sources" / slug
    reg_dir.mkdir(parents=True)
    existing = {"sources": [
        {"url": "https://existing.com/feed", "id": "existing-feed", "name": "Existing"},
    ]}
    (reg_dir / "registry.yaml").write_text(yaml.dump(existing))

    # Discovery returns a new feed
    from dataclasses import dataclass, field as dfield

    @dataclass
    class FakeResult:
        feeds: list = dfield(default_factory=list)

    with patch(
        "nexus.engine.sources.discovery.discover_sources",
        new_callable=AsyncMock,
    ) as mock_discover:
        mock_discover.return_value = FakeResult(feeds=[
            {"url": "https://new.com/feed", "id": "new-feed", "name": "New Source"},
        ])

        await source_rediscovery_job(config, AsyncMock(), data_dir)

    # Registry should now contain both feeds
    reg = yaml.safe_load((reg_dir / "registry.yaml").read_text())
    urls = {s["url"] for s in reg["sources"]}
    assert "https://existing.com/feed" in urls
    assert "https://new.com/feed" in urls
    assert len(reg["sources"]) == 2

    # Discovery was called with existing URLs to avoid re-discovering them
    call_kwargs = mock_discover.call_args.kwargs
    assert "https://existing.com/feed" in call_kwargs["existing_urls"]


async def test_source_rediscovery_skips_duplicates(tmp_path):
    """Re-discovery should not add feeds that already exist in the registry."""
    import yaml

    config = NexusConfig(
        user=UserConfig(name="Test"),
        topics=[TopicConfig(name="AI Research")],
        sources=SourcesConfig(discover_new_sources=True),
    )
    data_dir = tmp_path / "data"

    slug = "ai-research"
    reg_dir = data_dir / "sources" / slug
    reg_dir.mkdir(parents=True)
    existing = {"sources": [
        {"url": "https://feed.com/rss", "id": "feed1", "name": "Feed One"},
    ]}
    (reg_dir / "registry.yaml").write_text(yaml.dump(existing))

    from dataclasses import dataclass, field as dfield

    @dataclass
    class FakeResult:
        feeds: list = dfield(default_factory=list)

    with patch(
        "nexus.engine.sources.discovery.discover_sources",
        new_callable=AsyncMock,
    ) as mock_discover:
        # Discovery returns same URL as existing
        mock_discover.return_value = FakeResult(feeds=[
            {"url": "https://feed.com/rss", "id": "feed1-dup", "name": "Duplicate"},
        ])

        await source_rediscovery_job(config, AsyncMock(), data_dir)

    # Should still have exactly 1 feed
    reg = yaml.safe_load((reg_dir / "registry.yaml").read_text())
    assert len(reg["sources"]) == 1


async def test_source_rediscovery_handles_empty_registry(tmp_path):
    """Re-discovery should work when no registry exists yet."""
    import yaml

    config = NexusConfig(
        user=UserConfig(name="Test"),
        topics=[TopicConfig(name="Space Exploration")],
        sources=SourcesConfig(discover_new_sources=True),
    )
    data_dir = tmp_path / "data"

    from dataclasses import dataclass, field as dfield

    @dataclass
    class FakeResult:
        feeds: list = dfield(default_factory=list)

    with patch(
        "nexus.engine.sources.discovery.discover_sources",
        new_callable=AsyncMock,
    ) as mock_discover:
        mock_discover.return_value = FakeResult(feeds=[
            {"url": "https://space.com/feed", "id": "space", "name": "Space.com"},
        ])

        await source_rediscovery_job(config, AsyncMock(), data_dir)

    slug = "space-exploration"
    reg = yaml.safe_load((data_dir / "sources" / slug / "registry.yaml").read_text())
    assert len(reg["sources"]) == 1
    assert reg["sources"][0]["url"] == "https://space.com/feed"


async def test_source_rediscovery_error_doesnt_clobber_registry(tmp_path):
    """If discovery fails, existing registry should be untouched."""
    import yaml

    config = NexusConfig(
        user=UserConfig(name="Test"),
        topics=[TopicConfig(name="AI Research")],
        sources=SourcesConfig(discover_new_sources=True),
    )
    data_dir = tmp_path / "data"

    slug = "ai-research"
    reg_dir = data_dir / "sources" / slug
    reg_dir.mkdir(parents=True)
    existing = {"sources": [
        {"url": "https://existing.com/feed", "id": "existing", "name": "Existing"},
    ]}
    (reg_dir / "registry.yaml").write_text(yaml.dump(existing))

    with patch(
        "nexus.engine.sources.discovery.discover_sources",
        new_callable=AsyncMock,
        side_effect=RuntimeError("LLM down"),
    ):
        # Should not raise
        await source_rediscovery_job(config, AsyncMock(), data_dir)

    # Registry unchanged
    reg = yaml.safe_load((reg_dir / "registry.yaml").read_text())
    assert len(reg["sources"]) == 1


def test_schedule_jobs_with_rediscovery():
    """When discovery is enabled, re-discovery job should be scheduled."""
    config = NexusConfig(
        user=UserConfig(name="Tristan", timezone="America/Denver"),
        topics=[TopicConfig(name="AI Research")],
        sources=SourcesConfig(discover_new_sources=True, discovery_interval_days=7),
        breaking_news=BreakingNewsConfig(enabled=False),
    )
    scheduler = MagicMock()

    schedule_jobs(scheduler, config, MagicMock(), Path("/tmp"), MagicMock())

    call_ids = [c.kwargs["id"] for c in scheduler.add_job.call_args_list]
    assert "source_rediscovery" in call_ids


def test_schedule_jobs_no_rediscovery_when_disabled():
    """When discovery is disabled, re-discovery job should NOT be scheduled."""
    config = NexusConfig(
        user=UserConfig(name="Tristan", timezone="America/Denver"),
        sources=SourcesConfig(discover_new_sources=False),
        breaking_news=BreakingNewsConfig(enabled=False),
    )
    scheduler = MagicMock()

    schedule_jobs(scheduler, config, MagicMock(), Path("/tmp"), MagicMock())

    call_ids = [c.kwargs["id"] for c in scheduler.add_job.call_args_list]
    assert "source_rediscovery" not in call_ids
