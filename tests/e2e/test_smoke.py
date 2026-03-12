"""E2E smoke tests — hit real APIs with minimal volume.

Run with: .venv/bin/pytest tests/e2e/ -m e2e -v
Requires API keys in .env or environment.
"""

import pytest
from nexus.testing.smoke import SmokeTestConfig, run_smoke_test


pytestmark = [pytest.mark.e2e, pytest.mark.asyncio]


async def test_pipeline_produces_events(smoke_data_dir):
    """Full pipeline with 1 topic, ~20 articles, real LLM."""
    config = SmokeTestConfig(
        topic_name="AI/ML Research",
        max_articles=20,
        max_feeds=5,
        data_dir=smoke_data_dir,
    )
    result = await run_smoke_test(config)

    assert result.events_found > 0, f"No events extracted. Errors: {result.errors}"


async def test_briefing_is_nonempty(smoke_data_dir):
    """Briefing text is generated and substantial."""
    config = SmokeTestConfig(
        topic_name="AI/ML Research",
        max_articles=20,
        max_feeds=5,
        data_dir=smoke_data_dir,
    )
    result = await run_smoke_test(config)

    assert result.briefing_chars > 100, f"Briefing too short: {result.briefing_chars} chars"


async def test_cost_tracking_works(smoke_data_dir):
    """UsageTracker has calls and cost > 0 after pipeline run."""
    config = SmokeTestConfig(
        topic_name="AI/ML Research",
        max_articles=20,
        max_feeds=5,
        data_dir=smoke_data_dir,
    )
    result = await run_smoke_test(config)

    cost_check = next((c for c in result.checks if c.name == "cost_tracked"), None)
    assert cost_check is not None and cost_check.passed, f"Cost not tracked: {result.checks}"


async def test_cost_persisted_to_sqlite(smoke_data_dir):
    """Usage records actually written to SQLite store."""
    config = SmokeTestConfig(
        topic_name="AI/ML Research",
        max_articles=20,
        max_feeds=5,
        data_dir=smoke_data_dir,
    )
    result = await run_smoke_test(config)

    persist_check = next((c for c in result.checks if c.name == "cost_persisted"), None)
    assert persist_check is not None and persist_check.passed, f"Cost not persisted: {result.checks}"
