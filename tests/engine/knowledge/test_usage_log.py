"""Tests for usage_log schema migration and KnowledgeStore methods."""

import pytest
from nexus.engine.knowledge.store import KnowledgeStore


@pytest.fixture
async def store(tmp_path):
    s = KnowledgeStore(tmp_path / "knowledge.db")
    await s.initialize()
    yield s
    await s.close()


async def test_add_usage_record(store):
    """Insert a usage record and verify it persists."""
    await store.add_usage_record(
        date="2026-03-11",
        provider="gemini",
        model="gemini-3-flash-preview",
        config_key="filtering",
        input_tokens=1000,
        output_tokens=500,
        cost_usd=0.0003,
    )

    # Verify via raw SQL
    cursor = await store.db.execute("SELECT COUNT(*) FROM usage_log")
    row = await cursor.fetchone()
    assert row[0] == 1


async def test_get_usage_summary(store):
    """Aggregate usage by date."""
    await store.add_usage_record(
        date="2026-03-10", provider="gemini", model="gemini-3-flash-preview",
        config_key="filtering", input_tokens=1000, output_tokens=500, cost_usd=0.10,
    )
    await store.add_usage_record(
        date="2026-03-10", provider="deepseek", model="deepseek-chat",
        config_key="synthesis", input_tokens=2000, output_tokens=1000, cost_usd=0.20,
    )
    await store.add_usage_record(
        date="2026-03-11", provider="gemini", model="gemini-3-flash-preview",
        config_key="filtering", input_tokens=500, output_tokens=200, cost_usd=0.05,
    )

    summary = await store.get_usage_summary()
    assert len(summary) == 2  # Two distinct dates

    # Find date entries
    day10 = next(s for s in summary if s["date"] == "2026-03-10")
    day11 = next(s for s in summary if s["date"] == "2026-03-11")

    assert day10["total_cost_usd"] == pytest.approx(0.30)
    assert day10["total_input_tokens"] == 3000
    assert day10["total_output_tokens"] == 1500

    assert day11["total_cost_usd"] == pytest.approx(0.05)

    # Test with since_date filter
    filtered = await store.get_usage_summary(since_date="2026-03-11")
    assert len(filtered) == 1
    assert filtered[0]["date"] == "2026-03-11"


async def test_get_daily_cost(store):
    """Get total cost for a specific date."""
    await store.add_usage_record(
        date="2026-03-11", provider="gemini", model="gemini-3-flash-preview",
        config_key="filtering", input_tokens=1000, output_tokens=500, cost_usd=0.10,
    )
    await store.add_usage_record(
        date="2026-03-11", provider="deepseek", model="deepseek-chat",
        config_key="synthesis", input_tokens=2000, output_tokens=1000, cost_usd=0.25,
    )

    cost = await store.get_daily_cost("2026-03-11")
    assert cost == pytest.approx(0.35)


async def test_get_daily_cost_no_data(store):
    """No data for a date returns 0.0."""
    cost = await store.get_daily_cost("2026-01-01")
    assert cost == 0.0
