"""Tests for cached narrative page generation and staleness."""

import pytest
from datetime import date
from unittest.mock import AsyncMock

from nexus.engine.knowledge.pages import (
    compute_prompt_hash,
    generate_backstory,
    generate_entity_profile,
    generate_thread_deepdive,
    refresh_stale_pages,
    PAGE_CONFIGS,
)
from nexus.engine.knowledge.compression import Summary
from nexus.engine.knowledge.store import KnowledgeStore


@pytest.fixture
async def store(tmp_path):
    s = KnowledgeStore(tmp_path / "knowledge.db")
    await s.initialize()
    yield s
    await s.close()


@pytest.fixture
def mock_llm():
    return AsyncMock()


# ── Prompt hashing ───────────────────────────────────────────────


def test_prompt_hash_deterministic():
    data = {"key": "value", "list": [1, 2, 3]}
    h1 = compute_prompt_hash(data)
    h2 = compute_prompt_hash(data)
    assert h1 == h2
    assert len(h1) == 64  # SHA-256 hex


def test_prompt_hash_changes_on_data_change():
    h1 = compute_prompt_hash({"key": "v1"})
    h2 = compute_prompt_hash({"key": "v2"})
    assert h1 != h2


def test_prompt_hash_order_independent():
    h1 = compute_prompt_hash({"a": 1, "b": 2})
    h2 = compute_prompt_hash({"b": 2, "a": 1})
    assert h1 == h2


# ── Page config ──────────────────────────────────────────────────


def test_page_configs_defined():
    assert "backstory" in PAGE_CONFIGS
    assert "entity_profile" in PAGE_CONFIGS
    assert "thread_deepdive" in PAGE_CONFIGS
    assert "weekly_recap" in PAGE_CONFIGS


def test_page_configs_have_ttl():
    for name, cfg in PAGE_CONFIGS.items():
        assert "ttl_days" in cfg
        assert isinstance(cfg["ttl_days"], int)


# ── Backstory generation ────────────────────────────────────────


async def test_generate_backstory(mock_llm):
    mock_llm.complete.return_value = "# Background\n\nLong history of Iran-US relations..."

    summaries = [
        Summary(period_start=date(2026, 2, 24), period_end=date(2026, 3, 2),
                text="Week 1 summary", event_count=5),
        Summary(period_start=date(2026, 3, 3), period_end=date(2026, 3, 9),
                text="Week 2 summary", event_count=7),
    ]

    result = await generate_backstory(mock_llm, "Iran-US Relations", "iran-us", summaries)
    assert result["title"] == "Background: Iran-US Relations"
    assert result["slug"] == "backstory:iran-us"
    assert result["page_type"] == "backstory"
    assert "history" in result["content_md"].lower() or "background" in result["content_md"].lower()
    assert result["prompt_hash"]  # Non-empty


async def test_generate_backstory_empty_summaries(mock_llm):
    mock_llm.complete.return_value = "# Background\n\nNew topic..."

    result = await generate_backstory(mock_llm, "New Topic", "new-topic", [])
    assert result["content_md"]
    mock_llm.complete.assert_called_once()


# ── Entity profile generation ───────────────────────────────────


async def test_generate_entity_profile(mock_llm):
    mock_llm.complete.return_value = "# IAEA\n\nThe International Atomic Energy Agency..."

    entity = {
        "id": 1,
        "canonical_name": "IAEA",
        "entity_type": "org",
        "aliases": ["International Atomic Energy Agency"],
    }
    events_data = [
        {"summary": "IAEA inspections in Iran", "date": "2026-03-08"},
        {"summary": "IAEA board meeting", "date": "2026-03-09"},
    ]

    result = await generate_entity_profile(mock_llm, entity, events_data)
    assert result["slug"] == "entity:1"
    assert result["page_type"] == "entity_profile"
    assert result["title"] == "Entity Profile: IAEA"
    assert result["prompt_hash"]


# ── Thread deep-dive generation ──────────────────────────────────


async def test_generate_thread_deepdive(mock_llm):
    mock_llm.complete.return_value = "# Sanctions Escalation\n\nDeep analysis..."

    thread = {
        "slug": "sanctions-escalation",
        "headline": "US-Iran Sanctions Escalation",
        "significance": 8,
        "status": "active",
    }
    events_data = [
        {"summary": "New sanctions announced", "date": "2026-03-08"},
        {"summary": "Iran responds to sanctions", "date": "2026-03-09"},
    ]
    convergence = [{"fact": "Sanctions were announced", "confirmed_by": ["nyt", "bbc"]}]
    divergence = []

    result = await generate_thread_deepdive(mock_llm, thread, events_data, convergence, divergence)
    assert result["slug"] == "thread:sanctions-escalation"
    assert result["page_type"] == "thread_deepdive"
    assert "Sanctions Escalation" in result["title"]


# ── Refresh stale pages ─────────────────────────────────────────


async def test_refresh_stale_pages_generates_backstory(store, mock_llm):
    """First run should generate backstory pages for all topics."""
    mock_llm.complete.return_value = "# Background\n\nContent here..."

    # Add a summary so backstory has data
    summary = Summary(
        period_start=date(2026, 3, 3), period_end=date(2026, 3, 9),
        text="Weekly summary", event_count=5,
    )
    await store.add_summary(summary, "iran-us", "weekly")

    refreshed = await refresh_stale_pages(
        store, mock_llm, topic_slugs=["iran-us"],
        topic_names={"iran-us": "Iran-US Relations"},
    )
    assert refreshed >= 1

    page = await store.get_page("backstory:iran-us")
    assert page is not None
    assert page["page_type"] == "backstory"


async def test_refresh_skips_fresh_pages(store, mock_llm):
    """Pages within TTL should not be regenerated."""
    mock_llm.complete.return_value = "# Background\n\nContent..."

    # Save a fresh page
    await store.save_page(
        "backstory:iran-us", "Background", "backstory",
        "# Existing content", "iran-us", 30, "hash123",
    )

    refreshed = await refresh_stale_pages(
        store, mock_llm, topic_slugs=["iran-us"],
        topic_names={"iran-us": "Iran-US Relations"},
    )
    # Should not regenerate the fresh page
    assert refreshed == 0
    mock_llm.complete.assert_not_called()


async def test_refresh_regenerates_stale_pages(store, mock_llm):
    """Pages past TTL should be regenerated."""
    mock_llm.complete.return_value = "# Updated Background\n\nNew content..."

    # Save a stale page (TTL=0 means immediately stale)
    await store.save_page(
        "backstory:iran-us", "Background", "backstory",
        "# Old content", "iran-us", 0, "old_hash",
    )

    summary = Summary(
        period_start=date(2026, 3, 3), period_end=date(2026, 3, 9),
        text="Weekly summary", event_count=5,
    )
    await store.add_summary(summary, "iran-us", "weekly")

    refreshed = await refresh_stale_pages(
        store, mock_llm, topic_slugs=["iran-us"],
        topic_names={"iran-us": "Iran-US Relations"},
    )
    assert refreshed >= 1

    page = await store.get_page("backstory:iran-us")
    assert "Updated" in page["content_md"] or "New" in page["content_md"]
