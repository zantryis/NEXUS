"""Tests for knowledge layer event logging."""

import pytest
from pathlib import Path
from datetime import date
from unittest.mock import AsyncMock
from nexus.engine.knowledge.events import (
    Event,
    load_events,
    save_events,
    extract_event,
    append_events,
)
from nexus.engine.sources.polling import ContentItem
from nexus.config.models import TopicConfig


@pytest.fixture
def topic():
    return TopicConfig(name="AI Research", priority="high", subtopics=["agents"])


@pytest.fixture
def sample_event():
    return Event(
        date=date(2026, 3, 9),
        summary="New AI agent benchmark released by researchers",
        sources=[{"url": "https://example.com/1", "language": "en", "outlet": "Test"}],
        entities=["OpenAI"],
        relation_to_prior="First event",
        significance=8,
    )


def test_event_model(sample_event):
    assert sample_event.significance == 8
    assert len(sample_event.sources) == 1


def test_save_and_load_events(tmp_path, sample_event):
    events_file = tmp_path / "events.yaml"
    save_events(events_file, [sample_event])
    loaded = load_events(events_file)
    assert len(loaded) == 1
    assert loaded[0].summary == sample_event.summary
    assert loaded[0].date == date(2026, 3, 9)


def test_load_events_missing_file(tmp_path):
    events_file = tmp_path / "nonexistent.yaml"
    assert load_events(events_file) == []


def test_append_events(tmp_path, sample_event):
    events_file = tmp_path / "events.yaml"
    save_events(events_file, [sample_event])

    new_event = Event(
        date=date(2026, 3, 10),
        summary="Follow-up event",
        sources=[],
        entities=[],
        relation_to_prior="After first event",
        significance=6,
    )
    append_events(events_file, [new_event])
    loaded = load_events(events_file)
    assert len(loaded) == 2


@pytest.mark.asyncio
async def test_extract_event(topic):
    mock_llm = AsyncMock()
    mock_llm.complete.return_value = '''{
        "date": "2026-03-09",
        "summary": "Researchers release new benchmark",
        "entities": ["OpenAI"],
        "relation_to_prior": "Novel development",
        "significance": 8
    }'''

    item = ContentItem(
        title="AI Benchmark",
        url="https://example.com/bench",
        source_id="test-feed",
        full_text="Researchers released a new benchmark...",
        language="en",
    )

    event = await extract_event(mock_llm, item, topic, existing_events=[],
                                current_date=date(2026, 3, 10))
    assert event.summary == "Researchers release new benchmark"
    assert event.significance == 8
    assert event.sources[0]["url"] == "https://example.com/bench"
    assert event.date == date(2026, 3, 9)


@pytest.mark.asyncio
async def test_extract_event_clamps_future_date(topic):
    """LLM returns a future date — it gets clamped to current_date."""
    mock_llm = AsyncMock()
    mock_llm.complete.return_value = '''{
        "date": "2026-04-15",
        "summary": "Speculative future event",
        "entities": ["Iran"],
        "relation_to_prior": "",
        "significance": 7
    }'''

    item = ContentItem(
        title="Analysis",
        url="https://example.com/analysis",
        source_id="test-feed",
        full_text="Next month, negotiations will...",
        language="en",
    )

    event = await extract_event(mock_llm, item, topic, existing_events=[],
                                current_date=date(2026, 3, 10))
    assert event is not None
    assert event.date == date(2026, 3, 10)  # Clamped to current_date


@pytest.mark.asyncio
async def test_extract_event_passes_date_to_prompt(topic):
    """Verify current_date appears in the system prompt sent to LLM."""
    mock_llm = AsyncMock()
    mock_llm.complete.return_value = '''{
        "date": "2026-03-10",
        "summary": "Test event",
        "entities": [],
        "significance": 5
    }'''

    item = ContentItem(
        title="Test",
        url="https://example.com/test",
        source_id="test-feed",
        full_text="Test article text",
        language="en",
    )

    await extract_event(mock_llm, item, topic, existing_events=[],
                        current_date=date(2026, 3, 10))

    call_args = mock_llm.complete.call_args
    system_prompt = call_args.kwargs.get("system_prompt", call_args[1].get("system_prompt", ""))
    user_prompt = call_args.kwargs.get("user_prompt", call_args[1].get("user_prompt", ""))

    assert "2026-03-10" in system_prompt
    assert "2026-03-10" in user_prompt
