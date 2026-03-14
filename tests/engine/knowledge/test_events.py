"""Tests for knowledge layer event logging."""

import pytest
from datetime import date
from unittest.mock import AsyncMock
from nexus.engine.knowledge.events import (
    Event,
    VALID_TONES,
    load_events,
    save_events,
    extract_event,
    append_events,
    are_independent,
    has_independent_sources,
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


# ── Framing Extraction ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_extract_event_includes_framing(topic):
    """LLM returns editorial_tone/focus/actor_framing → combined into source framing."""
    mock_llm = AsyncMock()
    mock_llm.complete.return_value = '''{
        "date": "2026-03-09",
        "summary": "Iran condemns new sanctions",
        "entities": ["Iran", "US"],
        "relation_to_prior": "",
        "significance": 7,
        "editorial_tone": "critical",
        "editorial_focus": "Focuses on economic hardship and civilian impact",
        "actor_framing": "portrays US as aggressor imposing collective punishment"
    }'''

    item = ContentItem(
        title="Iran sanctions",
        url="https://example.com/iran",
        source_id="press-tv",
        source_affiliation="state",
        source_country="IR",
        full_text="Iran's foreign ministry condemned...",
        language="en",
    )

    event = await extract_event(mock_llm, item, topic, existing_events=[],
                                current_date=date(2026, 3, 10))
    assert event is not None
    framing = event.sources[0].get("framing", "")
    assert "[critical]" in framing
    assert "economic hardship" in framing
    assert "aggressor" in framing


@pytest.mark.asyncio
async def test_extract_event_framing_optional(topic):
    """LLM returns no framing fields → graceful fallback to empty string."""
    mock_llm = AsyncMock()
    mock_llm.complete.return_value = '''{
        "date": "2026-03-09",
        "summary": "Routine event",
        "entities": ["US"],
        "significance": 5
    }'''

    item = ContentItem(
        title="Routine",
        url="https://example.com/routine",
        source_id="test-feed",
        full_text="Nothing special here.",
        language="en",
    )

    event = await extract_event(mock_llm, item, topic, existing_events=[],
                                current_date=date(2026, 3, 10))
    assert event is not None
    framing = event.sources[0].get("framing", "")
    assert framing == ""


def test_merge_events_preserves_framing():
    """merge_events preserves framing in source dicts from both events."""
    from nexus.engine.knowledge.events import merge_events

    target = Event(
        date=date(2026, 3, 9),
        summary="Sanctions announced",
        entities=["US", "Iran"],
        sources=[{
            "url": "https://nyt.com/1", "outlet": "nyt",
            "affiliation": "private", "country": "US",
            "framing": "[neutral] Reports sanctions details; US as policy enforcer",
        }],
        significance=8,
    )
    source = Event(
        date=date(2026, 3, 9),
        summary="Sanctions announced",
        entities=["US", "Iran"],
        sources=[{
            "url": "https://tass.com/1", "outlet": "tass",
            "affiliation": "state", "country": "RU",
            "framing": "[critical] Emphasizes economic warfare; US as aggressor",
        }],
        significance=7,
    )

    merged = merge_events(target, source)
    assert len(merged.sources) == 2
    assert merged.sources[0]["framing"] == "[neutral] Reports sanctions details; US as policy enforcer"
    assert merged.sources[1]["framing"] == "[critical] Emphasizes economic warfare; US as aggressor"


# ── Tone Validation ───────────────────────────────────────────


def test_valid_tones_constant():
    """VALID_TONES has exactly 8 members matching the prompt vocabulary."""
    assert len(VALID_TONES) == 8
    assert "neutral" in VALID_TONES
    assert "alarmist" in VALID_TONES
    assert "critical" in VALID_TONES
    assert "defensive" in VALID_TONES


@pytest.mark.asyncio
async def test_extract_event_invalid_tone_defaults_to_neutral(topic):
    """LLM returns invalid editorial_tone → defaults to 'neutral'."""
    mock_llm = AsyncMock()
    mock_llm.complete.return_value = '''{
        "date": "2026-03-09",
        "summary": "Policy announcement",
        "entities": ["US"],
        "significance": 6,
        "editorial_tone": "hawkish",
        "editorial_focus": "Aggressive posture toward rivals",
        "actor_framing": "US as dominant power"
    }'''

    item = ContentItem(
        title="Policy",
        url="https://example.com/policy",
        source_id="test-feed",
        source_affiliation="private",
        source_country="US",
        full_text="The administration announced...",
        language="en",
    )

    event = await extract_event(mock_llm, item, topic, existing_events=[],
                                current_date=date(2026, 3, 10))
    assert event is not None
    framing = event.sources[0].get("framing", "")
    assert framing.startswith("[neutral]"), f"Expected [neutral], got: {framing}"
    assert "hawkish" not in framing


# ── Source Independence ───────────────────────────────────────


def test_are_independent_different_affiliation():
    """Different affiliation, same country → independent."""
    assert are_independent(
        {"affiliation": "private", "country": "US"},
        {"affiliation": "state", "country": "US"},
    ) is True


def test_are_independent_different_country():
    """Same affiliation, different country → independent."""
    assert are_independent(
        {"affiliation": "state", "country": "CN"},
        {"affiliation": "state", "country": "RU"},
    ) is True


def test_are_independent_same_both():
    """Same affiliation AND same country → NOT independent."""
    assert are_independent(
        {"affiliation": "state", "country": "CN"},
        {"affiliation": "state", "country": "CN"},
    ) is False


def test_are_independent_missing_metadata():
    """Missing or empty metadata → assume independent (benefit of doubt)."""
    assert are_independent(
        {"affiliation": "", "country": "US"},
        {"affiliation": "state", "country": "CN"},
    ) is True
    assert are_independent(
        {"affiliation": "private", "country": "US"},
        {},
    ) is True


def test_has_independent_sources():
    """Event with independent sources → True; same-affiliation/country → False."""
    independent_event = Event(
        date=date(2026, 3, 9),
        summary="Test",
        sources=[
            {"outlet": "nyt", "affiliation": "private", "country": "US"},
            {"outlet": "tass", "affiliation": "state", "country": "RU"},
        ],
    )
    assert has_independent_sources(independent_event) is True

    non_independent_event = Event(
        date=date(2026, 3, 9),
        summary="Test",
        sources=[
            {"outlet": "cgtn", "affiliation": "state", "country": "CN"},
            {"outlet": "xinhua", "affiliation": "state", "country": "CN"},
        ],
    )
    assert has_independent_sources(non_independent_event) is False

    single_source = Event(
        date=date(2026, 3, 9),
        summary="Test",
        sources=[{"outlet": "nyt", "affiliation": "private", "country": "US"}],
    )
    assert has_independent_sources(single_source) is False
