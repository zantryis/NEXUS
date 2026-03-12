"""Tests for Q&A agent."""

import pytest
from unittest.mock import AsyncMock
from datetime import date

from nexus.config.models import NexusConfig, UserConfig
from nexus.agent.qa import answer_question, _analyze_question, _gather_context
from nexus.engine.knowledge.events import Event


@pytest.fixture
def config():
    return NexusConfig(user=UserConfig(name="Tristan", output_language="en"))


@pytest.fixture
def mock_store():
    store = AsyncMock()
    store.get_topic_stats = AsyncMock(return_value=[
        {"topic_slug": "iran-us", "event_count": 10, "thread_count": 2, "latest_date": "2026-03-10"},
    ])
    store.get_recent_events = AsyncMock(return_value=[
        Event(date=date(2026, 3, 10), summary="US sanctions imposed",
              significance=8, entities=["US", "Iran"],
              sources=[{"outlet": "reuters"}]),
    ])
    store.get_active_threads = AsyncMock(return_value=[
        {"id": 1, "headline": "Sanctions Escalation", "significance": 8, "key_entities": ["US", "Iran"]},
    ])
    store.search_entities = AsyncMock(return_value=[])
    store.get_page = AsyncMock(return_value=None)
    store.get_summaries = AsyncMock(return_value=[])
    return store


async def test_analyze_question_returns_dict():
    llm = AsyncMock()
    llm.complete = AsyncMock(return_value='{"entities": ["Iran"], "intent": "recent", "language": "en"}')
    result = await _analyze_question(llm, "What happened with Iran?")
    assert result["entities"] == ["Iran"]
    assert result["intent"] == "recent"
    assert result["language"] == "en"


async def test_analyze_question_fallback_on_error():
    llm = AsyncMock()
    llm.complete = AsyncMock(side_effect=Exception("API error"))
    result = await _analyze_question(llm, "test")
    assert result == {"entities": [], "intent": "recent", "language": "en"}


async def test_analyze_question_chinese():
    llm = AsyncMock()
    llm.complete = AsyncMock(return_value='{"entities": ["伊朗"], "intent": "recent", "language": "zh"}')
    result = await _analyze_question(llm, "伊朗最近怎么了？")
    assert result["language"] == "zh"


async def test_gather_context_recent(mock_store):
    analysis = {"entities": [], "intent": "recent", "language": "en"}
    context = await _gather_context(mock_store, analysis)
    assert "US sanctions imposed" in context
    assert "Sanctions Escalation" in context


async def test_gather_context_entity_search(mock_store):
    mock_store.search_entities = AsyncMock(return_value=[
        {"id": 42, "canonical_name": "Iran", "entity_type": "country", "aliases": []},
    ])
    mock_store.get_events_for_entity = AsyncMock(return_value=[
        Event(date=date(2026, 3, 10), summary="Iran event",
              significance=7, entities=["Iran"], sources=[{"outlet": "bbc"}]),
    ])
    mock_store.get_threads_for_entity = AsyncMock(return_value=[])

    analysis = {"entities": ["Iran"], "intent": "recent", "language": "en"}
    context = await _gather_context(mock_store, analysis)
    assert "Iran event" in context
    mock_store.search_entities.assert_called_once_with("Iran", limit=3)


async def test_gather_context_background_pulls_pages(mock_store):
    mock_store.get_page = AsyncMock(return_value={
        "title": "Iran-US Backstory",
        "content_md": "The conflict traces back to 1953...",
    })
    analysis = {"entities": [], "intent": "background", "language": "en"}
    context = await _gather_context(mock_store, analysis)
    assert "1953" in context
    assert "Backstory" in context


async def test_gather_context_background_pulls_summaries(mock_store):
    from nexus.engine.knowledge.compression import Summary
    mock_store.get_summaries = AsyncMock(return_value=[
        Summary(period_start=date(2026, 3, 1), period_end=date(2026, 3, 7),
                text="Weekly summary of events", event_count=5),
    ])
    analysis = {"entities": [], "intent": "background", "language": "en"}
    context = await _gather_context(mock_store, analysis)
    assert "Weekly summary" in context


async def test_gather_context_empty_store():
    store = AsyncMock()
    store.get_topic_stats = AsyncMock(return_value=[])
    store.get_active_threads = AsyncMock(return_value=[])
    store.search_entities = AsyncMock(return_value=[])

    analysis = {"entities": ["nonexistent"], "intent": "recent", "language": "en"}
    context = await _gather_context(store, analysis)
    assert isinstance(context, str)


async def test_answer_question_two_stage(config, mock_store):
    """Full flow: analyze → gather → answer."""
    llm = AsyncMock()
    # First call: analysis. Second call: answer.
    llm.complete = AsyncMock(side_effect=[
        '{"entities": ["Iran", "sanctions"], "intent": "recent", "language": "en"}',
        "The US imposed new sanctions on Iran on March 10th.",
    ])

    answer = await answer_question(llm, mock_store, config, "What happened with Iran sanctions?")
    assert "sanctions" in answer.lower()
    assert llm.complete.call_count == 2

    # First call should use fast model
    first_call = llm.complete.call_args_list[0]
    assert first_call.kwargs["config_key"] == "filtering"

    # Second call should use agent model
    second_call = llm.complete.call_args_list[1]
    assert second_call.kwargs["config_key"] == "agent"


async def test_answer_question_chinese_response(config, mock_store):
    """Chinese question should get Chinese response."""
    llm = AsyncMock()
    llm.complete = AsyncMock(side_effect=[
        '{"entities": ["伊朗"], "intent": "recent", "language": "zh"}',
        "伊朗局势紧张。",
    ])

    await answer_question(llm, mock_store, config, "伊朗最近怎么了？")
    # System prompt should specify Chinese output
    second_call = llm.complete.call_args_list[1]
    assert "zh" in second_call.kwargs["system_prompt"]


async def test_answer_question_reference_intent(config, mock_store):
    """Reference questions should still work with general knowledge instruction."""
    llm = AsyncMock()
    llm.complete = AsyncMock(side_effect=[
        '{"entities": ["Red Bull", "engine"], "intent": "reference", "language": "en"}',
        "Red Bull uses Honda engines.",
    ])

    answer = await answer_question(llm, mock_store, config, "Who makes Red Bull's engine?")
    assert "Honda" in answer
    # System prompt should mention general knowledge
    second_call = llm.complete.call_args_list[1]
    assert "general knowledge" in second_call.kwargs["system_prompt"].lower()
