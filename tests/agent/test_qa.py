"""Tests for Q&A agent."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from datetime import date

from nexus.config.models import NexusConfig, UserConfig
from nexus.agent.qa import answer_question
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
        {"headline": "Sanctions Escalation", "significance": 8},
    ])
    return store


async def test_answer_question_calls_llm(config, mock_store):
    llm = AsyncMock()
    llm.complete = AsyncMock(return_value="The sanctions were imposed on March 10th.")

    answer = await answer_question(llm, mock_store, config, "What happened with Iran sanctions?")

    assert "sanctions" in answer.lower()
    llm.complete.assert_called_once()
    call_kwargs = llm.complete.call_args.kwargs
    assert call_kwargs["config_key"] == "agent"


async def test_answer_includes_context(config, mock_store):
    llm = AsyncMock()
    llm.complete = AsyncMock(return_value="Answer based on context.")

    await answer_question(llm, mock_store, config, "Question?")

    user_prompt = llm.complete.call_args.kwargs["user_prompt"]
    assert "US sanctions imposed" in user_prompt
    assert "Sanctions Escalation" in user_prompt


async def test_answer_empty_store(config):
    store = AsyncMock()
    store.get_topic_stats = AsyncMock(return_value=[])
    store.get_active_threads = AsyncMock(return_value=[])

    llm = AsyncMock()
    llm.complete = AsyncMock(return_value="No data available.")

    answer = await answer_question(llm, store, config, "Any news?")
    assert isinstance(answer, str)


async def test_answer_question_multiple_topics(config):
    store = AsyncMock()
    store.get_topic_stats = AsyncMock(return_value=[
        {"topic_slug": "iran-us", "event_count": 5, "thread_count": 1, "latest_date": "2026-03-10"},
        {"topic_slug": "ai-ml", "event_count": 3, "thread_count": 1, "latest_date": "2026-03-10"},
    ])
    store.get_recent_events = AsyncMock(return_value=[
        Event(date=date(2026, 3, 10), summary="Something",
              significance=5, entities=["X"], sources=[{"outlet": "test"}]),
    ])
    store.get_active_threads = AsyncMock(return_value=[])

    llm = AsyncMock()
    llm.complete = AsyncMock(return_value="Multi-topic answer.")

    await answer_question(llm, store, config, "What's happening?")
    # Should be called for both topics
    assert store.get_recent_events.call_count == 2
