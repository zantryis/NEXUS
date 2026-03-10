"""Tests for knowledge synthesis — TopicSynthesis builder."""

import json
import pytest
from datetime import date
from unittest.mock import AsyncMock
from nexus.config.models import TopicConfig
from nexus.engine.knowledge.events import Event
from nexus.engine.sources.polling import ContentItem
from nexus.engine.synthesis.knowledge import (
    TopicSynthesis, NarrativeThread, synthesize_topic,
)


@pytest.fixture
def topic():
    return TopicConfig(name="Iran-US Relations", subtopics=["sanctions", "nuclear"])


@pytest.fixture
def events():
    return [
        Event(
            date=date(2026, 3, 9),
            summary="US announces new sanctions on Iran",
            entities=["US", "Iran", "Treasury Dept"],
            sources=[{"url": "https://nyt.com/1", "outlet": "nyt", "affiliation": "private", "country": "US"}],
            significance=8,
        ),
        Event(
            date=date(2026, 3, 9),
            summary="Iran condemns new US sanctions",
            entities=["Iran", "US", "Foreign Ministry"],
            sources=[{"url": "https://tass.com/1", "outlet": "tass", "affiliation": "state", "country": "RU"}],
            significance=7,
        ),
    ]


@pytest.fixture
def articles():
    return [
        ContentItem(
            title="Sanctions article", url="https://nyt.com/1", source_id="nyt",
            source_affiliation="private", detected_language="en",
        ),
        ContentItem(
            title="Iran response", url="https://tass.com/1", source_id="tass",
            source_affiliation="state", detected_language="en",
        ),
    ]


@pytest.mark.asyncio
async def test_synthesize_topic_produces_threads(topic, events, articles):
    mock_llm = AsyncMock()
    mock_llm.complete.return_value = json.dumps({
        "threads": [
            {
                "headline": "US-Iran sanctions escalation",
                "event_indices": [0, 1],
                "convergence": ["New sanctions were announced"],
                "divergence": [{
                    "claim": "Sanctions impact",
                    "source_a": "nyt",
                    "framing_a": "Targeted response to nuclear program",
                    "source_b": "tass",
                    "framing_b": "Unjustified economic warfare",
                }],
                "key_entities": ["US", "Iran", "Treasury Dept"],
                "significance": 8,
            }
        ]
    })

    result = await synthesize_topic(mock_llm, topic, events, articles, [], [])

    assert isinstance(result, TopicSynthesis)
    assert result.topic_name == "Iran-US Relations"
    assert len(result.threads) == 1
    assert result.threads[0].headline == "US-Iran sanctions escalation"
    assert len(result.threads[0].convergence) == 1
    assert len(result.threads[0].divergence) == 1
    assert result.source_balance == {"private": 1, "state": 1}
    assert "en" in result.languages_represented


@pytest.mark.asyncio
async def test_synthesize_topic_empty_events(topic, articles):
    mock_llm = AsyncMock()
    result = await synthesize_topic(mock_llm, topic, [], articles, [], [])

    assert result.topic_name == "Iran-US Relations"
    assert result.threads == []
    assert result.metadata["event_count"] == 0
    mock_llm.complete.assert_not_called()


@pytest.mark.asyncio
async def test_synthesize_topic_fallback_on_bad_json(topic, events, articles):
    mock_llm = AsyncMock()
    mock_llm.complete.return_value = "not valid json"

    result = await synthesize_topic(mock_llm, topic, events, articles, [], [])

    # Fallback: one thread per event
    assert len(result.threads) == 2
    assert result.metadata.get("fallback") is True


def test_topic_synthesis_model():
    syn = TopicSynthesis(
        topic_name="Test",
        threads=[NarrativeThread(headline="Thread 1", significance=7)],
        source_balance={"private": 3, "state": 1},
        languages_represented=["en", "fa"],
    )
    assert syn.topic_name == "Test"
    assert len(syn.threads) == 1
    assert syn.threads[0].significance == 7


@pytest.mark.asyncio
async def test_synthesize_new_convergence_format(topic, events, articles):
    """New convergence format with fact + confirmed_by is parsed correctly."""
    mock_llm = AsyncMock()
    mock_llm.complete.return_value = json.dumps({
        "threads": [{
            "headline": "Sanctions thread",
            "event_indices": [0, 1],
            "convergence": [
                {"fact": "New sanctions were announced", "confirmed_by": ["nyt", "tass"]}
            ],
            "divergence": [{
                "shared_event": "US sanctions announcement",
                "source_a": "nyt", "framing_a": "Targeted response",
                "source_b": "tass", "framing_b": "Economic warfare",
            }],
            "key_entities": ["US", "Iran"],
            "significance": 8,
        }]
    })

    result = await synthesize_topic(mock_llm, topic, events, articles, [], [])
    thread = result.threads[0]
    assert len(thread.convergence) == 1
    assert isinstance(thread.convergence[0], dict)
    assert thread.convergence[0]["fact"] == "New sanctions were announced"
    assert "nyt" in thread.convergence[0]["confirmed_by"]
    assert thread.divergence[0]["shared_event"] == "US sanctions announcement"


@pytest.mark.asyncio
async def test_synthesize_single_source_empty_convergence(topic, articles):
    """When LLM correctly returns empty convergence for single-source thread."""
    single_source_events = [
        Event(
            date=date(2026, 3, 9),
            summary="BBC reports on sanctions",
            entities=["US", "Iran"],
            sources=[{"url": "https://bbc.com/1", "outlet": "bbc", "affiliation": "public", "country": "GB"}],
            significance=7,
        ),
    ]
    mock_llm = AsyncMock()
    mock_llm.complete.return_value = json.dumps({
        "threads": [{
            "headline": "Single source thread",
            "event_indices": [0],
            "convergence": [],
            "divergence": [],
            "key_entities": ["US", "Iran"],
            "significance": 7,
        }]
    })

    result = await synthesize_topic(mock_llm, topic, single_source_events, articles, [], [])
    assert result.threads[0].convergence == []
    assert result.threads[0].divergence == []


def test_build_synthesis_prompt_narrow():
    from nexus.engine.synthesis.knowledge import _build_synthesis_prompt
    topic = TopicConfig(name="Iran-US", scope="narrow", subtopics=["sanctions"])
    prompt = _build_synthesis_prompt(topic)
    assert "FOCUSED" in prompt
    assert "causal chains" in prompt


def test_build_synthesis_prompt_broad():
    from nexus.engine.synthesis.knowledge import _build_synthesis_prompt
    topic = TopicConfig(name="AI/ML", scope="broad", subtopics=["agents", "reasoning"])
    prompt = _build_synthesis_prompt(topic)
    assert "BROAD" in prompt
    assert "agents, reasoning" in prompt
    assert "Do NOT merge unrelated subfields" in prompt


def test_build_synthesis_prompt_medium():
    from nexus.engine.synthesis.knowledge import _build_synthesis_prompt
    topic = TopicConfig(name="Energy", scope="medium")
    prompt = _build_synthesis_prompt(topic)
    # Medium scope should not have the scope-specific instructions
    assert "BROAD" not in prompt
    assert "FOCUSED" not in prompt
    # But should still have the base content
    assert "knowledge synthesis engine" in prompt
