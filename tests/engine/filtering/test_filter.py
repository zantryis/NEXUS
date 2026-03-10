"""Tests for relevance filtering via LLM."""

import pytest
from unittest.mock import AsyncMock
from nexus.engine.sources.polling import ContentItem
from nexus.config.models import TopicConfig
from nexus.engine.filtering.filter import score_relevance, filter_items


@pytest.fixture
def topic():
    return TopicConfig(
        name="AI Research",
        priority="high",
        subtopics=["agents", "benchmarks"],
    )


@pytest.fixture
def item():
    return ContentItem(
        title="New AI Agent Benchmark Released",
        url="https://example.com/ai-bench",
        source_id="test",
        full_text="Researchers released a new benchmark for evaluating AI agents...",
    )


@pytest.mark.asyncio
async def test_score_relevance_high(topic, item):
    mock_llm = AsyncMock()
    mock_llm.complete.return_value = '{"score": 9, "reason": "Directly about AI benchmarks"}'

    score, reason = await score_relevance(mock_llm, item, topic)
    assert score == 9
    assert "benchmark" in reason.lower()
    mock_llm.complete.assert_called_once()


@pytest.mark.asyncio
async def test_score_relevance_low(topic, item):
    mock_llm = AsyncMock()
    mock_llm.complete.return_value = '{"score": 2, "reason": "Not related to AI"}'

    score, reason = await score_relevance(mock_llm, item, topic)
    assert score == 2


@pytest.mark.asyncio
async def test_score_relevance_handles_bad_json(topic, item):
    mock_llm = AsyncMock()
    mock_llm.complete.return_value = "not json at all"

    score, reason = await score_relevance(mock_llm, item, topic)
    assert score == 0
    assert "parse" in reason.lower()


@pytest.mark.asyncio
async def test_filter_items(topic, item):
    mock_llm = AsyncMock()
    mock_llm.complete.return_value = '{"score": 8, "reason": "Relevant"}'

    low_item = ContentItem(
        title="Cooking Recipe",
        url="https://example.com/food",
        source_id="test",
        full_text="How to make pasta.",
    )

    # First call scores high, second scores low
    mock_llm.complete.side_effect = [
        '{"score": 8, "reason": "Relevant"}',
        '{"score": 2, "reason": "Not relevant"}',
    ]

    results = await filter_items(mock_llm, [item, low_item], topic, threshold=5)
    assert len(results) == 1
    assert results[0].relevance_score == 8
