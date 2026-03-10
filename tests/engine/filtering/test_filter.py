"""Tests for relevance filtering via LLM."""

import pytest
from unittest.mock import AsyncMock
from nexus.engine.sources.polling import ContentItem
from nexus.config.models import TopicConfig
from nexus.engine.filtering.filter import score_relevance, score_batch, filter_items


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


@pytest.mark.asyncio
async def test_score_relevance_handles_bad_json(topic, item):
    mock_llm = AsyncMock()
    mock_llm.complete.return_value = "not json at all"

    score, reason = await score_relevance(mock_llm, item, topic)
    assert score == 0


@pytest.mark.asyncio
async def test_score_batch(topic):
    mock_llm = AsyncMock()
    mock_llm.complete.return_value = (
        '[{"id": 0, "score": 8, "reason": "Relevant"}, '
        '{"id": 1, "score": 2, "reason": "Not relevant"}]'
    )

    items = [
        ContentItem(title="AI News", url="https://a.com", source_id="t", full_text="AI stuff"),
        ContentItem(title="Cooking", url="https://b.com", source_id="t", full_text="Pasta"),
    ]

    scores = await score_batch(mock_llm, items, topic)
    assert len(scores) == 2
    assert scores[0] == (8, "Relevant")
    assert scores[1] == (2, "Not relevant")
    # Only one LLM call for the batch
    mock_llm.complete.assert_called_once()


@pytest.mark.asyncio
async def test_score_batch_fallback_on_bad_json(topic):
    mock_llm = AsyncMock()
    # First call (batch) fails, then two individual fallback calls
    mock_llm.complete.side_effect = [
        "bad json",
        '{"score": 7, "reason": "ok"}',
        '{"score": 3, "reason": "no"}',
    ]

    items = [
        ContentItem(title="A", url="https://a.com", source_id="t", full_text="text"),
        ContentItem(title="B", url="https://b.com", source_id="t", full_text="text"),
    ]

    scores = await score_batch(mock_llm, items, topic)
    assert len(scores) == 2
    assert scores[0][0] == 7
    assert scores[1][0] == 3


@pytest.mark.asyncio
async def test_filter_items_uses_batching(topic):
    mock_llm = AsyncMock()
    # One batch call returns both scores
    mock_llm.complete.return_value = (
        '[{"id": 0, "score": 8, "reason": "Relevant"}, '
        '{"id": 1, "score": 2, "reason": "Not relevant"}]'
    )

    items = [
        ContentItem(title="AI News", url="https://a.com", source_id="t", full_text="AI agents"),
        ContentItem(title="Cooking", url="https://b.com", source_id="t", full_text="Pasta"),
    ]

    results = await filter_items(mock_llm, items, topic, threshold=5)
    assert len(results) == 1
    assert results[0].relevance_score == 8
    # Should be a single batch call, not two individual calls
    assert mock_llm.complete.call_count == 1
