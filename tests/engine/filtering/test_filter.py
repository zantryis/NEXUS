"""Tests for relevance filtering via LLM."""

import json
import pytest
from datetime import date
from unittest.mock import AsyncMock
from nexus.engine.sources.polling import ContentItem
from nexus.config.models import TopicConfig
from nexus.engine.filtering.filter import (
    score_relevance, score_batch, filter_items, FilterResult,
    score_significance_batch, _format_event_context,
)


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
    mock_llm.complete.assert_called_once()


@pytest.mark.asyncio
async def test_score_batch_fallback_on_bad_json(topic):
    mock_llm = AsyncMock()
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
async def test_filter_items_pass1_only(topic):
    """Without recent_events, filter_items does pass 1 only."""
    mock_llm = AsyncMock()
    mock_llm.complete.return_value = (
        '[{"id": 0, "score": 8, "reason": "Relevant"}, '
        '{"id": 1, "score": 2, "reason": "Not relevant"}]'
    )

    items = [
        ContentItem(title="AI News", url="https://a.com", source_id="t", full_text="AI agents"),
        ContentItem(title="Cooking", url="https://b.com", source_id="t", full_text="Pasta"),
    ]

    result = await filter_items(mock_llm, items, topic, threshold=5)
    assert isinstance(result, FilterResult)
    assert len(result.accepted) == 1
    assert result.accepted[0].relevance_score == 8
    assert mock_llm.complete.call_count == 1
    # Log should have entries for both items
    assert len(result.log_entries) == 2


@pytest.mark.asyncio
async def test_filter_items_uses_topic_threshold():
    """filter_items uses topic.filter_threshold when no explicit threshold given."""
    topic = TopicConfig(
        name="AI Research",
        subtopics=["agents"],
        filter_threshold=8.0,
    )
    mock_llm = AsyncMock()
    mock_llm.complete.return_value = (
        '[{"id": 0, "score": 7, "reason": "Good"}, '
        '{"id": 1, "score": 9, "reason": "Great"}]'
    )
    items = [
        ContentItem(title="A", url="https://a.com", source_id="t", full_text="text"),
        ContentItem(title="B", url="https://b.com", source_id="t", full_text="text"),
    ]

    result = await filter_items(mock_llm, items, topic)
    assert len(result.accepted) == 1
    assert result.accepted[0].title == "B"

    # Explicit threshold overrides
    mock_llm.complete.return_value = (
        '[{"id": 0, "score": 7, "reason": "Good"}, '
        '{"id": 1, "score": 9, "reason": "Great"}]'
    )
    items2 = [
        ContentItem(title="A", url="https://a.com", source_id="t", full_text="text"),
        ContentItem(title="B", url="https://b.com", source_id="t", full_text="text"),
    ]
    result2 = await filter_items(mock_llm, items2, topic, threshold=5)
    assert len(result2.accepted) == 2


@pytest.mark.asyncio
async def test_filter_items_two_pass(topic):
    """With recent_events, filter_items runs both passes."""
    from nexus.engine.knowledge.events import Event

    mock_llm = AsyncMock()
    # Pass 1 response: both articles relevant
    pass1_response = '[{"id": 0, "score": 8, "reason": "Relevant"}, {"id": 1, "score": 7, "reason": "Relevant"}]'
    # Pass 2 response: article 0 is novel+significant, article 1 is not novel
    pass2_response = json.dumps([
        {"id": 0, "significance": 8, "is_novel": True, "reason": "New development"},
        {"id": 1, "significance": 3, "is_novel": False, "reason": "Already covered"},
    ])
    mock_llm.complete.side_effect = [pass1_response, pass2_response]

    items = [
        ContentItem(title="New AI breakthrough", url="https://a.com", source_id="t", full_text="Novel research"),
        ContentItem(title="Old AI news", url="https://b.com", source_id="t", full_text="We already know this"),
    ]

    recent = [Event(date=date(2026, 3, 8), summary="Previous AI event", significance=7)]
    result = await filter_items(mock_llm, items, topic, threshold=5, recent_events=recent)

    # Article 0 passes (novel + significant), article 1 filtered (sig=3, not novel)
    assert len(result.accepted) == 1
    assert result.accepted[0].title == "New AI breakthrough"
    # Two LLM calls: pass 1 batch + pass 2 batch
    assert mock_llm.complete.call_count == 2
    # Log should have entries for both items with pass 2 data
    assert len(result.log_entries) == 2


@pytest.mark.asyncio
async def test_score_significance_batch(topic):
    mock_llm = AsyncMock()
    mock_llm.complete.return_value = json.dumps([
        {"id": 0, "significance": 9, "is_novel": True, "reason": "Major"},
    ])

    items = [ContentItem(title="A", url="https://a.com", source_id="t", full_text="text")]
    results = await score_significance_batch(mock_llm, items, topic, "- [2026-03-08] Prior event")

    assert len(results) == 1
    assert results[0]["significance"] == 9
    assert results[0]["is_novel"] is True


def test_format_event_context():
    from nexus.engine.knowledge.events import Event

    events = [
        Event(date=date(2026, 3, 8), summary="Event A", significance=7),
        Event(date=date(2026, 3, 9), summary="Event B", significance=5),
    ]
    ctx = _format_event_context(events)
    assert "Event A" in ctx
    assert "Event B" in ctx
    assert "2026-03-08" in ctx


def test_format_event_context_empty():
    assert _format_event_context([]) == ""


@pytest.mark.asyncio
async def test_pass2_parse_error_fallback_rejects(topic):
    """Pass 2 batch parse failure → conservative fallback (reject, not accept)."""
    mock_llm = AsyncMock()
    mock_llm.complete.return_value = "totally not valid json {{{"

    items = [
        ContentItem(title="A", url="https://a.com", source_id="t", full_text="text"),
        ContentItem(title="B", url="https://b.com", source_id="t", full_text="text"),
    ]

    results = await score_significance_batch(mock_llm, items, topic, "- prior event")
    assert len(results) == 2
    for r in results:
        assert r["significance"] == 3, f"Expected significance=3, got {r['significance']}"
        assert r["is_novel"] is False, f"Expected is_novel=False, got {r['is_novel']}"
