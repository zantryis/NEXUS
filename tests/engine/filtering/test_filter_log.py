"""Tests for filter transparency — FilterResult log entries."""

import json
import pytest
from datetime import date
from unittest.mock import AsyncMock

from nexus.config.models import TopicConfig
from nexus.engine.filtering.filter import filter_items, FilterResult
from nexus.engine.knowledge.events import Event
from nexus.engine.sources.polling import ContentItem


@pytest.fixture
def topic():
    return TopicConfig(
        name="AI Research",
        priority="high",
        subtopics=["agents", "benchmarks"],
        filter_threshold=5.0,
    )


def _items():
    return [
        ContentItem(
            title="AI Agent Breakthrough", url="https://a.com",
            source_id="arxiv", full_text="Novel agent architecture...",
            source_affiliation="academic", source_country="US",
        ),
        ContentItem(
            title="Cooking Recipe", url="https://b.com",
            source_id="food-blog", full_text="How to make pasta...",
            source_affiliation="private", source_country="IT",
        ),
        ContentItem(
            title="AI Benchmark Results", url="https://c.com",
            source_id="papers", full_text="New benchmark shows...",
            source_affiliation="academic", source_country="GB",
        ),
    ]


async def test_filter_result_has_log_for_all_items(topic):
    """Every input item gets a log entry, regardless of outcome."""
    mock_llm = AsyncMock()
    mock_llm.complete.return_value = json.dumps([
        {"id": 0, "score": 9, "reason": "AI agents"},
        {"id": 1, "score": 1, "reason": "Not relevant"},
        {"id": 2, "score": 8, "reason": "AI benchmarks"},
    ])

    items = _items()
    result = await filter_items(mock_llm, items, topic, threshold=5)

    assert isinstance(result, FilterResult)
    assert len(result.log_entries) == 3
    assert len(result.accepted) == 2


async def test_rejected_items_have_correct_outcome(topic):
    """Items rejected in pass 1 are logged with outcome=rejected_relevance."""
    mock_llm = AsyncMock()
    mock_llm.complete.return_value = json.dumps([
        {"id": 0, "score": 8, "reason": "Relevant"},
        {"id": 1, "score": 2, "reason": "Not relevant"},
    ])

    items = _items()[:2]
    result = await filter_items(mock_llm, items, topic, threshold=5)

    rejected = [e for e in result.log_entries if e["outcome"] == "rejected_relevance"]
    assert len(rejected) == 1
    assert rejected[0]["url"] == "https://b.com"
    assert rejected[0]["relevance_score"] == 2
    assert rejected[0]["relevance_reason"] == "Not relevant"
    assert rejected[0]["passed_pass1"] is False


async def test_accepted_items_have_correct_outcome(topic):
    """Items accepted in pass 1 (no pass 2) are logged with outcome=accepted."""
    mock_llm = AsyncMock()
    mock_llm.complete.return_value = json.dumps([
        {"id": 0, "score": 9, "reason": "Great"},
    ])

    items = [_items()[0]]
    result = await filter_items(mock_llm, items, topic, threshold=5)

    accepted_logs = [e for e in result.log_entries if e["outcome"] == "accepted"]
    assert len(accepted_logs) == 1
    assert accepted_logs[0]["relevance_score"] == 9
    assert accepted_logs[0]["passed_pass1"] is True


async def test_pass2_data_recorded_in_log(topic):
    """When pass 2 runs, significance and novelty are recorded."""
    mock_llm = AsyncMock()
    pass1 = json.dumps([
        {"id": 0, "score": 8, "reason": "Relevant"},
        {"id": 1, "score": 7, "reason": "Relevant"},
    ])
    pass2 = json.dumps([
        {"id": 0, "significance": 9, "is_novel": True, "reason": "New development"},
        {"id": 1, "significance": 2, "is_novel": False, "reason": "Old news"},
    ])
    mock_llm.complete.side_effect = [pass1, pass2]

    items = _items()[:2]
    recent = [Event(date=date(2026, 3, 8), summary="Prior event", significance=5)]
    result = await filter_items(mock_llm, items, topic, threshold=5, recent_events=recent)

    # First item accepted (novel + significant)
    log_a = next(e for e in result.log_entries if e["url"] == "https://a.com")
    assert log_a["significance_score"] == 9
    assert log_a["is_novel"] is True
    assert log_a["significance_reason"] == "New development"
    assert log_a["passed_pass2"] is True
    assert log_a["outcome"] == "accepted"

    # Second item rejected (low significance + not novel)
    log_b = next(e for e in result.log_entries if e["url"] == "https://b.com")
    assert log_b["significance_score"] == 2
    assert log_b["is_novel"] is False
    assert log_b["passed_pass2"] is False
    assert log_b["outcome"] == "rejected_significance"


async def test_log_entries_have_source_metadata(topic):
    """Log entries capture source metadata from ContentItem."""
    mock_llm = AsyncMock()
    mock_llm.complete.return_value = json.dumps([
        {"id": 0, "score": 9, "reason": "Good"},
    ])

    items = [_items()[0]]
    result = await filter_items(mock_llm, items, topic, threshold=5)

    log = result.log_entries[0]
    assert log["source_id"] == "arxiv"
    assert log["source_affiliation"] == "academic"
    assert log["source_country"] == "US"
    assert log["topic_slug"] == "ai-research"


async def test_empty_items_returns_empty_result(topic):
    """Filtering empty list returns empty FilterResult."""
    mock_llm = AsyncMock()
    result = await filter_items(mock_llm, [], topic, threshold=5)

    assert isinstance(result, FilterResult)
    assert result.accepted == []
    assert result.log_entries == []


async def test_all_items_rejected_returns_empty_accepted(topic):
    """When all items fail pass 1, accepted is empty but log has all entries."""
    mock_llm = AsyncMock()
    mock_llm.complete.return_value = json.dumps([
        {"id": 0, "score": 1, "reason": "Irrelevant"},
        {"id": 1, "score": 2, "reason": "Irrelevant"},
    ])

    items = _items()[:2]
    result = await filter_items(mock_llm, items, topic, threshold=5)

    assert len(result.accepted) == 0
    assert len(result.log_entries) == 2
    assert all(e["outcome"] == "rejected_relevance" for e in result.log_entries)
