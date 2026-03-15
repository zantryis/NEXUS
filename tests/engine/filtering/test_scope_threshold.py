"""Tests for scope-dependent significance thresholds in filter_items()."""

import json
import pytest
from datetime import date
from unittest.mock import AsyncMock

from nexus.config.models import TopicConfig
from nexus.engine.filtering.filter import SIGNIFICANCE_THRESHOLD, filter_items
from nexus.engine.knowledge.events import Event
from nexus.engine.sources.polling import ContentItem


def test_significance_threshold_values():
    assert SIGNIFICANCE_THRESHOLD == {"narrow": 3, "medium": 4, "broad": 5}


def _make_topic(scope="medium"):
    return TopicConfig(
        name="Test Topic",
        subtopics=["sub"],
        scope=scope,
    )


def _make_items():
    return [
        ContentItem(
            title="Article A",
            url="https://a.com",
            source_id="t",
            full_text="Content about topic",
        ),
    ]


def _mock_llm(pass1_score=8, significance=4, is_novel=False):
    """Build a mock LLM that returns pass1 and pass2 responses."""
    mock = AsyncMock()
    pass1 = json.dumps([{"id": 0, "score": pass1_score, "reason": "Relevant"}])
    pass2 = json.dumps([{"id": 0, "significance": significance,
                         "is_novel": is_novel, "reason": "Assessment"}])
    mock.complete.side_effect = [pass1, pass2]
    return mock


@pytest.mark.asyncio
async def test_narrow_scope_threshold_3():
    """Narrow scope: significance >= 3 passes, so sig=3 is accepted."""
    topic = _make_topic("narrow")
    llm = _mock_llm(significance=3, is_novel=False)
    recent = [Event(date=date(2026, 3, 10), summary="Prior", significance=5)]

    result = await filter_items(llm, _make_items(), topic, threshold=5,
                                recent_events=recent)
    assert len(result.accepted) == 1


@pytest.mark.asyncio
async def test_narrow_scope_below_threshold():
    """Narrow scope: significance < 3 and not novel is rejected."""
    topic = _make_topic("narrow")
    llm = _mock_llm(significance=2, is_novel=False)
    recent = [Event(date=date(2026, 3, 10), summary="Prior", significance=5)]

    result = await filter_items(llm, _make_items(), topic, threshold=5,
                                recent_events=recent)
    assert len(result.accepted) == 0


@pytest.mark.asyncio
async def test_medium_scope_threshold_4():
    """Medium scope: significance >= 4 passes, so sig=4 is accepted."""
    topic = _make_topic("medium")
    llm = _mock_llm(significance=4, is_novel=False)
    recent = [Event(date=date(2026, 3, 10), summary="Prior", significance=5)]

    result = await filter_items(llm, _make_items(), topic, threshold=5,
                                recent_events=recent)
    assert len(result.accepted) == 1


@pytest.mark.asyncio
async def test_medium_scope_below_threshold():
    """Medium scope: significance=3, not novel → rejected."""
    topic = _make_topic("medium")
    llm = _mock_llm(significance=3, is_novel=False)
    recent = [Event(date=date(2026, 3, 10), summary="Prior", significance=5)]

    result = await filter_items(llm, _make_items(), topic, threshold=5,
                                recent_events=recent)
    assert len(result.accepted) == 0


@pytest.mark.asyncio
async def test_broad_scope_threshold_5():
    """Broad scope: significance >= 5 passes, so sig=5 is accepted."""
    topic = _make_topic("broad")
    llm = _mock_llm(significance=5, is_novel=False)
    recent = [Event(date=date(2026, 3, 10), summary="Prior", significance=5)]

    result = await filter_items(llm, _make_items(), topic, threshold=5,
                                recent_events=recent)
    assert len(result.accepted) == 1


@pytest.mark.asyncio
async def test_broad_scope_below_threshold():
    """Broad scope: significance=4, not novel → rejected."""
    topic = _make_topic("broad")
    llm = _mock_llm(significance=4, is_novel=False)
    recent = [Event(date=date(2026, 3, 10), summary="Prior", significance=5)]

    result = await filter_items(llm, _make_items(), topic, threshold=5,
                                recent_events=recent)
    assert len(result.accepted) == 0


@pytest.mark.asyncio
async def test_novel_bypasses_threshold():
    """Novel articles pass regardless of significance score."""
    topic = _make_topic("broad")
    llm = _mock_llm(significance=1, is_novel=True)
    recent = [Event(date=date(2026, 3, 10), summary="Prior", significance=5)]

    result = await filter_items(llm, _make_items(), topic, threshold=5,
                                recent_events=recent)
    assert len(result.accepted) == 1
