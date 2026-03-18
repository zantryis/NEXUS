"""Integration tests for hybrid funnel filtering (pairwise_filtering=True)."""

import json
import pytest
from datetime import date
from unittest.mock import AsyncMock

from nexus.config.models import TopicConfig
from nexus.engine.filtering.filter import filter_items, FilterResult
from nexus.engine.knowledge.events import Event
from nexus.engine.sources.polling import ContentItem


def _make_topic(**kwargs):
    defaults = dict(
        name="AI Research",
        subtopics=["agents", "LLMs"],
        pairwise_filtering=True,
    )
    defaults.update(kwargs)
    return TopicConfig(**defaults)


def _item(title, source_id="feed"):
    return ContentItem(
        title=title,
        url=f"https://example.com/{title.lower().replace(' ', '-')}",
        source_id=source_id,
        full_text=f"Full text about {title}",
    )


# ── Basic hybrid funnel flow ────────────────────────────────────


@pytest.mark.asyncio
async def test_hybrid_funnel_keep_items_pass_directly():
    """Items scoring 7+ go straight to KEEP pool, skip pairwise."""
    topic = _make_topic()
    items = [_item("AI Agent Paper"), _item("LLM Benchmark")]

    mock_llm = AsyncMock()
    # Pass 1: both score 8 (KEEP)
    mock_llm.complete.return_value = json.dumps([
        {"id": 0, "score": 8, "reason": "AI agents"},
        {"id": 1, "score": 9, "reason": "LLM benchmark"},
    ])

    result = await filter_items(mock_llm, items, topic, threshold=5)

    assert isinstance(result, FilterResult)
    assert len(result.accepted) == 2
    # Only 1 LLM call (pass 1 scoring) — no pairwise needed
    assert mock_llm.complete.call_count == 1


@pytest.mark.asyncio
async def test_hybrid_funnel_drop_items_rejected():
    """Items scoring 1-2 go to DROP, never see pairwise."""
    topic = _make_topic()
    items = [_item("Cooking Recipe"), _item("Cat Video")]

    mock_llm = AsyncMock()
    mock_llm.complete.return_value = json.dumps([
        {"id": 0, "score": 1, "reason": "Irrelevant"},
        {"id": 1, "score": 2, "reason": "Irrelevant"},
    ])

    result = await filter_items(mock_llm, items, topic, threshold=5)

    assert len(result.accepted) == 0
    assert all(e["outcome"] == "rejected_relevance" for e in result.log_entries)
    assert all(e["triage_band"] == "drop" for e in result.log_entries)


@pytest.mark.asyncio
async def test_hybrid_funnel_maybe_promoted_via_pairwise():
    """MAYBE item wins pairwise comparisons against KEEP refs → promoted."""
    topic = _make_topic()
    items = [_item("Strong AI Paper"), _item("Borderline Article")]

    mock_llm = AsyncMock()

    # Pass 1: first item scores 8 (KEEP), second scores 5 (MAYBE)
    pass1 = json.dumps([
        {"id": 0, "score": 8, "reason": "Strong match"},
        {"id": 1, "score": 5, "reason": "Borderline"},
    ])

    # Pairwise: MAYBE vs 1 ref (only 1 KEEP item), MAYBE wins
    pairwise = json.dumps([{"pair": 1, "winner": "A", "reason": "More specific"}])

    mock_llm.complete.side_effect = [pass1, pairwise]

    result = await filter_items(mock_llm, items, topic, threshold=5)

    assert len(result.accepted) == 2
    # Check MAYBE item was promoted via pairwise
    borderline_log = next(e for e in result.log_entries if "borderline" in e["url"])
    assert borderline_log["triage_band"] == "maybe"
    assert borderline_log["pairwise_promoted"] is True
    assert borderline_log["passed_pass1"] is True


@pytest.mark.asyncio
async def test_hybrid_funnel_maybe_rejected_via_pairwise():
    """MAYBE item loses pairwise comparisons → rejected."""
    topic = _make_topic()
    items = [_item("Strong AI Paper"), _item("Weak Article")]

    mock_llm = AsyncMock()

    pass1 = json.dumps([
        {"id": 0, "score": 8, "reason": "Strong match"},
        {"id": 1, "score": 4, "reason": "Weak"},
    ])

    # Pairwise: MAYBE loses
    pairwise = json.dumps([{"pair": 1, "winner": "B", "reason": "Ref is better"}])

    mock_llm.complete.side_effect = [pass1, pairwise]

    result = await filter_items(mock_llm, items, topic, threshold=5)

    assert len(result.accepted) == 1
    assert result.accepted[0].title == "Strong AI Paper"

    weak_log = next(e for e in result.log_entries if "weak" in e["url"])
    assert weak_log["triage_band"] == "maybe"
    assert weak_log["pairwise_promoted"] is False
    assert weak_log["outcome"] == "rejected_relevance"


# ── Hybrid funnel with pass 2 ──────────────────────────────────


@pytest.mark.asyncio
async def test_hybrid_funnel_with_pass2():
    """Full hybrid flow: pass 1 triage → pairwise → pass 2 significance."""
    topic = _make_topic()
    items = [_item("AI Breakthrough"), _item("Marginal Article"), _item("Junk")]

    recent = [Event(date=date(2026, 3, 10), summary="Prior event", significance=5)]

    mock_llm = AsyncMock()

    # Pass 1: KEEP(8), MAYBE(5), DROP(1)
    pass1 = json.dumps([
        {"id": 0, "score": 8, "reason": "AI breakthrough"},
        {"id": 1, "score": 5, "reason": "Marginal"},
        {"id": 2, "score": 1, "reason": "Junk"},
    ])

    # Pairwise: MAYBE wins against KEEP ref
    pairwise = json.dumps([{"pair": 1, "winner": "A", "reason": "Novel angle"}])

    # Pass 2: both survivors are significant
    pass2 = json.dumps([
        {"id": 0, "significance": 8, "is_novel": True, "reason": "New development"},
        {"id": 1, "significance": 6, "is_novel": True, "reason": "Adds context"},
    ])

    mock_llm.complete.side_effect = [pass1, pairwise, pass2]

    result = await filter_items(mock_llm, items, topic, threshold=5,
                                recent_events=recent)

    assert len(result.accepted) == 2
    assert mock_llm.complete.call_count == 3  # pass1 + pairwise + pass2

    junk_log = next(e for e in result.log_entries if "junk" in e["url"])
    assert junk_log["outcome"] == "rejected_relevance"
    assert junk_log["triage_band"] == "drop"


# ── Edge cases ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_hybrid_funnel_no_maybe_items():
    """Clean bimodal: all KEEP or DROP → no pairwise calls needed."""
    topic = _make_topic()
    items = [_item("Great Paper"), _item("Total Junk")]

    mock_llm = AsyncMock()
    mock_llm.complete.return_value = json.dumps([
        {"id": 0, "score": 9, "reason": "Perfect match"},
        {"id": 1, "score": 1, "reason": "Irrelevant"},
    ])

    result = await filter_items(mock_llm, items, topic, threshold=5)

    assert len(result.accepted) == 1
    # Only 1 LLM call — no pairwise needed for clean split
    assert mock_llm.complete.call_count == 1


@pytest.mark.asyncio
async def test_hybrid_funnel_all_maybe_no_refs():
    """All items score 3-6 → no KEEP refs → all MAYBE promoted (can't compare)."""
    topic = _make_topic()
    items = [_item("Article A"), _item("Article B")]

    mock_llm = AsyncMock()
    mock_llm.complete.return_value = json.dumps([
        {"id": 0, "score": 5, "reason": "Borderline"},
        {"id": 1, "score": 4, "reason": "Borderline"},
    ])

    result = await filter_items(mock_llm, items, topic, threshold=5)

    # All promoted since no refs to compare against
    assert len(result.accepted) == 2


@pytest.mark.asyncio
async def test_hybrid_funnel_disabled_uses_threshold():
    """With pairwise_filtering=False, uses simple threshold (original behavior)."""
    topic = _make_topic(pairwise_filtering=False)
    items = [_item("Score 8"), _item("Score 4")]

    mock_llm = AsyncMock()
    mock_llm.complete.return_value = json.dumps([
        {"id": 0, "score": 8, "reason": "Good"},
        {"id": 1, "score": 4, "reason": "Below threshold"},
    ])

    result = await filter_items(mock_llm, items, topic, threshold=5)

    assert len(result.accepted) == 1
    assert result.accepted[0].title == "Score 8"
    # No triage_band or pairwise_promoted in logs
    log = result.log_entries[0]
    assert "triage_band" not in log


@pytest.mark.asyncio
async def test_hybrid_funnel_log_entries_complete():
    """Hybrid funnel log entries have all expected fields."""
    topic = _make_topic()
    items = [_item("KEEP Item"), _item("MAYBE Item"), _item("DROP Item")]

    mock_llm = AsyncMock()
    pass1 = json.dumps([
        {"id": 0, "score": 9, "reason": "Keep"},
        {"id": 1, "score": 5, "reason": "Maybe"},
        {"id": 2, "score": 1, "reason": "Drop"},
    ])
    pairwise = json.dumps([{"pair": 1, "winner": "B", "reason": "Ref better"}])
    mock_llm.complete.side_effect = [pass1, pairwise]

    result = await filter_items(mock_llm, items, topic, threshold=5)

    assert len(result.log_entries) == 3

    keep_log = next(e for e in result.log_entries if "keep" in e["url"])
    assert keep_log["triage_band"] == "keep"
    assert keep_log["passed_pass1"] is True

    maybe_log = next(e for e in result.log_entries if "maybe" in e["url"])
    assert maybe_log["triage_band"] == "maybe"
    assert maybe_log["pairwise_promoted"] is False

    drop_log = next(e for e in result.log_entries if "drop" in e["url"])
    assert drop_log["triage_band"] == "drop"
    assert drop_log["outcome"] == "rejected_relevance"
