"""Tests for persistent thread matching and lifecycle."""

import json
from datetime import date
from unittest.mock import AsyncMock

from nexus.engine.knowledge.events import Event
from nexus.engine.synthesis.threads import (
    compute_entity_overlap,
    match_events_to_threads,
    create_thread_slug,
    promote_thread_status,
    check_staleness,
)


def _event(summary="Test", entities=None, d="2026-03-09", sig=7):
    return Event(
        date=date.fromisoformat(d),
        summary=summary,
        significance=sig,
        entities=entities or ["IAEA", "Iran"],
        sources=[{"url": "https://example.com", "outlet": "test", "affiliation": "private", "country": "US"}],
    )


def _thread(slug="t1", headline="Thread 1", entities=None, status="active"):
    return {
        "id": 1,
        "slug": slug,
        "headline": headline,
        "key_entities": entities or ["IAEA", "Iran"],
        "status": status,
        "significance": 7,
    }


# ── Entity overlap ──────────────────────────────────────────────


def test_entity_overlap_identical():
    assert compute_entity_overlap(["IAEA", "Iran"], ["IAEA", "Iran"]) == 1.0


def test_entity_overlap_partial():
    overlap = compute_entity_overlap(["IAEA", "Iran", "US"], ["IAEA", "Iran"])
    # Jaccard: 2/3
    assert abs(overlap - 2/3) < 0.01


def test_entity_overlap_none():
    assert compute_entity_overlap(["IAEA"], ["OpenAI"]) == 0.0


def test_entity_overlap_empty():
    assert compute_entity_overlap([], ["IAEA"]) == 0.0
    assert compute_entity_overlap(["IAEA"], []) == 0.0
    assert compute_entity_overlap([], []) == 0.0


def test_entity_overlap_case_insensitive():
    overlap = compute_entity_overlap(["iaea", "IRAN"], ["IAEA", "Iran"])
    assert overlap == 1.0


# ── Thread slug ──────────────────────────────────────────────────


def test_create_thread_slug():
    slug = create_thread_slug("US-Iran Sanctions Escalation")
    assert slug == "us-iran-sanctions-escalation"


def test_create_thread_slug_special_chars():
    slug = create_thread_slug("IAEA's Nuclear Deal — Update #3")
    assert "iaea" in slug
    assert "#" not in slug
    assert "—" not in slug


# ── Thread matching ──────────────────────────────────────────────


async def test_match_high_overlap_no_llm():
    """Events with >=0.5 entity overlap match without LLM."""
    mock_llm = AsyncMock()
    events = [_event(entities=["IAEA", "Iran"])]
    threads = [_thread(entities=["IAEA", "Iran", "US"])]

    matches = await match_events_to_threads(mock_llm, events, threads)
    assert len(matches) == 1
    assert matches[0].thread_slug == "t1"
    assert matches[0].is_new_thread is False
    # High overlap — should not need LLM
    mock_llm.complete.assert_not_called()


async def test_match_no_overlap_creates_new():
    """Events with no entity overlap become new threads."""
    mock_llm = AsyncMock()
    mock_llm.complete.return_value = json.dumps([
        {"headline": "New AI Development", "event_indices": [0]}
    ])
    events = [_event(entities=["OpenAI", "GPT"])]
    threads = [_thread(entities=["IAEA", "Iran"])]

    matches = await match_events_to_threads(mock_llm, events, threads)
    assert len(matches) == 1
    assert matches[0].is_new_thread is True


async def test_match_ambiguous_uses_llm():
    """Events with 0.3-0.5 overlap use LLM for confirmation."""
    mock_llm = AsyncMock()
    mock_llm.complete.return_value = json.dumps([
        {"event_index": 0, "thread_slug": "t1", "confidence": "high"}
    ])
    # 1/3 overlap = 0.33, in ambiguous zone
    events = [_event(entities=["IAEA", "OpenAI", "Google"])]
    threads = [_thread(entities=["IAEA", "Iran", "US"])]

    matches = await match_events_to_threads(mock_llm, events, threads)
    assert len(matches) == 1
    mock_llm.complete.assert_called_once()


async def test_match_empty_events():
    mock_llm = AsyncMock()
    matches = await match_events_to_threads(mock_llm, [], [_thread()])
    assert matches == []


async def test_match_empty_threads():
    """All events become new threads when no existing threads."""
    mock_llm = AsyncMock()
    mock_llm.complete.return_value = json.dumps([
        {"headline": "New Thread", "event_indices": [0]}
    ])
    events = [_event()]
    matches = await match_events_to_threads(mock_llm, events, [])
    assert len(matches) == 1
    assert matches[0].is_new_thread is True


async def test_match_multiple_events_to_same_thread():
    mock_llm = AsyncMock()
    events = [
        _event(summary="Sanctions announced", entities=["IAEA", "Iran"]),
        _event(summary="Iran responds", entities=["Iran", "IAEA"]),
    ]
    threads = [_thread(entities=["IAEA", "Iran"])]

    matches = await match_events_to_threads(mock_llm, events, threads)
    assert len(matches) == 2
    assert all(m.thread_slug == "t1" for m in matches)


async def test_match_events_to_different_threads():
    mock_llm = AsyncMock()
    events = [
        _event(summary="Iran sanctions", entities=["Iran", "US"]),
        _event(summary="AI progress", entities=["OpenAI", "Google"]),
    ]
    threads = [
        _thread(slug="iran", entities=["Iran", "US"]),
        _thread(slug="ai", entities=["OpenAI", "Google"]),
    ]
    # OpenAI event won't match Iran thread, so LLM should be skipped for the clear matches
    # but the second event might get grouped as new
    mock_llm.complete.return_value = json.dumps([
        {"headline": "AI Progress", "event_indices": [0]}
    ])

    matches = await match_events_to_threads(mock_llm, events, threads)
    assert len(matches) == 2


# ── Thread lifecycle ─────────────────────────────────────────────


def test_promote_emerging_to_active():
    """Thread with events from 2+ days should be 'active'."""
    event_dates = [date(2026, 3, 8), date(2026, 3, 9)]
    assert promote_thread_status("emerging", event_dates) == "active"


def test_keep_emerging_single_day():
    """Thread with events from only 1 day stays 'emerging'."""
    event_dates = [date(2026, 3, 9), date(2026, 3, 9)]
    assert promote_thread_status("emerging", event_dates) == "emerging"


def test_active_stays_active():
    """Already active thread stays active with new events."""
    event_dates = [date(2026, 3, 9)]
    assert promote_thread_status("active", event_dates) == "active"


def test_resolved_stays_resolved():
    """Resolved threads don't revert."""
    event_dates = [date(2026, 3, 9)]
    assert promote_thread_status("resolved", event_dates) == "resolved"


# ── LLM fallback ────────────────────────────────────────────────


async def test_match_llm_failure_falls_back():
    """If LLM fails, unmatched events become new threads."""
    mock_llm = AsyncMock()
    mock_llm.complete.side_effect = Exception("LLM failed")
    events = [_event(entities=["OpenAI", "Google"])]
    threads = [_thread(entities=["IAEA", "Iran"])]

    matches = await match_events_to_threads(mock_llm, events, threads)
    assert len(matches) == 1
    assert matches[0].is_new_thread is True


# ── Thread staleness ───────────────────────────────────────────


def test_check_staleness_active_recent():
    """Active thread with recent events stays active."""
    assert check_staleness("active", date(2026, 3, 9), reference_date=date(2026, 3, 14)) == "active"


def test_check_staleness_active_old():
    """Active thread with no recent events → stale."""
    assert check_staleness("active", date(2026, 2, 20), reference_date=date(2026, 3, 14)) == "stale"


def test_check_staleness_resolved_unaffected():
    """Resolved threads stay resolved regardless of event age."""
    assert check_staleness("resolved", date(2026, 1, 1), reference_date=date(2026, 3, 14)) == "resolved"
