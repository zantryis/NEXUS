"""Tests for persistent thread matching and lifecycle."""

import json
from datetime import date
from unittest.mock import AsyncMock

from nexus.engine.knowledge.events import Event
from nexus.engine.synthesis.threads import (
    compute_entity_overlap,
    compute_headline_similarity,
    compute_thread_similarity,
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
    """Events with high composite similarity match without LLM."""
    mock_llm = AsyncMock()
    events = [_event(summary="Iran IAEA sanctions update", entities=["IAEA", "Iran"])]
    threads = [_thread(headline="Iran IAEA Sanctions", entities=["IAEA", "Iran", "US"])]

    matches = await match_events_to_threads(mock_llm, events, threads)
    assert len(matches) == 1
    assert matches[0].thread_slug == "t1"
    assert matches[0].is_new_thread is False
    # High composite — should not need LLM
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
        _event(summary="Iran IAEA sanctions announced", entities=["IAEA", "Iran"]),
        _event(summary="Iran IAEA responds to sanctions", entities=["Iran", "IAEA"]),
    ]
    threads = [_thread(headline="Iran IAEA Sanctions", entities=["IAEA", "Iran"])]

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


def test_stale_revives_with_new_events():
    """Stale threads should revive when new events arrive."""
    event_dates = [date(2026, 3, 9), date(2026, 3, 10)]
    assert promote_thread_status("stale", event_dates) == "active"


def test_resolved_stays_resolved():
    """Resolved threads don't revert."""
    event_dates = [date(2026, 3, 9)]
    assert promote_thread_status("resolved", event_dates) == "resolved"


def test_merged_stays_merged():
    """Merged threads remain terminal."""
    event_dates = [date(2026, 3, 9)]
    assert promote_thread_status("merged", event_dates) == "merged"


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


def test_check_staleness_merged_unaffected():
    """Merged threads stay merged regardless of event age."""
    assert check_staleness("merged", date(2026, 1, 1), reference_date=date(2026, 3, 14)) == "merged"


# ── Thread merge candidates ─────────────────────────────────────


from nexus.engine.synthesis.threads import find_merge_candidates


def _merge_thread(id, slug, headline, entities, significance=7, status="active"):
    return {
        "id": id,
        "slug": slug,
        "headline": headline,
        "key_entities": entities,
        "status": status,
        "significance": significance,
        "created_at": "2026-03-01",
    }


async def test_find_merge_candidates_high_overlap_auto():
    """High composite score should auto-merge without LLM."""
    threads = [
        _merge_thread(1, "iran-sanctions", "Iran IAEA Sanctions Escalation",
                      ["Iran", "IAEA", "US"], significance=8),
        _merge_thread(2, "iran-nuclear", "Iran IAEA Sanctions Response",
                      ["Iran", "IAEA", "EU"], significance=5),
    ]
    # Entity Jaccard = 2/4 = 0.5, headline shares iran/iaea/sanctions
    # Composite well above 0.5
    llm = AsyncMock()
    pairs = await find_merge_candidates(threads, llm)
    assert len(pairs) == 1
    keep_id, absorb_id = pairs[0]
    assert keep_id == 1  # higher significance
    assert absorb_id == 2
    llm.complete.assert_not_called()


async def test_find_merge_candidates_low_overlap_ignored():
    """Jaccard < 0.3 should produce no candidates."""
    threads = [
        _merge_thread(1, "iran-sanctions", "Iran Sanctions", ["Iran", "IAEA"]),
        _merge_thread(2, "ai-progress", "AI Progress", ["OpenAI", "Google", "Meta"]),
    ]
    # Jaccard = 0/5 = 0.0
    pairs = await find_merge_candidates(threads)
    assert pairs == []


async def test_find_merge_candidates_ambiguous_llm_confirms():
    """Jaccard 0.3-0.5 + LLM says same arc → merge."""
    threads = [
        _merge_thread(1, "iran-talks", "Iran Talks", ["Iran", "US", "IAEA"], significance=8),
        _merge_thread(2, "nuclear-deal", "Nuclear Deal", ["Iran", "EU", "Russia", "IAEA"], significance=5),
    ]
    # Jaccard = |{iran, iaea}| / |{iran, us, iaea, eu, russia}| = 2/5 = 0.4
    llm = AsyncMock()
    llm.complete = AsyncMock(return_value=json.dumps({"pairs": [{"thread_a": 1, "thread_b": 2, "same_arc": True}]}))
    pairs = await find_merge_candidates(threads, llm)
    assert len(pairs) == 1
    assert pairs[0] == (1, 2)
    llm.complete.assert_called_once()


async def test_find_merge_candidates_ambiguous_llm_rejects():
    """Jaccard 0.3-0.5 + LLM says different → no merge."""
    threads = [
        _merge_thread(1, "iran-talks", "Iran Talks", ["Iran", "US", "IAEA"], significance=8),
        _merge_thread(2, "nuclear-deal", "Nuclear Deal", ["Iran", "EU", "Russia", "IAEA"], significance=5),
    ]
    llm = AsyncMock()
    llm.complete = AsyncMock(return_value=json.dumps({"pairs": [{"thread_a": 1, "thread_b": 2, "same_arc": False}]}))
    pairs = await find_merge_candidates(threads, llm)
    assert pairs == []


async def test_find_merge_candidates_keeps_higher_significance():
    """The thread with higher significance should be the keep_id."""
    threads = [
        _merge_thread(1, "thread-a", "Thread A", ["Iran", "IAEA"], significance=3),
        _merge_thread(2, "thread-b", "Thread B", ["Iran", "IAEA"], significance=9),
    ]
    pairs = await find_merge_candidates(threads)
    assert len(pairs) == 1
    assert pairs[0] == (2, 1)  # thread 2 kept (higher significance)


async def test_find_merge_candidates_chain_safe():
    """If A merges with B and B merges with C, A should absorb both."""
    threads = [
        _merge_thread(1, "a", "Thread A", ["Iran", "IAEA", "US"], significance=9),
        _merge_thread(2, "b", "Thread B", ["Iran", "IAEA", "EU"], significance=5),
        _merge_thread(3, "c", "Thread C", ["Iran", "IAEA", "Russia"], significance=3),
    ]
    # A-B: Jaccard = 2/4 = 0.5, B-C: Jaccard = 2/4 = 0.5, A-C: 2/4 = 0.5
    # All should merge into thread 1 (highest significance)
    pairs = await find_merge_candidates(threads)
    # After transitive resolution, both 2 and 3 should point to 1
    keep_ids = {p[0] for p in pairs}
    absorb_ids = {p[1] for p in pairs}
    assert keep_ids == {1}
    assert absorb_ids == {2, 3}


# ── Headline similarity ───────────────────────────────────────


def test_headline_similarity_identical():
    assert compute_headline_similarity(
        "Strait of Hormuz Blockade", "Strait of Hormuz Blockade"
    ) == 1.0


def test_headline_similarity_similar():
    score = compute_headline_similarity(
        "Strait of Hormuz Blockade and Global Energy Crisis",
        "Strait of Hormuz Conflict Triggers Global Energy Shock",
    )
    # Shared tokens: strait, hormuz, global, energy (after stop word removal)
    assert score >= 0.4


def test_headline_similarity_different():
    score = compute_headline_similarity(
        "Iran Nuclear Deal Negotiations",
        "AI Regulation Framework Updates",
    )
    assert score == 0.0


def test_headline_similarity_stop_words_ignored():
    score = compute_headline_similarity(
        "The Crisis in the Middle East",
        "Crisis in Middle East",
    )
    assert score == 1.0


def test_headline_similarity_empty():
    assert compute_headline_similarity("", "") == 0.0
    assert compute_headline_similarity("the and of", "in a to") == 0.0


def test_headline_similarity_case_insensitive():
    assert compute_headline_similarity("IRAN CRISIS", "iran crisis") == 1.0


# ── Composite thread similarity ────────────────────────────────


def test_composite_similarity_entity_and_headline():
    a = _merge_thread(1, "s1", "Iran Sanctions Escalation", ["Iran", "US", "IAEA"])
    b = _merge_thread(2, "s2", "Iran Sanctions Response", ["Iran", "EU", "Russia", "IAEA"])
    score = compute_thread_similarity(a, b)
    # Entity Jaccard = 2/5 = 0.4, headline has shared tokens
    # Composite should be meaningful
    assert 0.3 < score < 0.8


def test_composite_similarity_no_entities():
    """Threads with empty entity lists should match on headline alone."""
    a = _merge_thread(1, "s1", "Strait of Hormuz Blockade and Energy Crisis", [])
    b = _merge_thread(2, "s2", "Strait of Hormuz Conflict and Energy Shock", [])
    score = compute_thread_similarity(a, b)
    assert score >= 0.4


def test_composite_similarity_no_headline_overlap():
    """Entity overlap should still contribute when headlines don't match."""
    a = _merge_thread(1, "s1", "Thread Alpha", ["Iran", "IAEA"])
    b = _merge_thread(2, "s2", "Thread Beta", ["Iran", "IAEA"])
    score = compute_thread_similarity(a, b)
    # Entity Jaccard = 1.0, headline overlap ~0 (only "thread" shared)
    assert score > 0.3


def test_composite_similarity_one_empty_entities():
    """When one thread has entities and the other doesn't, headline gets more weight."""
    a = _merge_thread(1, "s1", "Global Energy Crisis Update", ["Iran", "OPEC"])
    b = _merge_thread(2, "s2", "Global Energy Crisis Intensifies", [])
    score = compute_thread_similarity(a, b)
    assert score >= 0.3


# ── Merge candidates with composite scoring ────────────────────


async def test_merge_candidates_low_entity_high_headline():
    """Low entity overlap but high headline similarity should reach LLM review."""
    threads = [
        _merge_thread(1, "hormuz-1", "Strait of Hormuz Blockade and Global Energy Crisis",
                      ["Iran", "IRGC", "Kuwait", "Saudi Arabia"] + [f"ent{i}" for i in range(20)],
                      significance=9),
        _merge_thread(2, "hormuz-2", "Strait of Hormuz Conflict Triggers Global Energy Shock",
                      ["Iran", "US Navy", "IEA", "UK"] + [f"other{i}" for i in range(20)],
                      significance=8),
    ]
    # Entity Jaccard is very low (~0.04), but headline similarity is high
    llm = AsyncMock()
    llm.complete = AsyncMock(return_value=json.dumps(
        {"pairs": [{"thread_a": 1, "thread_b": 2, "same_arc": True}]}
    ))
    pairs = await find_merge_candidates(threads, llm)
    # Should have reached LLM and merged
    assert len(pairs) == 1
    llm.complete.assert_called_once()


async def test_merge_candidates_no_entities_headline_match():
    """Threads with 0 entities should still merge on headline similarity."""
    threads = [
        _merge_thread(1, "crisis-1", "Global Energy Crisis and Geopolitical Realignment", [],
                      significance=9),
        _merge_thread(2, "crisis-2", "Global Energy Crisis Intensifies", [],
                      significance=10),
    ]
    # Both have empty entities, but headlines overlap
    pairs = await find_merge_candidates(threads)
    assert len(pairs) == 1
    assert pairs[0] == (2, 1)  # thread 2 kept (higher significance)
