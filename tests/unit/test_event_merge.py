"""Tests for event deduplication and merging."""

from datetime import date
from nexus.engine.knowledge.events import Event, is_duplicate_event, merge_events


def _event(d, summary, entities, sources=None, significance=5):
    return Event(
        date=d,
        summary=summary,
        entities=entities,
        sources=sources or [{"url": "https://example.com", "outlet": "test"}],
        significance=significance,
    )


def test_duplicate_same_entities_same_date():
    e1 = _event(date(2026, 3, 9), "Iran sanctions", ["Iran", "US", "EU"])
    e2 = _event(date(2026, 3, 9), "New Iran sanctions", ["Iran", "US", "EU"])
    assert is_duplicate_event(e2, e1) is True


def test_duplicate_partial_overlap():
    e1 = _event(date(2026, 3, 9), "Iran talks", ["Iran", "US", "EU", "IAEA"])
    e2 = _event(date(2026, 3, 9), "Iran US talks", ["Iran", "US", "Blinken"])
    # 2/3 overlap for e2 = 66% >= 60% threshold
    assert is_duplicate_event(e2, e1) is True


def test_not_duplicate_different_entities():
    e1 = _event(date(2026, 3, 9), "Iran sanctions", ["Iran", "US"])
    e2 = _event(date(2026, 3, 9), "China trade", ["China", "EU"])
    assert is_duplicate_event(e2, e1) is False


def test_not_duplicate_different_dates():
    e1 = _event(date(2026, 3, 7), "Iran talks", ["Iran", "US", "EU"])
    e2 = _event(date(2026, 3, 9), "Iran talks", ["Iran", "US", "EU"])
    assert is_duplicate_event(e2, e1) is False


def test_duplicate_adjacent_date():
    e1 = _event(date(2026, 3, 8), "Iran talks", ["Iran", "US", "EU"])
    e2 = _event(date(2026, 3, 9), "Iran talks continue", ["Iran", "US", "EU"])
    assert is_duplicate_event(e2, e1) is True


def test_not_duplicate_empty_entities():
    e1 = _event(date(2026, 3, 9), "Event 1", [])
    e2 = _event(date(2026, 3, 9), "Event 2", ["Iran"])
    assert is_duplicate_event(e2, e1) is False


def test_merge_combines_sources():
    e1 = _event(date(2026, 3, 9), "Iran talks", ["Iran", "US"],
                sources=[{"url": "https://bbc.com/1", "outlet": "bbc"}], significance=7)
    e2 = _event(date(2026, 3, 9), "Iran US talks", ["Iran", "US", "EU"],
                sources=[{"url": "https://aljazeera.com/1", "outlet": "aljazeera"}], significance=8)

    merged = merge_events(e1, e2)
    assert len(merged.sources) == 2
    assert merged.significance == 8  # Higher of the two
    assert "EU" in merged.entities
    assert len(merged.entities) == 3  # Iran, US, EU


def test_merge_deduplicates_sources():
    e1 = _event(date(2026, 3, 9), "Event", ["Iran"],
                sources=[{"url": "https://bbc.com/1", "outlet": "bbc"}])
    e2 = _event(date(2026, 3, 9), "Event", ["Iran"],
                sources=[{"url": "https://bbc.com/1", "outlet": "bbc"}])  # Same URL

    merged = merge_events(e1, e2)
    assert len(merged.sources) == 1


def test_merge_deduplicates_entities_case_insensitive():
    e1 = _event(date(2026, 3, 9), "Event", ["Iran", "US"])
    e2 = _event(date(2026, 3, 9), "Event", ["iran", "EU"])  # "iran" lowercase

    merged = merge_events(e1, e2)
    # Should have 3 unique entities (Iran, US, EU), not 4
    assert len(merged.entities) == 3
