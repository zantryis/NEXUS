"""Tests for perspective-aware diversity selection."""

from nexus.config.models import TopicConfig
from nexus.engine.sources.polling import ContentItem
from nexus.engine.filtering.filter import apply_perspective_diversity


def _item(title, affiliation, score):
    return ContentItem(
        title=title,
        url=f"https://example.com/{title.lower().replace(' ', '-')}",
        source_id="test",
        source_affiliation=affiliation,
        relevance_score=score,
    )


def test_low_diversity_pure_score_ranking():
    topic = TopicConfig(name="AI", subtopics=[], perspective_diversity="low")
    items = [
        _item("A", "private", 9),
        _item("B", "state", 7),
        _item("C", "private", 8),
    ]
    result = apply_perspective_diversity(items, topic, max_items=2)
    assert len(result) == 2
    assert result[0].title == "A"
    assert result[1].title == "C"


def test_high_diversity_guarantees_representation():
    topic = TopicConfig(name="Iran-US", subtopics=[], perspective_diversity="high")
    items = [
        _item("Western 1", "private", 9),
        _item("Western 2", "private", 8),
        _item("Western 3", "private", 7),
        _item("Western 4", "private", 6),
        _item("State 1", "state", 5),
        _item("State 2", "state", 4),
        _item("Public 1", "public", 3),
    ]
    result = apply_perspective_diversity(items, topic, max_items=5)

    # Should guarantee at least 1 from each type (20% of 5 = 1)
    affiliations = [r.source_affiliation for r in result]
    assert "state" in affiliations
    assert "public" in affiliations
    assert "private" in affiliations


def test_medium_diversity_softer_constraint():
    topic = TopicConfig(name="Energy", subtopics=[], perspective_diversity="medium")
    items = [
        _item("Private 1", "private", 9),
        _item("Private 2", "private", 8),
        _item("State 1", "state", 3),
    ]
    result = apply_perspective_diversity(items, topic, max_items=10)

    affiliations = [r.source_affiliation for r in result]
    assert "state" in affiliations  # Guaranteed minimum


def test_single_affiliation_no_constraint():
    topic = TopicConfig(name="AI", subtopics=[], perspective_diversity="high")
    items = [_item(f"A{i}", "private", 10 - i) for i in range(5)]
    result = apply_perspective_diversity(items, topic, max_items=3)
    assert len(result) == 3
    # Just score-ranked since only one affiliation
    assert result[0].title == "A0"


def test_empty_input():
    topic = TopicConfig(name="AI", subtopics=[], perspective_diversity="high")
    result = apply_perspective_diversity([], topic)
    assert result == []
