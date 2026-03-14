"""Tests for automated metrics computation."""

from datetime import date
from nexus.engine.evaluation.metrics import (
    source_diversity_index,
    convergence_ratio,
    independent_convergence_ratio,
    language_coverage,
    extraction_stats,
    event_dedup_ratio,
)
from nexus.engine.sources.polling import ContentItem
from nexus.engine.knowledge.events import Event


def _article(affiliation="private", lang="en", status="ok"):
    return ContentItem(
        title="Test", url="https://example.com", source_id="test",
        source_affiliation=affiliation, detected_language=lang,
        extraction_status=status,
    )


def _event(num_sources=1):
    return Event(
        date=date(2026, 3, 9), summary="Test event", significance=5,
        sources=[{"url": f"https://s{i}.com", "outlet": f"s{i}"} for i in range(num_sources)],
    )


def test_diversity_index_uniform():
    """Equal distribution = maximum entropy."""
    articles = [_article("private"), _article("state"), _article("public"), _article("nonprofit")]
    idx = source_diversity_index(articles)
    assert idx == 2.0  # log2(4) = 2.0 for uniform distribution


def test_diversity_index_single():
    """Single affiliation = zero entropy."""
    articles = [_article("private")] * 5
    assert source_diversity_index(articles) == 0.0


def test_diversity_index_empty():
    assert source_diversity_index([]) == 0.0


def test_convergence_ratio_all_multi():
    events = [_event(num_sources=2), _event(num_sources=3)]
    assert convergence_ratio(events) == 1.0


def test_convergence_ratio_none():
    events = [_event(num_sources=1), _event(num_sources=1)]
    assert convergence_ratio(events) == 0.0


def test_convergence_ratio_mixed():
    events = [_event(num_sources=2), _event(num_sources=1)]
    assert convergence_ratio(events) == 0.5


def test_language_coverage():
    articles = [_article(lang="en"), _article(lang="en"), _article(lang="fa")]
    cov = language_coverage(articles)
    assert cov["en"] == 0.667
    assert cov["fa"] == 0.333


def test_extraction_stats():
    articles = [_article(status="ok"), _article(status="ok"), _article(status="paywall")]
    stats = extraction_stats(articles)
    assert stats["ok"] == 2
    assert stats["paywall"] == 1


def test_event_dedup_ratio():
    assert event_dedup_ratio(10, 7) == 0.7
    assert event_dedup_ratio(0, 0) == 1.0


def test_independent_convergence_ratio():
    """Only counts events where at least one source pair is independent."""
    independent_event = Event(
        date=date(2026, 3, 9), summary="Test", significance=5,
        sources=[
            {"url": "https://a.com", "outlet": "nyt", "affiliation": "private", "country": "US"},
            {"url": "https://b.com", "outlet": "tass", "affiliation": "state", "country": "RU"},
        ],
    )
    non_independent_event = Event(
        date=date(2026, 3, 9), summary="Test", significance=5,
        sources=[
            {"url": "https://c.com", "outlet": "cgtn", "affiliation": "state", "country": "CN"},
            {"url": "https://d.com", "outlet": "xinhua", "affiliation": "state", "country": "CN"},
        ],
    )
    single_source = Event(
        date=date(2026, 3, 9), summary="Test", significance=5,
        sources=[{"url": "https://e.com", "outlet": "bbc", "affiliation": "public", "country": "GB"}],
    )

    assert independent_convergence_ratio([independent_event, non_independent_event, single_source]) == 0.333
    assert independent_convergence_ratio([independent_event]) == 1.0
    assert independent_convergence_ratio([non_independent_event]) == 0.0
    assert independent_convergence_ratio([]) == 0.0
