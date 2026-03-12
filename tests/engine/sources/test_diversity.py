"""Tests for source diversity scoring."""

from nexus.engine.sources.diversity import (
    compute_diversity, suggest_improvements, _shannon_entropy,
)


def test_shannon_entropy_uniform():
    """Uniform distribution maximizes entropy."""
    counts = {"a": 10, "b": 10, "c": 10}
    entropy = _shannon_entropy(counts)
    assert entropy > 1.5  # log2(3) ≈ 1.585


def test_shannon_entropy_single():
    """Single category has zero entropy."""
    counts = {"a": 10}
    assert _shannon_entropy(counts) == 0.0


def test_shannon_entropy_empty():
    assert _shannon_entropy({}) == 0.0


def test_compute_diversity_balanced():
    """Balanced feed set scores high."""
    feeds = [
        {"country": "US", "affiliation": "private", "language": "en"},
        {"country": "GB", "affiliation": "public", "language": "en"},
        {"country": "DE", "affiliation": "public", "language": "de"},
        {"country": "CN", "affiliation": "state", "language": "zh"},
        {"country": "JP", "affiliation": "public", "language": "ja"},
    ]
    m = compute_diversity(feeds)
    assert m.geographic_score > 1.5
    assert m.affiliation_score > 1.0
    assert m.language_score > 1.0
    assert m.overall > 1.0
    assert len(m.warnings) == 0


def test_compute_diversity_homogeneous():
    """All same country triggers warning."""
    feeds = [
        {"country": "US", "affiliation": "private", "language": "en"},
        {"country": "US", "affiliation": "private", "language": "en"},
        {"country": "US", "affiliation": "private", "language": "en"},
        {"country": "US", "affiliation": "private", "language": "en"},
        {"country": "US", "affiliation": "private", "language": "en"},
    ]
    m = compute_diversity(feeds)
    assert m.geographic_score == 0.0
    assert any("Geographic concentration" in w for w in m.warnings)
    assert any("Single perspective" in w for w in m.warnings)


def test_compute_diversity_empty():
    m = compute_diversity([])
    assert m.overall == 0.0
    assert len(m.warnings) == 1


def test_compute_diversity_unknown_affiliation_warning():
    feeds = [
        {"country": "US", "affiliation": "unknown", "language": "en"},
        {"country": "GB", "affiliation": "unknown", "language": "en"},
        {"country": "DE", "affiliation": "unknown", "language": "en"},
        {"country": "FR", "affiliation": "private", "language": "en"},
        {"country": "JP", "affiliation": "unknown", "language": "en"},
    ]
    m = compute_diversity(feeds)
    assert any("unclassified" in w for w in m.warnings)


def test_suggest_improvements_fills_gaps():
    """Suggestions prioritize underrepresented countries/affiliations."""
    current = [
        {"url": "http://a.com", "country": "US", "affiliation": "private"},
        {"url": "http://b.com", "country": "US", "affiliation": "private"},
    ]
    global_sources = [
        {"url": "http://c.com", "country": "GB", "affiliation": "public", "tier": "A"},
        {"url": "http://d.com", "country": "US", "affiliation": "private", "tier": "A"},
        {"url": "http://e.com", "country": "DE", "affiliation": "state", "tier": "B"},
    ]
    suggestions = suggest_improvements(current, global_sources, max_suggestions=2)
    assert len(suggestions) == 2
    # GB and DE should be prioritized over another US/private
    suggested_countries = {s["country"] for s in suggestions}
    assert "US" not in suggested_countries


def test_suggest_improvements_excludes_existing():
    """Don't suggest feeds already in current set."""
    current = [{"url": "http://a.com", "country": "US", "affiliation": "private"}]
    global_sources = [{"url": "http://a.com", "country": "GB", "affiliation": "public", "tier": "A"}]
    suggestions = suggest_improvements(current, global_sources)
    assert len(suggestions) == 0
