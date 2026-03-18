"""Tests for pairwise comparison engine (hybrid funnel filtering)."""

import json
import pytest
from unittest.mock import AsyncMock

from nexus.config.models import TopicConfig
from nexus.engine.sources.polling import ContentItem


# ── Helpers ──────────────────────────────────────────────────────


def _make_item(title: str, score: float | None = None, **kwargs) -> ContentItem:
    defaults = dict(
        title=title,
        url=f"https://example.com/{title.lower().replace(' ', '-')}",
        source_id="test-feed",
        snippet=f"Snippet about {title}",
        full_text=f"Full text about {title}",
    )
    defaults.update(kwargs)
    item = ContentItem(**defaults)
    if score is not None:
        item.relevance_score = score
    return item


def _make_topic(**kwargs) -> TopicConfig:
    defaults = dict(name="AI Research", subtopics=["agents", "LLMs"])
    defaults.update(kwargs)
    return TopicConfig(**defaults)


# ── detect_degenerate ────────────────────────────────────────────


class TestDetectDegenerate:
    """Tests for degenerate score distribution detection."""

    def test_normal_bimodal_not_degenerate(self):
        """Bimodal distribution (many 1s and 8-9s) is NOT degenerate."""
        from nexus.engine.filtering.pairwise import detect_degenerate

        # Mimics real data: ~37% score 1-2, ~46% score 7+
        scores = [1.0] * 37 + [5.0] * 17 + [9.0] * 46
        assert detect_degenerate(scores) is False

    def test_collapsed_distribution_is_degenerate(self):
        """All scores in a 2-point band → degenerate."""
        from nexus.engine.filtering.pairwise import detect_degenerate

        # Everything scores 3-4 (the "nothing is 7+" problem)
        scores = [3.0] * 40 + [4.0] * 55 + [5.0] * 5
        assert detect_degenerate(scores) is True

    def test_uniform_distribution_not_degenerate(self):
        """Scores spread across range → not degenerate."""
        from nexus.engine.filtering.pairwise import detect_degenerate

        scores = [float(i) for i in range(1, 11)] * 10
        assert detect_degenerate(scores) is False

    def test_empty_scores(self):
        """Empty list → not degenerate (nothing to process)."""
        from nexus.engine.filtering.pairwise import detect_degenerate

        assert detect_degenerate([]) is False

    def test_custom_threshold(self):
        """Custom threshold changes sensitivity."""
        from nexus.engine.filtering.pairwise import detect_degenerate

        scores = [4.0] * 60 + [8.0] * 40  # 60% in band 4-5, rest far away
        assert detect_degenerate(scores, threshold=0.5) is True
        assert detect_degenerate(scores, threshold=0.8) is False

    def test_single_value_degenerate(self):
        """All same score → degenerate."""
        from nexus.engine.filtering.pairwise import detect_degenerate

        scores = [5.0] * 100
        assert detect_degenerate(scores) is True


# ── select_references ────────────────────────────────────────────


class TestSelectReferences:
    """Tests for reference item selection from KEEP pool."""

    def test_selects_n_items(self):
        from nexus.engine.filtering.pairwise import select_references

        keep = [_make_item(f"Article {i}", score=8.0) for i in range(10)]
        refs = select_references(keep, n=3)
        assert len(refs) == 3

    def test_returns_all_when_pool_smaller_than_n(self):
        from nexus.engine.filtering.pairwise import select_references

        keep = [_make_item("Only One", score=9.0)]
        refs = select_references(keep, n=3)
        assert len(refs) == 1

    def test_prefers_source_diversity(self):
        """When items have different source affiliations, prefer diversity."""
        from nexus.engine.filtering.pairwise import select_references

        items = [
            _make_item("A1", score=8.0, source_affiliation="private", source_id="nyt"),
            _make_item("A2", score=8.0, source_affiliation="private", source_id="bbc"),
            _make_item("A3", score=8.0, source_affiliation="state", source_id="tass"),
            _make_item("A4", score=8.0, source_affiliation="academic", source_id="arxiv"),
            _make_item("A5", score=8.0, source_affiliation="public", source_id="pbs"),
        ]
        refs = select_references(items, n=3)
        affiliations = {r.source_affiliation for r in refs}
        # Should pick from different affiliations when possible
        assert len(affiliations) >= 2

    def test_empty_pool(self):
        from nexus.engine.filtering.pairwise import select_references

        refs = select_references([], n=3)
        assert refs == []


# ── compare_batch ────────────────────────────────────────────────


class TestCompareBatch:
    """Tests for LLM-based pairwise comparison."""

    @pytest.mark.asyncio
    async def test_basic_comparison(self):
        """MAYBE items compared against references, winners promoted."""
        from nexus.engine.filtering.pairwise import compare_batch

        maybe = [_make_item("Maybe Article", score=5.0)]
        refs = [
            _make_item("Strong Ref 1", score=9.0),
            _make_item("Strong Ref 2", score=8.0),
            _make_item("Strong Ref 3", score=8.0),
        ]
        topic = _make_topic()

        # LLM says MAYBE wins 2 out of 3 comparisons
        mock_llm = AsyncMock()
        mock_llm.complete.side_effect = [
            json.dumps([{"pair": 1, "winner": "A", "reason": "More specific"}]),
            json.dumps([{"pair": 1, "winner": "A", "reason": "Reports new finding"}]),
            json.dumps([{"pair": 1, "winner": "B", "reason": "Ref is stronger"}]),
        ]

        results = await compare_batch(mock_llm, maybe, refs, topic)
        assert len(results) == 1
        # 2 wins out of 3 → promoted
        assert results[0][0] is True  # promoted

    @pytest.mark.asyncio
    async def test_maybe_loses_majority(self):
        """MAYBE loses 2+ comparisons → not promoted."""
        from nexus.engine.filtering.pairwise import compare_batch

        maybe = [_make_item("Weak Maybe", score=4.0)]
        refs = [
            _make_item("Ref 1", score=9.0),
            _make_item("Ref 2", score=8.0),
            _make_item("Ref 3", score=8.0),
        ]
        topic = _make_topic()

        mock_llm = AsyncMock()
        mock_llm.complete.side_effect = [
            json.dumps([{"pair": 1, "winner": "B", "reason": "Ref better"}]),
            json.dumps([{"pair": 1, "winner": "B", "reason": "Ref better"}]),
            json.dumps([{"pair": 1, "winner": "A", "reason": "Maybe wins"}]),
        ]

        results = await compare_batch(mock_llm, maybe, refs, topic)
        assert len(results) == 1
        assert results[0][0] is False  # not promoted

    @pytest.mark.asyncio
    async def test_ties_count_as_wins(self):
        """Ties count in favor of the MAYBE item (benefit of the doubt)."""
        from nexus.engine.filtering.pairwise import compare_batch

        maybe = [_make_item("Borderline", score=5.0)]
        refs = [
            _make_item("Ref 1", score=8.0),
            _make_item("Ref 2", score=8.0),
            _make_item("Ref 3", score=8.0),
        ]
        topic = _make_topic()

        mock_llm = AsyncMock()
        mock_llm.complete.side_effect = [
            json.dumps([{"pair": 1, "winner": "tie", "reason": "Equal"}]),
            json.dumps([{"pair": 1, "winner": "tie", "reason": "Equal"}]),
            json.dumps([{"pair": 1, "winner": "B", "reason": "Ref wins"}]),
        ]

        results = await compare_batch(mock_llm, maybe, refs, topic)
        assert results[0][0] is True  # 2 ties → promoted

    @pytest.mark.asyncio
    async def test_llm_parse_error_defaults_to_not_promoted(self):
        """If LLM returns garbage, MAYBE item is not promoted (conservative)."""
        from nexus.engine.filtering.pairwise import compare_batch

        maybe = [_make_item("Test", score=5.0)]
        refs = [_make_item("Ref", score=8.0)]
        topic = _make_topic()

        mock_llm = AsyncMock()
        mock_llm.complete.side_effect = [
            "not valid json at all",
        ]

        results = await compare_batch(mock_llm, maybe, refs, topic)
        assert len(results) == 1
        assert results[0][0] is False  # conservative fallback


# ── resolve_maybe_items ──────────────────────────────────────────


class TestResolveMaybeItems:
    """Tests for the orchestrator that resolves MAYBE items."""

    @pytest.mark.asyncio
    async def test_promotes_winning_items(self):
        """Items that win majority of comparisons are returned as promoted."""
        from nexus.engine.filtering.pairwise import resolve_maybe_items

        maybe = [_make_item("Good Maybe", score=5.0)]
        keep = [
            _make_item("Strong 1", score=9.0, source_affiliation="private"),
            _make_item("Strong 2", score=8.0, source_affiliation="state"),
            _make_item("Strong 3", score=8.0, source_affiliation="academic"),
        ]
        topic = _make_topic()

        mock_llm = AsyncMock()
        # Wins all 3 comparisons
        mock_llm.complete.side_effect = [
            json.dumps([{"pair": 1, "winner": "A", "reason": "Good"}]),
            json.dumps([{"pair": 1, "winner": "A", "reason": "Good"}]),
            json.dumps([{"pair": 1, "winner": "A", "reason": "Good"}]),
        ]

        promoted = await resolve_maybe_items(mock_llm, maybe, keep, topic)
        assert len(promoted) == 1
        assert promoted[0].title == "Good Maybe"

    @pytest.mark.asyncio
    async def test_empty_maybe_returns_empty(self):
        """No MAYBE items → no work, return empty."""
        from nexus.engine.filtering.pairwise import resolve_maybe_items

        promoted = await resolve_maybe_items(
            AsyncMock(), [], [_make_item("Ref", score=9.0)], _make_topic()
        )
        assert promoted == []

    @pytest.mark.asyncio
    async def test_empty_keep_pool_returns_all_maybe(self):
        """No KEEP items means we can't compare → return all MAYBE items."""
        from nexus.engine.filtering.pairwise import resolve_maybe_items

        maybe = [_make_item("Only Maybe", score=5.0)]
        promoted = await resolve_maybe_items(
            AsyncMock(), maybe, [], _make_topic()
        )
        assert len(promoted) == 1


# ── Triage Splitting ─────────────────────────────────────────────


class TestTriageSplit:
    """Tests for splitting scored items into KEEP/MAYBE/DROP."""

    def test_basic_split(self):
        from nexus.engine.filtering.pairwise import triage_split

        items = [
            _make_item("High", score=9.0),
            _make_item("Mid", score=5.0),
            _make_item("Low", score=1.0),
        ]
        keep, maybe, drop = triage_split(items)
        assert len(keep) == 1
        assert keep[0].title == "High"
        assert len(maybe) == 1
        assert maybe[0].title == "Mid"
        assert len(drop) == 1
        assert drop[0].title == "Low"

    def test_boundary_scores(self):
        """Score 7 → KEEP, score 3 → MAYBE, score 2 → DROP."""
        from nexus.engine.filtering.pairwise import triage_split

        items = [
            _make_item("Seven", score=7.0),
            _make_item("Three", score=3.0),
            _make_item("Two", score=2.0),
        ]
        keep, maybe, drop = triage_split(items)
        assert len(keep) == 1
        assert len(maybe) == 1
        assert len(drop) == 1

    def test_none_scores_go_to_maybe(self):
        """Items with no score go to MAYBE (unscored)."""
        from nexus.engine.filtering.pairwise import triage_split

        items = [_make_item("Unscored")]
        keep, maybe, drop = triage_split(items)
        assert len(maybe) == 1
        assert len(keep) == 0
        assert len(drop) == 0
