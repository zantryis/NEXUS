"""Tests for LLM-based projection evaluation judge."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from unittest.mock import AsyncMock

import pytest

from nexus.engine.projection.evaluation import (
    ProjectionJudgment,
    EventVerdict,
    judge_projection_item,
    evaluate_projection_item,
)


@dataclass
class FakeEvent:
    event_id: int
    date: date
    summary: str
    significance: int = 5
    entities: list = field(default_factory=list)
    sources: list = field(default_factory=list)
    relation_to_prior: str = ""


def _mock_llm(response_json: dict) -> AsyncMock:
    """Build a mock LLM client that returns the given JSON."""
    llm = AsyncMock()
    llm.complete = AsyncMock(return_value=json.dumps(response_json))
    return llm


# ── Core scoring logic ──


class TestJudgeProjectionItem:
    """LLM judge correctly classifies events and computes aggregate score."""

    async def test_all_confirms_returns_high_score(self):
        """When all events confirm the claim, score should be 1.0."""
        events = [
            FakeEvent(1, date(2026, 3, 10), "Russia withdraws troops from border"),
            FakeEvent(2, date(2026, 3, 12), "Kremlin announces troop pullback"),
        ]
        llm = _mock_llm({"verdicts": [
            {"event_index": 0, "verdict": "confirms", "rationale": "Direct troop withdrawal"},
            {"event_index": 1, "verdict": "confirms", "rationale": "Official confirmation"},
        ]})
        result = await judge_projection_item(
            "Russia will pull back troops from the Ukrainian border",
            "Official announcement of troop withdrawal",
            events, llm,
        )
        assert isinstance(result, ProjectionJudgment)
        assert result.score == 1.0
        assert result.outcome_status == "hit"
        assert len(result.verdicts) == 2
        assert all(v.verdict == "confirms" for v in result.verdicts)

    async def test_all_contradicts_returns_zero(self):
        """When all events contradict the claim, score should be 0.0."""
        events = [
            FakeEvent(1, date(2026, 3, 10), "Russia deploys additional 10,000 troops to border"),
        ]
        llm = _mock_llm({"verdicts": [
            {"event_index": 0, "verdict": "contradicts", "rationale": "More troops, not fewer"},
        ]})
        result = await judge_projection_item(
            "Russia will pull back troops from the Ukrainian border",
            "Official announcement of troop withdrawal",
            events, llm,
        )
        assert result.score == 0.0
        assert result.outcome_status == "miss"

    async def test_mixed_verdicts_returns_medium_score(self):
        """Mix of confirms and contradicts should yield intermediate score."""
        events = [
            FakeEvent(1, date(2026, 3, 10), "Russia pulls back 2,000 troops"),
            FakeEvent(2, date(2026, 3, 12), "Russia deploys new missile systems"),
        ]
        llm = _mock_llm({"verdicts": [
            {"event_index": 0, "verdict": "confirms", "rationale": "Partial pullback"},
            {"event_index": 1, "verdict": "contradicts", "rationale": "Escalation, not de-escalation"},
        ]})
        result = await judge_projection_item(
            "Russia will de-escalate military presence near Ukraine",
            "Troop withdrawal or weapons removal",
            events, llm,
        )
        assert result.score == 0.5
        assert result.outcome_status == "mixed"

    async def test_irrelevant_events_excluded_from_score(self):
        """Events classified as irrelevant should not affect the score."""
        events = [
            FakeEvent(1, date(2026, 3, 10), "Russia confirms troop withdrawal"),
            FakeEvent(2, date(2026, 3, 11), "Apple releases new iPhone"),
            FakeEvent(3, date(2026, 3, 12), "Weather forecast for Moscow"),
        ]
        llm = _mock_llm({"verdicts": [
            {"event_index": 0, "verdict": "confirms", "rationale": "Direct confirmation"},
            {"event_index": 1, "verdict": "irrelevant", "rationale": "Unrelated tech news"},
            {"event_index": 2, "verdict": "irrelevant", "rationale": "Weather, not geopolitics"},
        ]})
        result = await judge_projection_item(
            "Russia will pull back troops",
            "Troop withdrawal announcement",
            events, llm,
        )
        # Only 1 relevant event (confirms) → score = 1.0
        assert result.score == 1.0
        assert len(result.verdicts) == 3
        assert result.relevant_count == 1

    async def test_all_irrelevant_returns_zero(self):
        """If no events are relevant, score should be 0.0 (miss)."""
        events = [
            FakeEvent(1, date(2026, 3, 10), "Apple releases new iPhone"),
        ]
        llm = _mock_llm({"verdicts": [
            {"event_index": 0, "verdict": "irrelevant", "rationale": "Unrelated"},
        ]})
        result = await judge_projection_item(
            "Russia will pull back troops",
            "Troop withdrawal announcement",
            events, llm,
        )
        assert result.score == 0.0
        assert result.outcome_status == "miss"
        assert result.relevant_count == 0

    async def test_partially_confirms_scores_half(self):
        """partially_confirms should score 0.5 per event."""
        events = [
            FakeEvent(1, date(2026, 3, 10), "Russia pulls back some units but not all"),
        ]
        llm = _mock_llm({"verdicts": [
            {"event_index": 0, "verdict": "partially_confirms", "rationale": "Partial, not full"},
        ]})
        result = await judge_projection_item(
            "Russia will fully withdraw troops",
            "Complete withdrawal announced",
            events, llm,
        )
        assert result.score == 0.5
        assert result.outcome_status == "mixed"

    async def test_empty_events_returns_zero(self):
        """No future events → score 0.0, no LLM call needed."""
        llm = _mock_llm({})  # Should not be called
        result = await judge_projection_item(
            "Something will happen",
            "A signpost",
            [], llm,
        )
        assert result.score == 0.0
        assert result.outcome_status == "miss"
        llm.complete.assert_not_called()


# ── Error handling and fallback ──


class TestJudgeFallback:
    """LLM failures should fall back to keyword matching gracefully."""

    async def test_llm_json_parse_error_falls_back(self):
        """If LLM returns invalid JSON, fall back to keyword matching."""
        events = [
            FakeEvent(1, date(2026, 3, 10), "Iran nuclear talks resumed"),
        ]
        llm = AsyncMock()
        llm.complete = AsyncMock(return_value="This is not JSON at all")
        result = await judge_projection_item(
            "Iran will resume nuclear talks",
            "Talks begin again",
            events, llm,
        )
        assert isinstance(result, ProjectionJudgment)
        assert result.used_fallback is True
        # Keyword match should find "Iran", "nuclear", "talks" overlap
        assert result.score > 0.0

    async def test_llm_exception_falls_back(self):
        """If LLM call raises, fall back to keyword matching."""
        events = [
            FakeEvent(1, date(2026, 3, 10), "Iran nuclear talks resumed"),
        ]
        llm = AsyncMock()
        llm.complete = AsyncMock(side_effect=Exception("API timeout"))
        result = await judge_projection_item(
            "Iran will resume nuclear talks",
            "Talks begin again",
            events, llm,
        )
        assert isinstance(result, ProjectionJudgment)
        assert result.used_fallback is True

    async def test_llm_missing_verdicts_key_falls_back(self):
        """If LLM returns JSON but missing 'verdicts' key, fall back."""
        events = [
            FakeEvent(1, date(2026, 3, 10), "Iran nuclear talks resumed"),
        ]
        llm = _mock_llm({"wrong_key": "oops"})
        result = await judge_projection_item(
            "Iran will resume nuclear talks",
            "Talks begin again",
            events, llm,
        )
        assert result.used_fallback is True

    async def test_llm_partial_verdicts_handles_gracefully(self):
        """If LLM returns fewer verdicts than events, handle missing ones."""
        events = [
            FakeEvent(1, date(2026, 3, 10), "Iran nuclear talks resumed"),
            FakeEvent(2, date(2026, 3, 11), "Iran oil exports rise"),
        ]
        # Only 1 verdict for 2 events
        llm = _mock_llm({"verdicts": [
            {"event_index": 0, "verdict": "confirms", "rationale": "Talks resumed"},
        ]})
        result = await judge_projection_item(
            "Iran will resume nuclear talks",
            "Talks begin again",
            events, llm,
        )
        # Should still work — missing events treated as irrelevant
        assert isinstance(result, ProjectionJudgment)
        assert result.score > 0.0


# ── Prompt construction ──


class TestJudgePrompt:
    """The prompt sent to the LLM should contain claim, signpost, and events."""

    async def test_prompt_includes_claim_and_signpost(self):
        events = [
            FakeEvent(1, date(2026, 3, 10), "Iran nuclear talks resumed"),
        ]
        llm = _mock_llm({"verdicts": [
            {"event_index": 0, "verdict": "confirms", "rationale": "Match"},
        ]})
        await judge_projection_item(
            "Iran will resume nuclear talks",
            "Official diplomatic announcement",
            events, llm,
        )
        call_kwargs = llm.complete.call_args
        user_prompt = call_kwargs.kwargs.get("user_prompt", call_kwargs[1].get("user_prompt", ""))
        assert "Iran will resume nuclear talks" in user_prompt
        assert "Official diplomatic announcement" in user_prompt

    async def test_prompt_includes_event_summaries(self):
        events = [
            FakeEvent(1, date(2026, 3, 10), "Iran and US hold bilateral meeting"),
            FakeEvent(2, date(2026, 3, 12), "EU sanctions package announced"),
        ]
        llm = _mock_llm({"verdicts": [
            {"event_index": 0, "verdict": "confirms", "rationale": "Match"},
            {"event_index": 1, "verdict": "irrelevant", "rationale": "Unrelated"},
        ]})
        await judge_projection_item(
            "Iran will resume nuclear talks",
            "Talks begin",
            events, llm,
        )
        call_kwargs = llm.complete.call_args
        user_prompt = call_kwargs.kwargs.get("user_prompt", call_kwargs[1].get("user_prompt", ""))
        assert "Iran and US hold bilateral meeting" in user_prompt
        assert "EU sanctions package announced" in user_prompt

    async def test_uses_filtering_config_key(self):
        """Should use the cheap/fast 'filtering' model, not the expensive 'agent' model."""
        events = [FakeEvent(1, date(2026, 3, 10), "Something happened")]
        llm = _mock_llm({"verdicts": [
            {"event_index": 0, "verdict": "irrelevant", "rationale": "No match"},
        ]})
        await judge_projection_item("Claim", "Signpost", events, llm)
        call_kwargs = llm.complete.call_args
        config_key = call_kwargs.kwargs.get("config_key", call_kwargs[1].get("config_key", ""))
        assert config_key == "filtering"


# ── Negation handling (the key weakness of keyword matching) ──


class TestNegationHandling:
    """The LLM judge should correctly handle negation that keyword matching misses."""

    async def test_opposite_action_classified_as_contradicts(self):
        """'Russia withdraws' should contradict 'Russia enters' — keyword match would miss this."""
        events = [
            FakeEvent(1, date(2026, 3, 10), "Russia enters new round of talks with Ukraine"),
        ]
        # The LLM should recognize this contradicts withdrawal
        llm = _mock_llm({"verdicts": [
            {"event_index": 0, "verdict": "contradicts", "rationale": "Entering talks, not withdrawing"},
        ]})
        result = await judge_projection_item(
            "Russia will withdraw from peace negotiations",
            "Russia announces exit from talks",
            events, llm,
        )
        assert result.score == 0.0
        # Compare to keyword fallback which would incorrectly match on "Russia" and "talks"
        keyword_score = evaluate_projection_item(
            "Russia will withdraw from peace negotiations",
            "Russia announces exit from talks",
            events,
        )
        # Keyword match finds overlap → positive score (incorrect)
        assert keyword_score > 0.0, "Keyword baseline should incorrectly match here"


# ── Backward compatibility ──


class TestKeywordFallbackStillWorks:
    """The deterministic keyword evaluator should remain unchanged."""

    def test_keyword_match_basic(self):
        events = [FakeEvent(1, date(2026, 3, 10), "Iran nuclear talks resumed")]
        score = evaluate_projection_item("Iran nuclear negotiations", "Talks resume", events)
        assert score > 0.0

    def test_keyword_no_match(self):
        events = [FakeEvent(1, date(2026, 3, 10), "Apple releases new iPhone")]
        score = evaluate_projection_item("Iran nuclear negotiations", "Talks resume", events)
        assert score == 0.0

    def test_keyword_empty_events(self):
        score = evaluate_projection_item("Iran nuclear", "Talks", [])
        assert score == 0.0
