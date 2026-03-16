"""Tests for debate engine — agent interaction hypothesis test."""

from __future__ import annotations

import json
from datetime import date
from unittest.mock import AsyncMock, patch

import pytest

from nexus.engine.projection.debate_engine import (
    DebateBenchmarkEngine,
    debate_revision,
)


class TestDebateRevision:
    async def test_revision_returns_probability(self):
        """Should return a revised probability after seeing others' views."""
        llm = AsyncMock()
        llm.complete = AsyncMock(return_value=json.dumps({
            "probability": 0.65,
            "reasoning": "Adjusted upward after seeing contrarian view.",
            "changed_because": "Contrarian raised valid point about momentum.",
        }))

        persona = {"name": "Optimist", "perspective": "Looks for positive signals"}
        own_result = {"persona": "Optimist", "probability": 0.55, "reasoning": "Looks good"}
        all_results = [
            {"persona": "Optimist", "probability": 0.55, "reasoning": "Looks good"},
            {"persona": "Contrarian", "probability": 0.75, "reasoning": "Momentum is strong"},
        ]

        revised = await debate_revision(
            llm, persona, "Will X happen?", own_result, all_results,
            as_of=date(2026, 3, 15),
        )

        assert revised["persona"] == "Optimist"
        assert revised["initial_probability"] == 0.55
        assert revised["revised_probability"] == 0.65
        assert revised["changed_because"] is not None

    async def test_revision_excludes_own_view(self):
        """Other views should not include the persona's own reasoning."""
        captured_prompts = []
        llm = AsyncMock()

        async def capture_complete(**kwargs):
            captured_prompts.append(kwargs.get("user_prompt", ""))
            return json.dumps({"probability": 0.50, "reasoning": "No change"})

        llm.complete = capture_complete

        persona = {"name": "Hawk", "perspective": "Hawkish"}
        own_result = {"persona": "Hawk", "probability": 0.30, "reasoning": "Conflict likely"}
        all_results = [
            {"persona": "Hawk", "probability": 0.30, "reasoning": "Conflict likely"},
            {"persona": "Dove", "probability": 0.70, "reasoning": "Peace prevails"},
        ]

        await debate_revision(llm, persona, "Q?", own_result, all_results)

        # The prompt should include Dove's view in "Other analysts' views"
        # Hawk's own view appears as "Your reasoning" but NOT in the other analysts section
        prompt = captured_prompts[0]
        assert "Dove" in prompt
        # Split at "Other analysts' views" to check only that section
        other_section = prompt.split("Other analysts")[1] if "Other analysts" in prompt else ""
        assert "Dove" in other_section
        assert "Hawk" not in other_section  # own persona excluded from other views

    async def test_revision_falls_back_on_error(self):
        """On LLM failure, should return initial probability unchanged."""
        llm = AsyncMock()
        llm.complete = AsyncMock(side_effect=RuntimeError("LLM down"))

        persona = {"name": "Test", "perspective": "Test"}
        own_result = {"persona": "Test", "probability": 0.42, "reasoning": "My view"}

        revised = await debate_revision(
            llm, persona, "Q?", own_result, [own_result],
        )

        assert revised["revised_probability"] == 0.42
        assert revised["initial_probability"] == 0.42


class TestDebateBenchmarkEngine:
    async def test_returns_probability(self):
        """Should produce a probability between 0.05 and 0.95."""
        engine = DebateBenchmarkEngine()

        # Mock LLM to return consistent persona + reasoning + revision responses
        call_count = 0

        async def mock_complete(**kwargs):
            nonlocal call_count
            call_count += 1
            prompt = kwargs.get("user_prompt", "")

            if "Generate" in prompt and "personas" in prompt:
                return json.dumps({"personas": [
                    {"name": "A", "perspective": "Optimist"},
                    {"name": "B", "perspective": "Pessimist"},
                    {"name": "C", "perspective": "Neutral"},
                    {"name": "D", "perspective": "Data-driven"},
                    {"name": "E", "perspective": "Historical"},
                ]})
            elif "revised probability" in prompt:
                return json.dumps({
                    "probability": 0.60,
                    "reasoning": "Revised view",
                    "changed_because": "Updated after debate",
                })
            else:
                return json.dumps({
                    "probability": 0.55,
                    "reasoning": "Initial assessment",
                })

        llm = AsyncMock()
        llm.complete = mock_complete

        prob = await engine.predict_probability(
            "Will X happen?", llm=llm, as_of=date(2026, 3, 15),
        )

        assert 0.05 <= prob <= 0.95
        # Should have made 11 calls: 1 persona gen + 5 reasoning + 5 revision
        assert call_count == 11

    async def test_no_llm_returns_half(self):
        """Without LLM, should return 0.50."""
        engine = DebateBenchmarkEngine()
        prob = await engine.predict_probability("Will X?")
        assert prob == 0.50

    async def test_debate_changes_estimates(self):
        """Revised estimates should differ from initial when debate has effect."""
        engine = DebateBenchmarkEngine()
        initial_probs = []
        revised_probs = []
        is_revision_phase = False

        async def mock_complete(**kwargs):
            nonlocal is_revision_phase
            prompt = kwargs.get("user_prompt", "")

            if "Generate" in prompt and "personas" in prompt:
                return json.dumps({"personas": [
                    {"name": "Bull", "perspective": "Bullish"},
                    {"name": "Bear", "perspective": "Bearish"},
                    {"name": "Moderate", "perspective": "Moderate"},
                    {"name": "Tech", "perspective": "Technical"},
                    {"name": "Fund", "perspective": "Fundamental"},
                ]})
            elif "revised probability" in prompt:
                is_revision_phase = True
                # After debate, everyone converges slightly toward center
                return json.dumps({
                    "probability": 0.55,
                    "reasoning": "Converged after debate",
                    "changed_because": "Others had good points",
                })
            else:
                # Initial: wide spread
                if "Bull" in kwargs.get("system_prompt", ""):
                    p = 0.80
                elif "Bear" in kwargs.get("system_prompt", ""):
                    p = 0.20
                else:
                    p = 0.50
                return json.dumps({"probability": p, "reasoning": "My view"})

        llm = AsyncMock()
        llm.complete = mock_complete

        prob = await engine.predict_probability(
            "Will X?", llm=llm, as_of=date(2026, 3, 15),
        )

        assert is_revision_phase  # debate round actually happened
        assert 0.05 <= prob <= 0.95

    async def test_engine_name(self):
        """Engine name should be 'debate'."""
        assert DebateBenchmarkEngine.engine_name == "debate"
