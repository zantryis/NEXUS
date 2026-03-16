"""Tests for naked LLM baseline engine — no knowledge context."""

from __future__ import annotations

import json
from datetime import date
from unittest.mock import AsyncMock

import pytest

from nexus.engine.projection.naked_engine import (
    NakedBenchmarkEngine,
    NakedForecastEngine,
)


def _make_mock_llm(probability: float):
    """Mock LLM that returns a fixed probability JSON."""
    response = json.dumps({"probability": probability})

    async def mock_complete(*, config_key, system_prompt, user_prompt, json_response=False):
        return response

    llm = AsyncMock()
    llm.complete = mock_complete
    return llm


class TestNakedBenchmarkEngine:
    async def test_returns_bounded_probability(self):
        """Should return a probability in [0.05, 0.95]."""
        llm = _make_mock_llm(0.99)
        engine = NakedBenchmarkEngine()
        prob = await engine.predict_probability(
            "Will X happen?", llm=llm, as_of=date(2026, 3, 10)
        )
        assert 0.05 <= prob <= 0.95

    async def test_returns_probability_from_llm(self):
        """Should return the LLM's probability (after calibration)."""
        llm = _make_mock_llm(0.60)
        engine = NakedBenchmarkEngine()
        prob = await engine.predict_probability(
            "Will Y happen?", llm=llm, as_of=date(2026, 3, 10)
        )
        # After extremize(0.60, gamma=0.8), result should be closer to 0.5
        assert 0.50 <= prob <= 0.60

    async def test_no_context_in_prompt(self):
        """Prompt should contain zero knowledge context."""
        captured_prompts = []

        async def spy_complete(*, config_key, system_prompt, user_prompt, json_response=False):
            captured_prompts.append(user_prompt)
            return json.dumps({"probability": 0.50})

        llm = AsyncMock()
        llm.complete = spy_complete

        engine = NakedBenchmarkEngine()
        await engine.predict_probability(
            "Will Z happen?", llm=llm, as_of=date(2026, 3, 10)
        )

        prompt = captured_prompts[0]
        # Should contain the question
        assert "Will Z happen?" in prompt
        # Should contain the date
        assert "2026-03-10" in prompt
        # Should NOT contain knowledge-related terms
        assert "event" not in prompt.lower() or "events" not in prompt.lower()
        assert "entity" not in prompt.lower()
        assert "relationship" not in prompt.lower()
        assert "thread" not in prompt.lower()

    async def test_applies_overconfidence_correction(self):
        """High probabilities should be compressed toward 0.5."""
        llm = _make_mock_llm(0.90)
        engine = NakedBenchmarkEngine()
        prob = await engine.predict_probability(
            "Will high-conf happen?", llm=llm, as_of=date(2026, 3, 10)
        )
        # extremize(0.90, gamma=0.8) compresses toward 0.5, so should be < 0.90
        assert prob < 0.90

    async def test_handles_llm_failure(self):
        """Should return 0.50 if LLM fails."""
        async def failing_complete(**kwargs):
            raise RuntimeError("LLM error")

        llm = AsyncMock()
        llm.complete = failing_complete

        engine = NakedBenchmarkEngine()
        prob = await engine.predict_probability(
            "Will X happen?", llm=llm, as_of=date(2026, 3, 10)
        )
        assert prob == pytest.approx(0.50, abs=0.01)

    async def test_engine_name(self):
        """Engine should identify itself."""
        engine = NakedBenchmarkEngine()
        assert engine.engine_name == "naked"

    async def test_works_without_llm(self):
        """Should return 0.50 when no LLM provided."""
        engine = NakedBenchmarkEngine()
        prob = await engine.predict_probability(
            "Will X happen?", as_of=date(2026, 3, 10)
        )
        assert prob == pytest.approx(0.50, abs=0.01)


class TestNakedForecastEngine:
    def test_engine_name(self):
        assert NakedForecastEngine().engine_name == "naked"
