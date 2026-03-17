"""Tests for multi-perspective prediction engine — MiroFish-inspired debate + aggregation."""

from __future__ import annotations

import json
from datetime import date
from unittest.mock import AsyncMock

import pytest

from nexus.engine.projection.perspective_engine import (
    PerspectiveBenchmarkEngine,
    PerspectiveForecastEngine,
    generate_personas,
    reason_as_persona,
)


def _make_mock_llm(responses: list[dict | str] | dict | str):
    """Mock LLM that returns sequential responses."""
    if isinstance(responses, (dict, str)):
        responses = [responses]

    call_count = [0]

    async def mock_complete(*, config_key, system_prompt, user_prompt, json_response=False):
        idx = min(call_count[0], len(responses) - 1)
        call_count[0] += 1
        resp = responses[idx]
        return json.dumps(resp) if isinstance(resp, dict) else resp

    llm = AsyncMock()
    llm.complete = mock_complete
    return llm


class TestGeneratePersonas:
    async def test_generates_personas(self):
        """Should generate a list of analyst personas."""
        llm = _make_mock_llm({
            "personas": [
                {"name": "Hawkish Analyst", "perspective": "Focuses on conflict escalation"},
                {"name": "Dovish Analyst", "perspective": "Focuses on diplomatic resolution"},
                {"name": "Data-Driven Analyst", "perspective": "Focuses on economic indicators"},
            ]
        })
        personas = await generate_personas(
            llm, "Will Israel and Saudi Arabia normalize?", num_personas=3
        )
        assert len(personas) == 3
        assert all("name" in p for p in personas)
        assert all("perspective" in p for p in personas)

    async def test_generates_diverse_perspectives(self):
        """Personas should have distinct perspectives."""
        llm = _make_mock_llm({
            "personas": [
                {"name": "Optimist", "perspective": "Sees positive momentum"},
                {"name": "Skeptic", "perspective": "Doubts diplomatic progress"},
                {"name": "Historian", "perspective": "Compares to past attempts"},
            ]
        })
        personas = await generate_personas(llm, "Question?", num_personas=3)
        perspectives = [p["perspective"] for p in personas]
        assert len(set(perspectives)) == len(perspectives)  # all unique

    async def test_handles_llm_failure(self):
        """Should return default personas on LLM failure."""
        async def failing_complete(**kwargs):
            raise RuntimeError("LLM error")

        llm = AsyncMock()
        llm.complete = failing_complete

        personas = await generate_personas(llm, "Question?", num_personas=3)
        assert len(personas) >= 2  # should have fallback personas


class TestReasonAsPersona:
    async def test_returns_probability(self):
        """Each persona should return a probability."""
        llm = _make_mock_llm({
            "probability": 0.65,
            "reasoning": "Based on my hawkish perspective...",
        })
        persona = {"name": "Hawk", "perspective": "Conflict-focused"}
        result = await reason_as_persona(
            llm, persona, "Will war happen?",
            knowledge_context="Recent tensions escalated.",
            as_of=date(2026, 3, 10),
        )
        assert 0.0 <= result["probability"] <= 1.0

    async def test_includes_reasoning(self):
        """Result should include reasoning text."""
        llm = _make_mock_llm({
            "probability": 0.40,
            "reasoning": "Historical patterns suggest...",
        })
        persona = {"name": "Historian", "perspective": "Historical patterns"}
        result = await reason_as_persona(
            llm, persona, "Will X happen?",
            as_of=date(2026, 3, 10),
        )
        assert "reasoning" in result
        assert len(result["reasoning"]) > 0


class TestPerspectiveBenchmarkEngine:
    async def test_returns_bounded_probability(self):
        """Should return a probability in [0.05, 0.95]."""
        # Persona generation + 3 persona reasoning + optional synthesis
        responses = [
            {"personas": [
                {"name": "A", "perspective": "Optimistic"},
                {"name": "B", "perspective": "Pessimistic"},
                {"name": "C", "perspective": "Neutral"},
            ]},
            {"probability": 0.70, "reasoning": "Optimistic view"},
            {"probability": 0.30, "reasoning": "Pessimistic view"},
            {"probability": 0.50, "reasoning": "Neutral view"},
        ]
        llm = _make_mock_llm(responses)
        engine = PerspectiveBenchmarkEngine()
        prob = await engine.predict_probability(
            "Will X happen?", llm=llm, as_of=date(2026, 3, 10)
        )
        assert 0.05 <= prob <= 0.95

    async def test_aggregates_via_geometric_mean(self):
        """Should aggregate persona probabilities via geometric_mean_of_odds."""
        responses = [
            {"personas": [
                {"name": "A", "perspective": "High"},
                {"name": "B", "perspective": "Low"},
            ]},
            {"probability": 0.80, "reasoning": "High"},
            {"probability": 0.20, "reasoning": "Low"},
        ]
        llm = _make_mock_llm(responses)
        engine = PerspectiveBenchmarkEngine()
        prob = await engine.predict_probability(
            "Will X happen?", llm=llm, as_of=date(2026, 3, 10)
        )
        # geometric_mean_of_odds([0.80, 0.20]) ≈ 0.50 (opposing views cancel)
        assert 0.35 <= prob <= 0.65

    async def test_engine_name(self):
        assert PerspectiveBenchmarkEngine().engine_name == "perspective"

    async def test_handles_llm_failure(self):
        """Should return 0.50 if LLM fails."""
        async def failing_complete(**kwargs):
            raise RuntimeError("LLM error")

        llm = AsyncMock()
        llm.complete = failing_complete

        engine = PerspectiveBenchmarkEngine()
        prob = await engine.predict_probability(
            "Will X happen?", llm=llm, as_of=date(2026, 3, 10)
        )
        assert prob == pytest.approx(0.50, abs=0.05)


class TestPerspectiveForecastEngine:
    def test_engine_name(self):
        assert PerspectiveForecastEngine().engine_name == "perspective"
