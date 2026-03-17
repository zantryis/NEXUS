"""Tests for LLM-based claim generation and upgraded ActorForecastEngine."""

from __future__ import annotations

import json
from datetime import date
from unittest.mock import AsyncMock


from nexus.engine.knowledge.events import Event
from nexus.engine.projection.actor_engine import (
    ActorForecastEngine,
    generate_claims_from_context,
)
from nexus.engine.projection.forecasting import ForecastEngineInput
from nexus.engine.projection.models import ForecastRun
from nexus.engine.synthesis.knowledge import NarrativeThread


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_CLAIMS_JSON = json.dumps([
    {
        "claim": "Iran will impose new transit fees on non-allied vessels in the Strait of Hormuz by March 24",
        "reasoning": "Recent restrictions on US-affiliated vessels and shift to non-dollar oil trade suggest escalation.",
        "signpost": "Iranian maritime authority announcement or shipping industry advisory",
        "confidence": "medium",
        "horizon_days": 7,
        "source_thread_headline": "Iran Strait of Hormuz Restrictions",
    },
    {
        "claim": "The US will announce additional sanctions targeting Iranian oil exports within 14 days",
        "reasoning": "Congressional pressure and recent Iranian transit restrictions create momentum for new sanctions package.",
        "signpost": "Treasury Department OFAC designation or executive order",
        "confidence": "high",
        "horizon_days": 14,
        "source_thread_headline": "US-Iran Sanctions Escalation",
    },
])


def _make_threads() -> list[NarrativeThread]:
    return [
        NarrativeThread(
            headline="Iran Strait of Hormuz Restrictions",
            events=[
                Event(date=date(2026, 3, 10), summary="Iran restricts US-affiliated vessels in Strait", entities=["Iran", "US"]),
                Event(date=date(2026, 3, 14), summary="Iran shifts oil trade to non-dollar currencies", entities=["Iran"]),
            ],
            key_entities=["Iran", "Strait of Hormuz", "US"],
            significance=8,
            thread_id=101,
            slug="iran-strait-restrictions",
            trajectory_label="accelerating",
            momentum_score=5.0,
        ),
        NarrativeThread(
            headline="US-Iran Sanctions Escalation",
            events=[
                Event(date=date(2026, 3, 8), summary="Congress introduces new Iran sanctions bill", entities=["US", "Iran"]),
            ],
            key_entities=["US", "Iran", "Congress"],
            significance=7,
            thread_id=102,
            slug="us-iran-sanctions",
            trajectory_label="steady",
            momentum_score=2.0,
        ),
    ]


def _make_payload(threads=None, events=None) -> ForecastEngineInput:
    threads = threads or _make_threads()
    events = events or [e for t in threads for e in t.events]
    return ForecastEngineInput(
        topic_slug="iran-us-relations",
        topic_name="Iran-US Relations",
        run_date=date(2026, 3, 17),
        threads=threads,
        recent_events=events,
        cross_topic_signals=[],
    )


def _make_llm(response: str):
    """Create a mock LLM that returns a fixed string response."""
    async def mock_complete(*, config_key, system_prompt, user_prompt, json_response=False):
        return response

    llm = AsyncMock()
    llm.complete = mock_complete
    return llm


# ---------------------------------------------------------------------------
# generate_claims_from_context
# ---------------------------------------------------------------------------


class TestGenerateClaimsFromContext:
    async def test_returns_structured_claims(self):
        """Should parse LLM JSON into structured claim dicts."""
        llm = _make_llm(SAMPLE_CLAIMS_JSON)
        payload = _make_payload()

        claims = await generate_claims_from_context(llm, payload, max_claims=3)

        assert len(claims) == 2
        assert claims[0]["claim"] == "Iran will impose new transit fees on non-allied vessels in the Strait of Hormuz by March 24"
        assert claims[0]["confidence"] == "medium"
        assert claims[0]["horizon_days"] == 7
        assert claims[0]["signpost"] != ""
        assert claims[0]["reasoning"] != ""

    async def test_handles_llm_failure(self):
        """Should return empty list on LLM exception."""
        async def fail_complete(*, config_key, system_prompt, user_prompt, json_response=False):
            raise RuntimeError("LLM down")

        llm = AsyncMock()
        llm.complete = fail_complete
        payload = _make_payload()

        claims = await generate_claims_from_context(llm, payload)
        assert claims == []

    async def test_handles_malformed_json(self):
        """Should return empty list if LLM returns invalid JSON."""
        llm = _make_llm("This is not JSON at all, sorry!")
        payload = _make_payload()

        claims = await generate_claims_from_context(llm, payload)
        assert claims == []

    async def test_handles_json_with_markdown_fences(self):
        """Should handle LLM wrapping JSON in markdown code fences."""
        llm = _make_llm(f"```json\n{SAMPLE_CLAIMS_JSON}\n```")
        payload = _make_payload()

        claims = await generate_claims_from_context(llm, payload)
        assert len(claims) == 2

    async def test_caps_at_max_claims(self):
        """Should return at most max_claims items."""
        many_claims = json.dumps([
            {"claim": f"Claim {i}", "reasoning": "r", "signpost": "s",
             "confidence": "medium", "horizon_days": 7,
             "source_thread_headline": "Thread"}
            for i in range(10)
        ])
        llm = _make_llm(many_claims)
        payload = _make_payload()

        claims = await generate_claims_from_context(llm, payload, max_claims=3)
        assert len(claims) <= 3

    async def test_prompt_includes_thread_context(self):
        """Should include thread headlines, trajectories, and events in the prompt."""
        captured = []

        async def capture_complete(*, config_key, system_prompt, user_prompt, json_response=False):
            captured.append(user_prompt)
            return SAMPLE_CLAIMS_JSON

        llm = AsyncMock()
        llm.complete = capture_complete
        payload = _make_payload()

        await generate_claims_from_context(llm, payload)

        assert len(captured) == 1
        prompt = captured[0]
        assert "Iran Strait of Hormuz Restrictions" in prompt
        assert "accelerating" in prompt
        assert "Iran restricts US-affiliated vessels" in prompt

    async def test_uses_filtering_config_key(self):
        """Should use cheap model via config_key='filtering'."""
        captured_kwargs = []

        async def capture_complete(*, config_key, system_prompt, user_prompt, json_response=False):
            captured_kwargs.append(config_key)
            return SAMPLE_CLAIMS_JSON

        llm = AsyncMock()
        llm.complete = capture_complete

        await generate_claims_from_context(llm, _make_payload())
        assert captured_kwargs[0] == "filtering"


# ---------------------------------------------------------------------------
# ActorForecastEngine.generate() — LLM claim path
# ---------------------------------------------------------------------------


class TestActorForecastEngineClaims:
    async def test_uses_llm_claims_when_available(self):
        """With LLM, should produce questions from claim generation, not templates."""
        llm = _make_llm(SAMPLE_CLAIMS_JSON)
        engine = ActorForecastEngine()
        payload = _make_payload()

        run = await engine.generate(llm, payload)

        assert isinstance(run, ForecastRun)
        assert len(run.questions) == 2
        # Should be the LLM-generated claim, not a template
        assert "significant new developments" not in run.questions[0].question
        assert "Iran will impose new transit fees" in run.questions[0].question

    async def test_falls_back_to_heuristic_without_llm(self):
        """Without LLM, should still use heuristic fallback."""
        engine = ActorForecastEngine()
        payload = _make_payload()

        run = await engine.generate(None, payload)

        assert isinstance(run, ForecastRun)
        # Heuristic path produces template questions
        if run.questions:
            assert "significant new developments" in run.questions[0].question or "produce" in run.questions[0].question

    async def test_claims_have_evidence_thread_ids(self):
        """Generated questions should link back to source threads."""
        llm = _make_llm(SAMPLE_CLAIMS_JSON)
        engine = ActorForecastEngine()
        payload = _make_payload()

        run = await engine.generate(llm, payload)

        # First claim references "Iran Strait of Hormuz Restrictions" → thread_id=101
        q1 = run.questions[0]
        assert 101 in q1.evidence_thread_ids

    async def test_claims_have_reasoning_in_signals(self):
        """Reasoning should be stored in signals_cited for display."""
        llm = _make_llm(SAMPLE_CLAIMS_JSON)
        engine = ActorForecastEngine()
        payload = _make_payload()

        run = await engine.generate(llm, payload)

        q1 = run.questions[0]
        reasoning_signals = [s for s in q1.signals_cited if s.startswith("reasoning:")]
        assert len(reasoning_signals) == 1
        assert "restrictions" in reasoning_signals[0].lower() or "escalation" in reasoning_signals[0].lower()

    async def test_target_variable_is_topic_claim(self):
        """New claims should use 'topic_claim' target variable."""
        llm = _make_llm(SAMPLE_CLAIMS_JSON)
        engine = ActorForecastEngine()

        run = await engine.generate(llm, _make_payload())

        for q in run.questions:
            assert q.target_variable == "topic_claim"

    async def test_confidence_maps_to_probability(self):
        """Confidence labels should map to reasonable probabilities."""
        llm = _make_llm(SAMPLE_CLAIMS_JSON)
        engine = ActorForecastEngine()

        run = await engine.generate(llm, _make_payload())

        # First claim is "medium" confidence, second is "high"
        assert run.questions[0].probability < run.questions[1].probability

    async def test_falls_back_on_llm_failure(self):
        """If LLM fails, should gracefully fall back to heuristic questions."""
        async def fail_complete(*, config_key, system_prompt, user_prompt, json_response=False):
            raise RuntimeError("LLM down")

        llm = AsyncMock()
        llm.complete = fail_complete
        engine = ActorForecastEngine()
        payload = _make_payload()

        run = await engine.generate(llm, payload)

        assert isinstance(run, ForecastRun)
        # Should have fallen back to heuristic path
        assert run.engine == "actor"
