"""Tests for swarm forecasting: multi-persona ensemble + aggregation."""

from __future__ import annotations

import json
import math

import pytest

from nexus.engine.projection.swarm import (
    PERSONAS,
    PersonaForecast,
    SwarmResult,
    anchor_blend,
    build_forecast_context,
    extremize,
    geometric_mean_of_odds,
    run_swarm,
    SwarmForecastEngine,
)
from nexus.engine.projection.forecasting import (
    ForecastEngineInput,
    ForecastQuestion,
    _build_candidate_catalog,
)
from nexus.engine.projection.models import CrossTopicSignal, ForecastRun
from nexus.engine.synthesis.knowledge import NarrativeThread
from nexus.engine.knowledge.events import Event

from datetime import date, timedelta
from unittest.mock import AsyncMock, MagicMock, patch


# ---------------------------------------------------------------------------
# Pure math: geometric_mean_of_odds
# ---------------------------------------------------------------------------

class TestGeometricMeanOfOdds:
    def test_equal_inputs_returns_same(self):
        result = geometric_mean_of_odds([0.6, 0.6, 0.6])
        assert result == pytest.approx(0.6, abs=0.01)

    def test_symmetric_around_half(self):
        # 0.3 and 0.7 have symmetric log-odds around 0
        result = geometric_mean_of_odds([0.3, 0.7])
        assert result == pytest.approx(0.5, abs=0.01)

    def test_high_values_pull_up(self):
        result = geometric_mean_of_odds([0.8, 0.8, 0.3])
        assert result > 0.5

    def test_low_values_pull_down(self):
        result = geometric_mean_of_odds([0.2, 0.2, 0.7])
        assert result < 0.5

    def test_weighted_shifts_toward_heavy(self):
        unweighted = geometric_mean_of_odds([0.3, 0.8])
        weighted = geometric_mean_of_odds([0.3, 0.8], weights=[1.0, 3.0])
        assert weighted > unweighted

    def test_single_input(self):
        result = geometric_mean_of_odds([0.75])
        assert result == pytest.approx(0.75, abs=0.01)

    def test_clips_to_valid_range(self):
        result = geometric_mean_of_odds([0.01, 0.99])
        assert 0.02 <= result <= 0.98

    def test_all_high(self):
        result = geometric_mean_of_odds([0.9, 0.85, 0.88])
        assert result > 0.8

    def test_all_low(self):
        result = geometric_mean_of_odds([0.1, 0.15, 0.12])
        assert result < 0.2


# ---------------------------------------------------------------------------
# Pure math: extremize
# ---------------------------------------------------------------------------

class TestExtremize:
    def test_pushes_above_half_higher(self):
        assert extremize(0.6, gamma=2.5) > 0.6

    def test_pushes_below_half_lower(self):
        assert extremize(0.4, gamma=2.5) < 0.4

    def test_preserves_half(self):
        assert extremize(0.5, gamma=2.5) == pytest.approx(0.5, abs=0.001)

    def test_gamma_1_is_identity(self):
        for p in [0.1, 0.3, 0.5, 0.7, 0.9]:
            assert extremize(p, gamma=1.0) == pytest.approx(p, abs=0.001)

    def test_higher_gamma_more_extreme(self):
        p = 0.7
        assert extremize(p, gamma=3.0) > extremize(p, gamma=2.0) > extremize(p, gamma=1.5)

    def test_clips_to_valid_range(self):
        assert extremize(0.99, gamma=5.0) <= 0.98
        assert extremize(0.01, gamma=5.0) >= 0.02

    def test_symmetric_around_half(self):
        above = extremize(0.7, gamma=2.5)
        below = extremize(0.3, gamma=2.5)
        # Should be symmetric: above-0.5 == 0.5-below
        assert abs((above - 0.5) - (0.5 - below)) < 0.01

    def test_gamma_below_1_compresses(self):
        """gamma < 1.0 should compress probabilities toward 0.5."""
        assert extremize(0.7, gamma=0.8) < 0.7
        assert extremize(0.3, gamma=0.8) > 0.3


# ---------------------------------------------------------------------------
# Anchor blend
# ---------------------------------------------------------------------------

class TestAnchorBlend:
    def test_full_weight_returns_swarm(self):
        result = anchor_blend(0.80, 0.50, swarm_weight=1.0)
        assert result == pytest.approx(0.80, abs=0.01)

    def test_zero_weight_returns_anchor(self):
        result = anchor_blend(0.80, 0.50, swarm_weight=0.0)
        assert result == pytest.approx(0.50, abs=0.01)

    def test_default_weight_blends(self):
        result = anchor_blend(0.80, 0.50, swarm_weight=0.4)
        # 0.50 + 0.4 * (0.80 - 0.50) = 0.50 + 0.12 = 0.62
        assert result == pytest.approx(0.62, abs=0.01)

    def test_clips_to_valid_range(self):
        assert anchor_blend(0.99, 0.98, swarm_weight=1.0) <= 0.98
        assert anchor_blend(0.01, 0.02, swarm_weight=1.0) >= 0.02

    def test_swarm_below_anchor_pulls_down(self):
        result = anchor_blend(0.20, 0.50, swarm_weight=0.4)
        assert result < 0.50


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------

def _make_thread(*, trajectory="accelerating", momentum=5.0, events=None):
    evts = events or [
        Event(
            date=date(2026, 3, 10),
            summary="Test partnership announced between Tesla and Google",
            significance=8,
            sources=[
                {"outlet": "Reuters", "affiliation": "wire", "country": "US", "language": "en"},
                {"outlet": "BBC", "affiliation": "state", "country": "GB", "language": "en"},
            ],
            entities=["Tesla", "Google"],
        ),
    ]
    return NarrativeThread(
        headline="Test Thread",
        events=evts,
        significance=8,
        thread_id=42,
        slug="test-thread",
        status="active",
        snapshot_count=6,
        velocity_7d=3.0,
        acceleration_7d=1.0,
        significance_trend_7d=0.5,
        momentum_score=momentum,
        trajectory_label=trajectory,
        convergence=[
            {"fact": "Multiple sources confirm Tesla-Google deal", "confirmed_by": ["Reuters", "BBC"]},
        ],
        divergence=[
            {"shared_event": "Partnership scope", "source_a": "Reuters", "framing_a": "Limited pilot",
             "source_b": "TechCrunch", "framing_b": "Full strategic alliance"},
        ],
        key_entities=["Tesla", "Google"],
    )


def _make_payload(threads=None, events=None, signals=None):
    thread = _make_thread()
    if threads is None:
        threads = [thread]
    if events is None:
        events = thread.events
    return ForecastEngineInput(
        topic_slug="ai-ml-research",
        topic_name="AI & ML Research",
        run_date=date(2026, 3, 12),
        threads=threads,
        recent_events=events,
        cross_topic_signals=signals or [],
    )


def _make_question(**overrides):
    defaults = dict(
        question="Will Tesla and Google lead to a new partnership event by 2026-03-19?",
        forecast_type="binary",
        target_variable="partnership_or_product_event",
        target_metadata={"topic_slug": "ai-ml-research", "anchor_entities": ["Tesla", "Google"]},
        probability=0.55,
        base_rate=0.55,
        resolution_criteria="Resolves true if a partnership event occurs.",
        resolution_date=date(2026, 3, 19),
        horizon_days=7,
        signpost="Tesla-Google partnership discussions accelerating",
        signals_cited=["trajectory:accelerating"],
        evidence_event_ids=[],
        evidence_thread_ids=[42],
        cross_topic_signal_ids=[],
    )
    defaults.update(overrides)
    return ForecastQuestion(**defaults)


class TestBuildForecastContext:
    def test_includes_base_fields(self):
        payload = _make_payload()
        question = _make_question()
        ctx = build_forecast_context(question, payload)
        assert "question" in ctx
        assert "target_variable" in ctx
        assert "prior_base_rate" in ctx

    def test_includes_thread_signals(self):
        payload = _make_payload()
        question = _make_question()
        ctx = build_forecast_context(question, payload)
        assert ctx["thread_trajectory"] == "accelerating"
        assert ctx["thread_momentum"] == 5.0
        assert len(ctx["convergence_facts"]) == 1
        assert len(ctx["divergence_facts"]) == 1

    def test_includes_source_diversity(self):
        payload = _make_payload()
        question = _make_question()
        ctx = build_forecast_context(question, payload)
        assert ctx["source_count"] >= 2
        assert ctx["country_count"] >= 1

    def test_includes_empirical_base_rate(self):
        payload = _make_payload()
        question = _make_question()
        cal_data = [
            {"target_variable": "partnership_or_product_event", "resolved_bool": True},
            {"target_variable": "partnership_or_product_event", "resolved_bool": True},
            {"target_variable": "partnership_or_product_event", "resolved_bool": False},
        ]
        ctx = build_forecast_context(question, payload, calibration_data=cal_data)
        assert ctx["empirical_hit_rate"] == pytest.approx(2 / 3, abs=0.01)
        assert ctx["empirical_n"] == 3

    def test_handles_no_calibration_data(self):
        payload = _make_payload()
        question = _make_question()
        ctx = build_forecast_context(question, payload, calibration_data=None)
        assert ctx["empirical_hit_rate"] is None
        assert ctx["empirical_n"] == 0

    def test_handles_no_threads(self):
        """When no threads exist at all, thread fields should be None."""
        payload = _make_payload(threads=[])
        question = _make_question(target_metadata={"topic_slug": "ai-ml-research", "thread_id": 999})
        ctx = build_forecast_context(question, payload)
        assert ctx["thread_trajectory"] is None
        assert ctx["thread_momentum"] is None


# ---------------------------------------------------------------------------
# Swarm runner (mock LLM)
# ---------------------------------------------------------------------------

def _mock_llm_for_personas(persona_probs: dict[str, float]):
    """Create a mock LLM that returns different probabilities per persona system prompt."""
    async def mock_complete(*, config_key, system_prompt, user_prompt, json_response=False):
        for persona_name, prob in persona_probs.items():
            if persona_name in system_prompt.lower() or persona_name.replace("_", " ") in system_prompt.lower():
                return json.dumps({
                    "reasoning": f"Test reasoning from {persona_name}",
                    "probability": prob,
                    "confidence": "medium",
                })
        # Default
        return json.dumps({"reasoning": "default", "probability": 0.5, "confidence": "low"})

    llm = AsyncMock()
    llm.complete = mock_complete
    return llm


@pytest.mark.asyncio
class TestRunSwarm:
    async def test_returns_swarm_result(self):
        llm = _mock_llm_for_personas({
            "calibration": 0.65,
            "momentum": 0.75,
            "skeptic": 0.40,
            "cross-domain": 0.60,
        })
        payload = _make_payload()
        question = _make_question()
        result = await run_swarm(llm, question, payload)
        assert isinstance(result, SwarmResult)
        assert len(result.forecasts) > 0
        assert 0.02 <= result.final_probability <= 0.98

    async def test_aggregates_via_geometric_mean(self):
        """With known persona outputs, the aggregated probability should match geometric mean of odds."""
        llm = _mock_llm_for_personas({
            "calibration": 0.70,
            "momentum": 0.70,
            "skeptic": 0.70,
            "cross-domain": 0.70,
        })
        payload = _make_payload()
        question = _make_question()
        result = await run_swarm(llm, question, payload, gamma=1.0)  # no extremization
        # All at 0.70 → geometric mean of odds = 0.70
        assert result.aggregated_probability == pytest.approx(0.70, abs=0.05)

    async def test_extremization_with_high_gamma(self):
        llm = _mock_llm_for_personas({
            "calibration": 0.70,
            "momentum": 0.70,
            "skeptic": 0.70,
            "cross-domain": 0.70,
        })
        payload = _make_payload()
        question = _make_question()
        result = await run_swarm(llm, question, payload, gamma=2.5)
        # After extremization with gamma>1, 0.70 should be pushed higher
        assert result.extremized_probability > 0.70

    async def test_compression_with_default_gamma(self):
        """Default gamma < 1.0 compresses overconfident LLM estimates toward 0.5."""
        llm = _mock_llm_for_personas({
            "calibration": 0.70,
            "momentum": 0.70,
            "skeptic": 0.70,
            "cross-domain": 0.70,
        })
        payload = _make_payload()
        question = _make_question()
        result = await run_swarm(llm, question, payload)  # default gamma=0.8
        # After compression, 0.70 should be pulled toward 0.5
        assert result.extremized_probability < 0.70

    async def test_diverse_personas_produce_wider_range(self):
        """When personas disagree significantly, the spread should be wide."""
        llm = _mock_llm_for_personas({
            "calibration": 0.30,
            "momentum": 0.85,
            "skeptic": 0.20,
            "cross-domain": 0.70,
        })
        payload = _make_payload()
        question = _make_question()
        result = await run_swarm(llm, question, payload)
        # The aggregation should land somewhere between the extremes
        probs = [f.probability for f in result.forecasts]
        assert min(probs) < 0.35
        assert max(probs) > 0.65

    async def test_handles_persona_failure(self):
        """If one persona fails, others still aggregate."""
        call_count = 0

        async def flaky_complete(*, config_key, system_prompt, user_prompt, json_response=False):
            nonlocal call_count
            call_count += 1
            if "skeptic" in system_prompt.lower():
                raise RuntimeError("LLM failure")
            return json.dumps({"reasoning": "ok", "probability": 0.60, "confidence": "medium"})

        llm = AsyncMock()
        llm.complete = flaky_complete
        payload = _make_payload()
        question = _make_question()
        result = await run_swarm(llm, question, payload)
        # Should still get results from non-failing personas
        assert len(result.forecasts) >= 2
        assert result.final_probability > 0.0

    async def test_all_personas_fail_returns_fallback(self):
        async def always_fail(*, config_key, system_prompt, user_prompt, json_response=False):
            raise RuntimeError("total failure")

        llm = AsyncMock()
        llm.complete = always_fail
        payload = _make_payload()
        question = _make_question(probability=0.45)
        result = await run_swarm(llm, question, payload)
        # Falls back to the question's deterministic probability
        assert result.final_probability == pytest.approx(0.45, abs=0.01)
        assert len(result.forecasts) == 0

    async def test_per_persona_reasoning_captured(self):
        llm = _mock_llm_for_personas({
            "calibration": 0.65,
            "momentum": 0.75,
            "skeptic": 0.40,
            "cross-domain": 0.60,
        })
        payload = _make_payload()
        question = _make_question()
        result = await run_swarm(llm, question, payload)
        for forecast in result.forecasts:
            assert forecast.reasoning
            assert forecast.persona


# ---------------------------------------------------------------------------
# SwarmForecastEngine
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestSwarmForecastEngine:
    async def test_generates_forecast_run(self):
        llm = _mock_llm_for_personas({
            "calibration": 0.65,
            "momentum": 0.75,
            "skeptic": 0.40,
            "cross-domain": 0.60,
        })
        engine = SwarmForecastEngine()
        payload = _make_payload()
        run = await engine.generate(llm, payload)
        assert isinstance(run, ForecastRun)
        assert run.engine == "swarm"
        assert len(run.questions) > 0
        for q in run.questions:
            assert 0.02 <= q.probability <= 0.98

    async def test_without_llm_uses_deterministic_fallback(self):
        engine = SwarmForecastEngine()
        payload = _make_payload()
        run = await engine.generate(None, payload)
        assert run.engine == "swarm"
        # Should still produce questions (deterministic fallback)
        assert len(run.questions) > 0

    async def test_probabilities_differ_from_native(self):
        """Swarm should produce different probabilities than deterministic native."""
        llm = _mock_llm_for_personas({
            "calibration": 0.80,
            "momentum": 0.85,
            "skeptic": 0.60,
            "cross-domain": 0.75,
        })
        engine = SwarmForecastEngine()
        payload = _make_payload()

        swarm_run = await engine.generate(llm, payload)

        from nexus.engine.projection.forecasting import NativeForecastEngine
        native_run = await NativeForecastEngine().generate(None, payload)

        # At least one question should have a different probability
        swarm_probs = sorted(q.probability for q in swarm_run.questions)
        native_probs = sorted(q.probability for q in native_run.questions)
        assert swarm_probs != native_probs


# ---------------------------------------------------------------------------
# Persona definitions
# ---------------------------------------------------------------------------

class TestPersonas:
    def test_all_personas_have_required_fields(self):
        for name, spec in PERSONAS.items():
            assert "system" in spec, f"{name} missing system prompt"
            assert "weight" in spec, f"{name} missing weight"
            assert spec["weight"] > 0, f"{name} has non-positive weight"

    def test_at_least_three_personas(self):
        assert len(PERSONAS) >= 3

    def test_weights_sum_reasonable(self):
        total = sum(s["weight"] for s in PERSONAS.values())
        assert total > 2.0  # at least meaningful diversity
