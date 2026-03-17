"""Tests for the structural prediction engine."""

from __future__ import annotations

import json
from datetime import date
from unittest.mock import AsyncMock


from nexus.engine.projection.evidence import EvidencePackage
from nexus.engine.projection.models import (
    StructuralAssessment,
)
from nexus.engine.projection.structural_engine import (
    StructuralBenchmarkEngine,
    predict_structural,
)


def _empty_evidence(question: str = "Will X happen?") -> EvidencePackage:
    return EvidencePackage(
        question=question,
        as_of=date(2026, 3, 15),
        entities=[],
        threads=[],
        convergence=[],
        divergence=[],
        causal_chains=[],
        relationships=[],
        relationship_changes=[],
        cross_topic_signals=[],
        recent_events=[],
        coverage={"entities_found": 0, "threads_found": 0},
    )


def _rich_evidence(question: str = "Will Iran do X?") -> EvidencePackage:
    return EvidencePackage(
        question=question,
        as_of=date(2026, 3, 15),
        entities=[{"name": "Iran", "entity_id": 1}],
        threads=[{
            "headline": "Iran nuclear talks",
            "trajectory_label": "accelerating",
            "momentum_score": 45.0,
            "velocity_7d": 3.0,
        }],
        convergence=[{
            "fact_text": "Talks resumed",
            "confirmed_by": ["NYT", "BBC"],
        }],
        divergence=[],
        causal_chains=[],
        relationships=[{
            "source_entity_name": "Iran",
            "target_entity_name": "US",
            "relation_type": "diplomatic_tension",
        }],
        relationship_changes=[],
        cross_topic_signals=[],
        recent_events=[{
            "date": "2026-03-10",
            "summary": "Iran resumed enrichment",
            "significance": 8,
        }],
        coverage={"entities_found": 1, "threads_found": 1,
                  "events_found": 1, "convergence_found": 1},
    )


def _make_llm_mock(responses: list[str]) -> AsyncMock:
    """Build a mock LLMClient that returns successive JSON responses."""
    llm = AsyncMock()
    llm.complete = AsyncMock(side_effect=responses)
    return llm


def _assessment_json(
    verdict: str = "yes",
    confidence: str = "medium",
    factors: list[dict] | None = None,
    reasoning: str = "Test reasoning",
    base_rate_reasoning: str = "Base rate ~30%",
    key_uncertainties: list[str] | None = None,
    signposts: list[str] | None = None,
) -> str:
    return json.dumps({
        "verdict": verdict,
        "confidence": confidence,
        "factors": factors or [
            {"factor": "Recent escalation", "direction": "supports_yes",
             "weight": "strong", "source_type": "trajectory"},
        ],
        "reasoning": reasoning,
        "base_rate_reasoning": base_rate_reasoning,
        "key_uncertainties": key_uncertainties or ["Diplomatic backchannel unknown"],
        "signposts": signposts or ["Watch for IAEA report"],
    })


def _contrarian_json(
    verdict: str = "no",
    confidence: str = "low",
    contrarian_argument: str = "Historical pattern: talks always resume",
    wildcards: list[str] | None = None,
) -> str:
    return json.dumps({
        "verdict": verdict,
        "confidence": confidence,
        "contrarian_argument": contrarian_argument,
        "wildcards": wildcards or ["Regime change"],
        "base_rate_critique": "Overweighting recent events",
    })


def _supervisor_json(
    verdict: str = "yes",
    confidence: str = "medium",
    factors: list[dict] | None = None,
    reasoning: str = "Reconciled: base rate analyst more compelling",
    contrarian_view: str = "Contrarian notes talks may resume",
    key_uncertainties: list[str] | None = None,
    signposts: list[str] | None = None,
) -> str:
    return json.dumps({
        "verdict": verdict,
        "confidence": confidence,
        "factors": factors or [
            {"factor": "Escalation pattern", "direction": "supports_yes",
             "weight": "strong", "source_type": "trajectory"},
            {"factor": "Historical talks pattern", "direction": "supports_no",
             "weight": "moderate", "source_type": "world_knowledge"},
        ],
        "reasoning": reasoning,
        "contrarian_view": contrarian_view,
        "key_uncertainties": key_uncertainties or ["Diplomatic channels"],
        "signposts": signposts or ["IAEA report", "UN session"],
    })


class TestPredictStructural:
    async def test_makes_three_llm_calls(self):
        """Engine should make exactly 3 LLM calls: base rate, contrarian, supervisor."""
        llm = _make_llm_mock([
            _assessment_json(),
            _contrarian_json(),
            _supervisor_json(),
        ])
        evidence = _rich_evidence()

        await predict_structural(llm, evidence)

        assert llm.complete.call_count == 3

    async def test_returns_structural_assessment(self):
        """Should return a StructuralAssessment model."""
        llm = _make_llm_mock([
            _assessment_json(),
            _contrarian_json(),
            _supervisor_json(),
        ])
        result = await predict_structural(llm, _rich_evidence())

        assert isinstance(result, StructuralAssessment)
        assert result.verdict in ("yes", "no", "uncertain")
        assert result.confidence in ("high", "medium", "low")

    async def test_has_kg_evidence_flag(self):
        """has_kg_evidence should reflect whether the KG had data."""
        llm = _make_llm_mock([
            _assessment_json(),
            _contrarian_json(),
            _supervisor_json(),
        ])

        # Rich evidence → has_kg_evidence=True
        result = await predict_structural(llm, _rich_evidence())
        assert result.has_kg_evidence is True

        # Empty evidence → has_kg_evidence=False
        llm2 = _make_llm_mock([
            _assessment_json(),
            _contrarian_json(),
            _supervisor_json(),
        ])
        result2 = await predict_structural(llm2, _empty_evidence())
        assert result2.has_kg_evidence is False

    async def test_prompts_include_evidence(self):
        """Base rate call should include formatted evidence in the prompt."""
        llm = _make_llm_mock([
            _assessment_json(),
            _contrarian_json(),
            _supervisor_json(),
        ])
        await predict_structural(llm, _rich_evidence("Will Iran do X?"))

        # First call is base rate — check the user prompt includes evidence
        first_call = llm.complete.call_args_list[0]
        user_prompt = first_call.args[2] if len(first_call.args) > 2 else first_call.kwargs.get("user_prompt", "")
        assert "Iran" in user_prompt
        assert "accelerating" in user_prompt or "Intelligence Evidence" in user_prompt

    async def test_prompts_include_question(self):
        """All calls should include the question text."""
        llm = _make_llm_mock([
            _assessment_json(),
            _contrarian_json(),
            _supervisor_json(),
        ])
        await predict_structural(llm, _rich_evidence("Will Iran develop nukes?"))

        for call in llm.complete.call_args_list:
            user_prompt = call.args[2] if len(call.args) > 2 else call.kwargs.get("user_prompt", "")
            assert "Will Iran develop nukes?" in user_prompt

    async def test_empty_evidence_still_forecasts(self):
        """Engine should always forecast, even without KG evidence (using LLM world knowledge)."""
        llm = _make_llm_mock([
            _assessment_json(verdict="no", confidence="low"),
            _contrarian_json(verdict="yes", confidence="low"),
            _supervisor_json(verdict="no", confidence="low"),
        ])
        result = await predict_structural(llm, _empty_evidence())
        assert result.verdict == "no"
        assert result.confidence == "low"

    async def test_supervisor_output_used_as_final(self):
        """The supervisor (3rd call) output should be the final assessment."""
        llm = _make_llm_mock([
            _assessment_json(verdict="yes", confidence="high"),
            _contrarian_json(verdict="no", confidence="low"),
            _supervisor_json(verdict="yes", confidence="medium",
                             reasoning="Supervisor reconciled"),
        ])
        result = await predict_structural(llm, _rich_evidence())
        assert result.verdict == "yes"
        assert result.confidence == "medium"
        assert "Supervisor reconciled" in result.reasoning


class TestImpliedProbability:
    def test_yes_high(self):
        a = StructuralAssessment(question="?", verdict="yes", confidence="high")
        assert a.implied_probability == 0.92

    def test_no_medium(self):
        a = StructuralAssessment(question="?", verdict="no", confidence="medium")
        assert a.implied_probability == 0.25

    def test_uncertain_any(self):
        for conf in ("high", "medium", "low"):
            a = StructuralAssessment(question="?", verdict="uncertain", confidence=conf)
            assert a.implied_probability == 0.50

    def test_binary_prediction(self):
        assert StructuralAssessment(question="?", verdict="yes", confidence="high").binary_prediction is True
        assert StructuralAssessment(question="?", verdict="no", confidence="low").binary_prediction is False
        assert StructuralAssessment(question="?", verdict="uncertain", confidence="medium").binary_prediction is None


class TestBenchmarkEngineProtocol:
    async def test_predict_probability_returns_float(self):
        """StructuralBenchmarkEngine must satisfy the BenchmarkEngine protocol."""
        engine = StructuralBenchmarkEngine()
        assert engine.engine_name == "structural"

        store = AsyncMock()
        store.get_all_entities = AsyncMock(return_value=[])
        store.get_cross_topic_signals_as_of = AsyncMock(return_value=[])

        llm = _make_llm_mock([
            _assessment_json(verdict="no", confidence="medium"),
            _contrarian_json(),
            _supervisor_json(verdict="no", confidence="medium"),
        ])

        prob = await engine.predict_probability(
            "Will X happen?", llm=llm, store=store, as_of=date(2026, 3, 15),
        )
        assert isinstance(prob, float)
        assert 0.0 <= prob <= 1.0

    async def test_ignores_market_prob(self):
        """Structural engine should NOT use market_prob at all."""
        engine = StructuralBenchmarkEngine()

        store = AsyncMock()
        store.get_all_entities = AsyncMock(return_value=[])
        store.get_cross_topic_signals_as_of = AsyncMock(return_value=[])

        llm = _make_llm_mock([
            _assessment_json(verdict="no", confidence="high"),
            _contrarian_json(),
            _supervisor_json(verdict="no", confidence="high"),
        ])

        prob = await engine.predict_probability(
            "Will X?", llm=llm, store=store,
            market_prob=0.95,  # should be ignored
            as_of=date(2026, 3, 15),
        )
        # Should return 0.08 (no+high), NOT anywhere near 0.95
        assert prob < 0.15
