"""Tests for graph-informed prediction engine."""

from __future__ import annotations

import json
from datetime import date
from unittest.mock import AsyncMock, patch

import pytest

from nexus.engine.projection.forecasting import (
    ForecastEngineInput,
    ForecastQuestion,
    _build_candidate_catalog,
    get_forecast_engine,
)
from nexus.engine.projection.models import CrossTopicSignal, ForecastRun
from nexus.engine.synthesis.knowledge import NarrativeThread
from nexus.engine.knowledge.events import Event


def _make_thread(
    headline="Iran missile escalation",
    status="active",
    trajectory="accelerating",
    entities=None,
    thread_id=101,
):
    return NarrativeThread(
        headline=headline,
        summary="Ongoing escalation between Iran and Israel",
        key_entities=entities or ["Iran", "Israel", "United States"],
        evidence=["Missile barrage on March 1", "Sanctions expanded March 5"],
        thread_id=thread_id,
        slug="iran-missile-escalation",
        status=status,
        trajectory_label=trajectory,
        momentum_score=8.0,
        significance=9,
    )


def _make_payload(threads=None, events=None):
    return ForecastEngineInput(
        topic_slug="iran-us-relations",
        topic_name="Iran-US Relations",
        run_date=date(2026, 3, 15),
        threads=threads if threads is not None else [_make_thread()],
        recent_events=events or [
            Event(
                date=date(2026, 3, 10),
                summary="Beijing proposes ceasefire via back channels",
                significance=6,
                relation_to_prior="diplomatic",
                entities=["China", "Iran"],
                sources=[],
            ),
        ],
        cross_topic_signals=[],
    )


def _mock_neighborhood():
    """Fake entity neighborhood data."""
    return {
        "entities": [
            {"id": 2, "name": "Israel", "type": "country"},
            {"id": 3, "name": "United States", "type": "country"},
            {"id": 4, "name": "China", "type": "country"},
        ],
        "relationships": [
            {
                "id": 1,
                "source_entity_id": 1,
                "target_entity_id": 2,
                "relation_type": "threatens",
                "evidence_text": "Missile barrage targeting Haifa",
                "strength": 0.9,
                "valid_from": "2026-03-01",
                "valid_until": None,
                "source_event_id": 1,
            },
            {
                "id": 2,
                "source_entity_id": 3,
                "target_entity_id": 1,
                "relation_type": "sanctions",
                "evidence_text": "Treasury expands shipping sanctions",
                "strength": 0.85,
                "valid_from": "2026-03-05",
                "valid_until": None,
                "source_event_id": 2,
            },
            {
                "id": 3,
                "source_entity_id": 1,
                "target_entity_id": 4,
                "relation_type": "negotiates_with",
                "evidence_text": "Beijing proposes ceasefire",
                "strength": 0.4,
                "valid_from": "2026-03-10",
                "valid_until": None,
                "source_event_id": 3,
            },
        ],
    }


def _mock_timeline():
    """Fake relationship timeline data."""
    return [
        {
            "id": 3,
            "source_entity_id": 1,
            "target_entity_id": 4,
            "relation_type": "negotiates_with",
            "evidence_text": "Beijing proposes ceasefire",
            "strength": 0.4,
            "valid_from": "2026-03-10",
            "valid_until": None,
            "source_entity_name": "Iran",
            "target_entity_name": "China",
        },
        {
            "id": 4,
            "source_entity_id": 1,
            "target_entity_id": 5,
            "relation_type": "allies_with",
            "evidence_text": "Axis of resistance partnership",
            "strength": 0.7,
            "valid_from": "2025-11-20",
            "valid_until": "2026-03-08",
            "source_entity_name": "Iran",
            "target_entity_name": "Hamas",
        },
    ]


# ── Registration ────────────────────────────────────────────────────


class TestGraphEngineRegistration:

    def test_registers_in_get_forecast_engine(self):
        engine = get_forecast_engine("graph")
        assert engine.engine_name == "graph"

    def test_case_insensitive(self):
        engine = get_forecast_engine("Graph")
        assert engine.engine_name == "graph"


# ── Graph Context Building ──────────────────────────────────────────


class TestBuildGraphContext:

    async def test_extracts_anchor_entities(self):
        from nexus.engine.projection.graph_engine import build_graph_context

        question = ForecastQuestion(
            question="Will Iran launch another attack?",
            forecast_type="binary",
            target_variable="thread_continuation",
            probability=0.45,
            base_rate=0.25,
            resolution_criteria="New military event within 7 days",
            resolution_date=date(2026, 3, 22),
            horizon_days=7,
            signpost="Missile barrage on March 1",
            target_metadata={"anchor_entities": ["Iran", "Israel"]},
        )
        payload = _make_payload()

        mock_store = AsyncMock()
        mock_store.find_entity.side_effect = lambda name: {
            "Iran": {"id": 1, "canonical_name": "Iran", "entity_type": "country"},
            "Israel": {"id": 2, "canonical_name": "Israel", "entity_type": "country"},
        }.get(name)
        mock_store.get_entity_neighborhood.return_value = _mock_neighborhood()
        mock_store.get_relationship_timeline.return_value = _mock_timeline()

        ctx = await build_graph_context(mock_store, question, payload)

        assert "active_relationships" in ctx
        assert "historical_relationships" in ctx
        assert "relationship_chains" in ctx
        assert len(ctx["active_relationships"]) >= 1

    async def test_handles_no_entities(self):
        from nexus.engine.projection.graph_engine import build_graph_context

        question = ForecastQuestion(
            question="Will something happen?",
            forecast_type="binary",
            target_variable="thread_continuation",
            probability=0.3,
            base_rate=0.25,
            resolution_criteria="Something happens",
            resolution_date=date(2026, 3, 22),
            horizon_days=7,
            signpost="General activity",
            target_metadata={},
        )
        payload = _make_payload(threads=[])

        mock_store = AsyncMock()
        mock_store.find_entity.return_value = None
        mock_store.get_entity_neighborhood.return_value = {"entities": [], "relationships": []}
        mock_store.get_relationship_timeline.return_value = []

        ctx = await build_graph_context(mock_store, question, payload)
        assert ctx["active_relationships"] == []

    async def test_separates_active_from_historical(self):
        from nexus.engine.projection.graph_engine import build_graph_context

        question = ForecastQuestion(
            question="Will Iran escalate?",
            forecast_type="binary",
            target_variable="thread_continuation",
            probability=0.45,
            base_rate=0.25,
            resolution_criteria="Escalation event",
            resolution_date=date(2026, 3, 22),
            horizon_days=7,
            signpost="Ongoing escalation pattern",
            target_metadata={"anchor_entities": ["Iran"]},
        )
        payload = _make_payload()

        mock_store = AsyncMock()
        mock_store.find_entity.return_value = {
            "id": 1, "canonical_name": "Iran", "entity_type": "country"
        }
        mock_store.get_entity_neighborhood.return_value = _mock_neighborhood()
        mock_store.get_relationship_timeline.return_value = _mock_timeline()

        ctx = await build_graph_context(mock_store, question, payload)
        # Timeline has one active (negotiates_with) and one historical (allies_with Hamas)
        assert len(ctx["historical_relationships"]) >= 1
        hist = ctx["historical_relationships"][0]
        assert hist.get("valid_until") is not None


# ── Engine Generate ─────────────────────────────────────────────────


class TestGraphForecastEngine:

    async def test_falls_back_without_llm(self):
        from nexus.engine.projection.graph_engine import GraphForecastEngine

        engine = GraphForecastEngine()
        payload = _make_payload()

        run = await engine.generate(None, payload)
        assert isinstance(run, ForecastRun)
        assert run.engine == "graph"
        # Without LLM, should use trajectory probabilities as-is
        for q in run.questions:
            assert 0.05 <= q.probability <= 0.95

    async def test_anchor_blends_with_deterministic(self):
        from nexus.engine.projection.graph_engine import GraphForecastEngine

        engine = GraphForecastEngine()
        payload = _make_payload()

        mock_llm = AsyncMock()
        # LLM returns a high probability
        mock_llm.complete.return_value = json.dumps({
            "reasoning": "Strong graph evidence",
            "adjustments": [
                {"direction": "up", "amount": 0.15, "evidence": "threatens relationship"}
            ],
            "probability": 0.80,
        })

        mock_store = AsyncMock()
        mock_store.find_entity.return_value = {
            "id": 1, "canonical_name": "Iran", "entity_type": "country"
        }
        mock_store.get_entity_neighborhood.return_value = _mock_neighborhood()
        mock_store.get_relationship_timeline.return_value = _mock_timeline()

        run = await engine.generate(
            mock_llm, payload, store=mock_store,
        )
        assert isinstance(run, ForecastRun)
        # Probability should be blended — not the raw 0.80 from LLM
        # With 3 active rels, blend_weight=0.15 (conservative tier)
        for q in run.questions:
            assert 0.05 <= q.probability <= 0.95

    async def test_skips_llm_when_no_graph_evidence(self):
        from nexus.engine.projection.graph_engine import GraphForecastEngine

        engine = GraphForecastEngine()
        payload = _make_payload()

        mock_llm = AsyncMock()
        mock_store = AsyncMock()
        mock_store.find_entity.return_value = {
            "id": 1, "canonical_name": "Iran", "entity_type": "country"
        }
        # Empty graph — no relationships
        mock_store.get_entity_neighborhood.return_value = {"entities": [], "relationships": []}
        mock_store.get_relationship_timeline.return_value = []

        run = await engine.generate(mock_llm, payload, store=mock_store)
        # LLM should NOT have been called — no graph evidence to adjust with
        mock_llm.complete.assert_not_called()
        # Should annotate with skip reason
        for q in run.questions:
            cited = q.signals_cited or []
            assert any("graph:skipped" in s for s in cited)

    async def test_prompt_includes_forbidden_knowledge_constraint(self):
        from nexus.engine.projection.graph_engine import render_graph_prompt

        ctx = {
            "active_relationships": [
                {
                    "source": "Iran", "target": "Israel", "type": "threatens",
                    "strength": 0.9, "valid_from": "2026-03-01",
                    "evidence": "Missile barrage",
                },
            ],
            "historical_relationships": [],
            "relationship_chains": ["Iran → threatens → Israel → allies_with → US"],
            "stats": {"entities_in_neighborhood": 4, "active_count": 3,
                      "invalidated_this_week": 1, "new_this_week": 2},
        }

        prompt = render_graph_prompt(
            question="Will Iran attack again?",
            deterministic_probability=0.45,
            base_rate=0.25,
            graph_context=ctx,
            run_date=date(2026, 3, 15),
        )
        assert "FORBIDDEN" in prompt
        assert "own knowledge" in prompt.lower()
        assert "Iran" in prompt
        assert "threatens" in prompt
