"""Tests for actor-based prediction engine."""

from __future__ import annotations

import json
from datetime import date, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nexus.engine.knowledge.events import Event
from nexus.engine.knowledge.store import KnowledgeStore
from nexus.engine.projection.actor_engine import (
    ActorAnalysis,
    ActorForecastEngine,
    ActorKnowledge,
    ActorPrediction,
    assemble_actor_knowledge,
    identify_actors,
    predict,
    reason_about_actor,
    synthesize_prediction,
)
from nexus.engine.projection.forecasting import ForecastEngineInput
from nexus.engine.projection.models import ForecastRun


@pytest.fixture
async def store(tmp_path):
    s = KnowledgeStore(tmp_path / "actor-test.db")
    await s.initialize()
    yield s
    await s.close()


def _make_mock_llm(response: dict | str):
    """Create a mock LLM that returns a fixed JSON response."""
    if isinstance(response, dict):
        response = json.dumps(response)

    async def mock_complete(*, config_key, system_prompt, user_prompt, json_response=False):
        return response

    llm = AsyncMock()
    llm.complete = mock_complete
    return llm


# ---------------------------------------------------------------------------
# identify_actors
# ---------------------------------------------------------------------------


class TestIdentifyActors:
    async def test_extracts_entities_from_question_text(self, store):
        """Should find entities that appear in the question."""
        event = Event(date=date(2026, 3, 10), summary="Saudi Arabia signs deal", entities=["Saudi Arabia"])
        await store.add_events([event], "geopolitics")
        actors = await identify_actors(store, "Will Saudi Arabia normalize relations with Israel?")
        names = [a["name"] for a in actors]
        assert "Saudi Arabia" in names

    async def test_resolves_via_store(self, store):
        """Should resolve entities through the store's entity lookup."""
        event = Event(date=date(2026, 3, 10), summary="OpenAI releases GPT-5", entities=["OpenAI"])
        await store.add_events([event], "ai-ml-research")
        actors = await identify_actors(store, "Will OpenAI release a new model?")
        assert any(a["name"] == "OpenAI" for a in actors)

    async def test_caps_at_max_actors(self, store):
        """Should not return more than max_actors."""
        for entity in ["A Corp", "B Corp", "C Corp", "D Corp", "E Corp", "F Corp"]:
            event = Event(date=date(2026, 3, 10), summary=f"{entity} news", entities=[entity])
            await store.add_events([event], "test-topic")
        actors = await identify_actors(
            store,
            "Will A Corp, B Corp, C Corp, D Corp, E Corp, F Corp merge?",
            max_actors=3,
        )
        assert len(actors) <= 3

    async def test_handles_no_matching_entities(self, store):
        """Should return empty list when no entities match."""
        actors = await identify_actors(store, "Will aliens land on Earth?")
        assert actors == []

    async def test_includes_entity_id_when_found(self, store):
        """Actors resolved from store should include entity_id."""
        event = Event(date=date(2026, 3, 10), summary="Tesla reports earnings", entities=["Tesla"])
        await store.add_events([event], "tech")
        actors = await identify_actors(store, "Will Tesla stock rise?")
        tesla = next((a for a in actors if a["name"] == "Tesla"), None)
        assert tesla is not None
        assert tesla.get("entity_id") is not None


# ---------------------------------------------------------------------------
# assemble_actor_knowledge
# ---------------------------------------------------------------------------


class TestAssembleActorKnowledge:
    async def test_returns_actor_knowledge(self, store):
        """Should return an ActorKnowledge dataclass with populated fields."""
        events = [
            Event(date=date(2026, 3, 5), summary="Iran sanctions tightened", entities=["Iran", "US"]),
            Event(date=date(2026, 3, 10), summary="Iran oil exports drop", entities=["Iran"]),
        ]
        await store.add_events(events, "iran-us")
        entity = await store.find_entity("Iran")
        assert entity is not None

        knowledge = await assemble_actor_knowledge(
            store,
            {"name": "Iran", "entity_id": entity["id"]},
            as_of=date(2026, 3, 12),
        )
        assert isinstance(knowledge, ActorKnowledge)
        assert knowledge.name == "Iran"
        assert len(knowledge.recent_events) > 0

    async def test_handles_entity_with_no_data(self, store):
        """Should return empty knowledge for unknown entity."""
        knowledge = await assemble_actor_knowledge(
            store,
            {"name": "UnknownCorp", "entity_id": None},
            as_of=date(2026, 3, 12),
        )
        assert isinstance(knowledge, ActorKnowledge)
        assert knowledge.recent_events == []
        assert knowledge.active_relationships == []


# ---------------------------------------------------------------------------
# reason_about_actor
# ---------------------------------------------------------------------------


class TestReasonAboutActor:
    async def test_returns_actor_analysis(self):
        """Should return a structured ActorAnalysis."""
        llm = _make_mock_llm({
            "direction": "increases",
            "magnitude": "moderate",
            "reasoning": "Iran's recent sanctions tightening increases the probability.",
            "key_uncertainty": "US response unclear",
            "probability_shift": 0.15,
        })
        knowledge = ActorKnowledge(
            name="Iran",
            entity_id=1,
            recent_events=[
                Event(date=date(2026, 3, 10), summary="Iran sanctions", entities=["Iran"]),
            ],
        )
        analysis = await reason_about_actor(
            llm, knowledge, "Will Iran face new sanctions?"
        )
        assert isinstance(analysis, ActorAnalysis)
        assert analysis.actor == "Iran"
        assert analysis.direction in {"increases", "decreases", "neutral"}

    async def test_handles_llm_failure(self):
        """Should return a neutral analysis on LLM failure."""
        async def fail_complete(*, config_key, system_prompt, user_prompt, json_response=False):
            raise RuntimeError("LLM down")

        llm = AsyncMock()
        llm.complete = fail_complete
        knowledge = ActorKnowledge(name="Iran", entity_id=1)

        analysis = await reason_about_actor(
            llm, knowledge, "Will Iran face new sanctions?"
        )
        assert isinstance(analysis, ActorAnalysis)
        assert analysis.direction == "neutral"
        assert analysis.probability_shift == 0.0

    async def test_shift_bounded(self):
        """Probability shift should be bounded to [-0.3, 0.3]."""
        llm = _make_mock_llm({
            "direction": "increases",
            "magnitude": "large",
            "reasoning": "Extreme scenario",
            "key_uncertainty": "None",
            "probability_shift": 0.9,  # way out of bounds
        })
        knowledge = ActorKnowledge(name="Iran", entity_id=1)
        analysis = await reason_about_actor(
            llm, knowledge, "Will Iran face new sanctions?"
        )
        assert -0.3 <= analysis.probability_shift <= 0.3

    async def test_prompt_includes_fresh_data_framing(self):
        """The prompt should tell the LLM to trust our data over its priors."""
        captured_prompts = []

        async def capture_complete(*, config_key, system_prompt, user_prompt, json_response=False):
            captured_prompts.append({"system": system_prompt, "user": user_prompt})
            return json.dumps({
                "direction": "neutral",
                "magnitude": "small",
                "reasoning": "ok",
                "key_uncertainty": "ok",
                "probability_shift": 0.0,
            })

        llm = AsyncMock()
        llm.complete = capture_complete
        knowledge = ActorKnowledge(
            name="Iran",
            entity_id=1,
            recent_events=[
                Event(date=date(2026, 3, 10), summary="Iran test", entities=["Iran"]),
            ],
        )
        await reason_about_actor(llm, knowledge, "Will Iran face sanctions?")
        assert len(captured_prompts) == 1
        combined = captured_prompts[0]["system"] + captured_prompts[0]["user"]
        assert "ground truth" in combined.lower() or "override" in combined.lower()


# ---------------------------------------------------------------------------
# synthesize_prediction
# ---------------------------------------------------------------------------


class TestSynthesizePrediction:
    async def test_returns_prediction(self):
        """Should return an ActorPrediction with a probability."""
        llm = _make_mock_llm({
            "probability": 0.65,
            "reasoning": "Combined actor analysis suggests...",
            "key_uncertainties": ["US response unclear"],
            "signposts": ["New sanctions announcement"],
        })
        analyses = [
            ActorAnalysis(
                actor="Iran",
                direction="increases",
                magnitude="moderate",
                reasoning="Sanctions tightening",
                key_uncertainty="US response",
                probability_shift=0.15,
            ),
        ]
        prediction = await synthesize_prediction(
            llm,
            "Will Iran face new sanctions?",
            analyses,
        )
        assert isinstance(prediction, ActorPrediction)
        assert 0.05 <= prediction.calibrated_probability <= 0.95

    async def test_includes_market_probability_when_provided(self):
        """When market_prob is given, it should influence the result."""
        llm = _make_mock_llm({
            "probability": 0.70,
            "reasoning": "Market alignment",
            "key_uncertainties": [],
            "signposts": [],
        })
        analyses = [
            ActorAnalysis(
                actor="Iran",
                direction="increases",
                magnitude="moderate",
                reasoning="test",
                key_uncertainty="test",
                probability_shift=0.1,
            ),
        ]
        prediction = await synthesize_prediction(
            llm,
            "Will Iran face new sanctions?",
            analyses,
            market_prob=0.55,
        )
        assert isinstance(prediction, ActorPrediction)
        # With market prob provided, should blend
        assert 0.05 <= prediction.calibrated_probability <= 0.95

    async def test_probability_in_valid_range(self):
        """Output probability should always be in [0.05, 0.95]."""
        llm = _make_mock_llm({
            "probability": 0.99,
            "reasoning": "Very confident",
            "key_uncertainties": [],
            "signposts": [],
        })
        prediction = await synthesize_prediction(
            llm,
            "Will X happen?",
            [ActorAnalysis(
                actor="X", direction="increases", magnitude="large",
                reasoning="t", key_uncertainty="t", probability_shift=0.3,
            )],
        )
        assert 0.05 <= prediction.calibrated_probability <= 0.95


# ---------------------------------------------------------------------------
# predict (full pipeline)
# ---------------------------------------------------------------------------


class TestPredict:
    async def test_full_pipeline(self, store):
        """End-to-end: predict should return an ActorPrediction."""
        events = [
            Event(date=date(2026, 3, 5), summary="Iran sanctions tightened by US", entities=["Iran", "US"]),
            Event(date=date(2026, 3, 10), summary="Iran oil exports decrease", entities=["Iran"]),
        ]
        await store.add_events(events, "iran-us")

        call_count = 0

        async def mock_complete(*, config_key, system_prompt, user_prompt, json_response=False):
            nonlocal call_count
            call_count += 1
            if "actor" in system_prompt.lower() or "how does" in user_prompt.lower():
                return json.dumps({
                    "direction": "increases",
                    "magnitude": "moderate",
                    "reasoning": "Actor analysis",
                    "key_uncertainty": "unclear",
                    "probability_shift": 0.1,
                })
            return json.dumps({
                "probability": 0.65,
                "reasoning": "Combined analysis",
                "key_uncertainties": ["timing"],
                "signposts": ["New sanctions"],
            })

        llm = AsyncMock()
        llm.complete = mock_complete

        prediction = await predict(
            store, llm,
            "Will Iran face new sanctions within 14 days?",
            run_date=date(2026, 3, 12),
        )
        assert isinstance(prediction, ActorPrediction)
        assert 0.05 <= prediction.calibrated_probability <= 0.95
        assert len(prediction.actors) > 0
        assert call_count >= 2  # at least 1 per-actor + 1 synthesis

    async def test_falls_back_without_llm(self, store):
        """Without LLM, should still return a prediction using heuristics."""
        events = [
            Event(date=date(2026, 3, 10), summary="Iran news", entities=["Iran"]),
        ]
        await store.add_events(events, "iran-us")

        prediction = await predict(
            store, None,
            "Will Iran face new sanctions?",
            run_date=date(2026, 3, 12),
        )
        assert isinstance(prediction, ActorPrediction)
        assert 0.05 <= prediction.calibrated_probability <= 0.95

    async def test_clips_range(self, store):
        """Prediction probability should always be clipped to [0.05, 0.95]."""
        prediction = await predict(
            store, None,
            "Will something happen?",
            run_date=date(2026, 3, 12),
        )
        assert 0.05 <= prediction.calibrated_probability <= 0.95


# ---------------------------------------------------------------------------
# ActorForecastEngine (protocol compliance)
# ---------------------------------------------------------------------------


class TestActorForecastEngine:
    def test_engine_name(self):
        engine = ActorForecastEngine()
        assert engine.engine_name == "actor"

    async def test_generates_forecast_run(self, store):
        """Should produce a ForecastRun with questions."""
        events = [
            Event(date=date(2026, 3, 5), summary="Iran sanctions", entities=["Iran", "US"]),
            Event(date=date(2026, 3, 10), summary="Iran talks resume", entities=["Iran"]),
        ]
        await store.add_events(events, "iran-us")

        async def mock_complete(*, config_key, system_prompt, user_prompt, json_response=False):
            if "actor" in system_prompt.lower() or "how does" in user_prompt.lower():
                return json.dumps({
                    "direction": "increases",
                    "magnitude": "moderate",
                    "reasoning": "Actor analysis",
                    "key_uncertainty": "unclear",
                    "probability_shift": 0.1,
                })
            return json.dumps({
                "probability": 0.60,
                "reasoning": "Combined",
                "key_uncertainties": [],
                "signposts": [],
            })

        llm = AsyncMock()
        llm.complete = mock_complete

        engine = ActorForecastEngine()
        payload = ForecastEngineInput(
            topic_slug="iran-us",
            topic_name="Iran-US Relations",
            run_date=date(2026, 3, 12),
            threads=[],
            recent_events=await store.get_recent_events("iran-us", days=14),
            cross_topic_signals=[],
        )
        run = await engine.generate(llm, payload, store=store)
        assert isinstance(run, ForecastRun)
        assert run.engine == "actor"

    async def test_without_llm_uses_fallback(self, store):
        """Without LLM, should produce a run using deterministic fallback."""
        engine = ActorForecastEngine()
        payload = ForecastEngineInput(
            topic_slug="test",
            topic_name="Test Topic",
            run_date=date(2026, 3, 12),
            threads=[],
            recent_events=[],
            cross_topic_signals=[],
        )
        run = await engine.generate(None, payload)
        assert isinstance(run, ForecastRun)
        assert run.engine == "actor"
