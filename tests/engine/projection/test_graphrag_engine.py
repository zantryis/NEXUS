"""Tests for GraphRAG-enhanced prediction engine — graph traversal + structured reasoning."""

from __future__ import annotations

import json
from datetime import date, timedelta
from unittest.mock import AsyncMock

import pytest

from nexus.engine.knowledge.events import Event
from nexus.engine.knowledge.store import KnowledgeStore
from nexus.engine.projection.graphrag_engine import (
    GraphRAGBenchmarkEngine,
    GraphRAGForecastEngine,
    extract_entities_from_question,
    gather_graph_evidence,
    rank_evidence,
)


@pytest.fixture
async def store(tmp_path):
    s = KnowledgeStore(tmp_path / "graphrag-test.db")
    await s.initialize()
    yield s
    await s.close()


def _make_mock_llm(response: dict | str):
    if isinstance(response, dict):
        response = json.dumps(response)

    async def mock_complete(*, config_key, system_prompt, user_prompt, json_response=False):
        return response

    llm = AsyncMock()
    llm.complete = mock_complete
    return llm


async def _seed_entities_and_events(store):
    """Seed a small knowledge graph for testing."""
    # Add events that create entities
    events = [
        Event(
            date=date(2026, 3, 1),
            summary="Saudi Arabia announces diplomatic initiative toward Israel",
            entities=["Saudi Arabia", "Israel"],
            significance=8,
        ),
        Event(
            date=date(2026, 3, 5),
            summary="US brokers meeting between Saudi and Israeli officials",
            entities=["United States", "Saudi Arabia", "Israel"],
            significance=9,
        ),
        Event(
            date=date(2026, 3, 8),
            summary="Iran condemns Saudi-Israel talks",
            entities=["Iran", "Saudi Arabia", "Israel"],
            significance=7,
        ),
    ]
    await store.add_events(events, "geopolitics")

    # Add entity relationships
    entities = await store.get_all_entities()
    entity_map = {e["canonical_name"]: e["id"] for e in entities}

    if "Saudi Arabia" in entity_map and "Israel" in entity_map:
        await store.save_entity_relationship({
            "source_entity_id": entity_map["Saudi Arabia"],
            "target_entity_id": entity_map["Israel"],
            "relation_type": "diplomatic_engagement",
            "evidence_text": "Normalization talks brokered by US",
            "strength": 0.8,
            "valid_from": date(2026, 3, 1).isoformat(),
        })
    if "Iran" in entity_map and "Saudi Arabia" in entity_map:
        await store.save_entity_relationship({
            "source_entity_id": entity_map["Iran"],
            "target_entity_id": entity_map["Saudi Arabia"],
            "relation_type": "opposition",
            "evidence_text": "Iran opposes Saudi-Israel normalization",
            "strength": 0.7,
            "valid_from": date(2026, 3, 8).isoformat(),
        })

    return entity_map


# ── extract_entities_from_question ────────────────────────────────────


class TestExtractEntities:
    async def test_extracts_via_llm(self, store):
        """LLM-based extraction should return entity names."""
        llm = _make_mock_llm({"entities": ["Saudi Arabia", "Israel"]})
        entities = await extract_entities_from_question(
            store, llm, "Will Saudi Arabia normalize relations with Israel?"
        )
        names = [e["name"] for e in entities]
        assert "Saudi Arabia" in names or "Israel" in names

    async def test_resolves_against_store(self, store):
        """Extracted entities should be resolved against the store."""
        await _seed_entities_and_events(store)
        llm = _make_mock_llm({"entities": ["Saudi Arabia", "Israel"]})

        entities = await extract_entities_from_question(
            store, llm, "Will Saudi Arabia normalize relations with Israel?"
        )

        # Should have entity_ids from store resolution
        resolved = [e for e in entities if e.get("entity_id") is not None]
        assert len(resolved) >= 1

    async def test_falls_back_to_keyword_matching(self, store):
        """Without LLM, should use keyword matching."""
        await _seed_entities_and_events(store)

        entities = await extract_entities_from_question(
            store, None, "Will Saudi Arabia normalize relations with Israel?"
        )

        names = [e["name"] for e in entities]
        assert len(names) >= 1  # Should find at least one via keywords


# ── gather_graph_evidence ─────────────────────────────────────────────


class TestGatherGraphEvidence:
    async def test_includes_relationships(self, store):
        """Evidence should include entity relationships."""
        entity_map = await _seed_entities_and_events(store)

        sa_id = entity_map.get("Saudi Arabia")
        if sa_id is None:
            pytest.skip("Saudi Arabia entity not created")

        evidence = await gather_graph_evidence(
            store,
            entity_ids=[sa_id],
            as_of=date(2026, 3, 10),
        )

        assert len(evidence["relationships"]) >= 1

    async def test_includes_events(self, store):
        """Evidence should include events for entities."""
        entity_map = await _seed_entities_and_events(store)

        sa_id = entity_map.get("Saudi Arabia")
        if sa_id is None:
            pytest.skip("Saudi Arabia entity not created")

        evidence = await gather_graph_evidence(
            store,
            entity_ids=[sa_id],
            as_of=date(2026, 3, 10),
        )

        assert len(evidence["events"]) >= 1

    async def test_multi_hop_neighbors(self, store):
        """Should discover entities connected through the graph."""
        entity_map = await _seed_entities_and_events(store)

        sa_id = entity_map.get("Saudi Arabia")
        if sa_id is None:
            pytest.skip("Saudi Arabia entity not created")

        evidence = await gather_graph_evidence(
            store,
            entity_ids=[sa_id],
            as_of=date(2026, 3, 10),
        )

        # Saudi Arabia is connected to Israel and Iran
        neighbor_names = {n.get("name", "") for n in evidence.get("neighbors", [])}
        assert len(neighbor_names) >= 1


# ── rank_evidence ─────────────────────────────────────────────────────


class TestRankEvidence:
    def test_ranks_by_recency(self):
        """More recent events should rank higher."""
        evidence = {
            "events": [
                {"date": "2026-03-01", "summary": "Old event", "significance": 5},
                {"date": "2026-03-10", "summary": "Recent event", "significance": 5},
            ],
            "relationships": [],
            "neighbors": [],
        }
        ranked = rank_evidence(evidence, as_of=date(2026, 3, 10))
        assert ranked["events"][0]["summary"] == "Recent event"

    def test_ranks_by_significance(self):
        """Higher significance events should rank higher."""
        evidence = {
            "events": [
                {"date": "2026-03-10", "summary": "Low sig", "significance": 3},
                {"date": "2026-03-10", "summary": "High sig", "significance": 9},
            ],
            "relationships": [],
            "neighbors": [],
        }
        ranked = rank_evidence(evidence, as_of=date(2026, 3, 10))
        assert ranked["events"][0]["summary"] == "High sig"

    def test_caps_evidence(self):
        """Should cap evidence to prevent prompt bloat."""
        evidence = {
            "events": [
                {"date": f"2026-03-{i:02d}", "summary": f"Event {i}", "significance": 5}
                for i in range(1, 25)
            ],
            "relationships": [],
            "neighbors": [],
        }
        ranked = rank_evidence(evidence, as_of=date(2026, 3, 15), max_events=10)
        assert len(ranked["events"]) == 10


# ── GraphRAGBenchmarkEngine ──────────────────────────────────────────


class TestGraphRAGBenchmarkEngine:
    async def test_returns_bounded_probability(self, store):
        """Should return probability in [0.05, 0.95]."""
        await _seed_entities_and_events(store)
        llm = _make_mock_llm({"entities": ["Saudi Arabia"], "probability": 0.65, "reasoning": "test"})
        engine = GraphRAGBenchmarkEngine()
        prob = await engine.predict_probability(
            "Will Saudi Arabia normalize?",
            llm=llm,
            store=store,
            as_of=date(2026, 3, 10),
        )
        assert 0.05 <= prob <= 0.95

    async def test_engine_name(self):
        assert GraphRAGBenchmarkEngine().engine_name == "graphrag"

    async def test_works_without_store(self):
        """Should fallback gracefully without knowledge store."""
        llm = _make_mock_llm({"entities": [], "probability": 0.50, "reasoning": "no data"})
        engine = GraphRAGBenchmarkEngine()
        prob = await engine.predict_probability(
            "Will X happen?", llm=llm, as_of=date(2026, 3, 10)
        )
        assert 0.05 <= prob <= 0.95


class TestGraphRAGForecastEngine:
    def test_engine_name(self):
        assert GraphRAGForecastEngine().engine_name == "graphrag"
