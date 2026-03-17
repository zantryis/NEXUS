"""Tests for entity-entity relationships — extraction, invalidation, graph queries."""

import json
import pytest
from datetime import date
from unittest.mock import AsyncMock, patch

from nexus.engine.knowledge.store import KnowledgeStore


@pytest.fixture
async def store(tmp_path):
    s = KnowledgeStore(tmp_path / "knowledge.db")
    await s.initialize()
    yield s
    await s.close()


async def _seed_entities(store: KnowledgeStore) -> dict[str, int]:
    """Insert test entities, return name→id mapping."""
    ids = {}
    for name, etype in [
        ("Iran", "country"),
        ("Israel", "country"),
        ("United States", "country"),
        ("China", "country"),
        ("Hamas", "org"),
        ("IRGC", "org"),
    ]:
        ids[name] = await store.upsert_entity(name, etype)
    return ids


async def _seed_events(store: KnowledgeStore) -> list[int]:
    """Insert test events, return event IDs."""
    from nexus.engine.knowledge.events import Event

    events = [
        Event(
            date=date(2026, 3, 1),
            summary="Iran launches missile barrage targeting Haifa",
            significance=9,
            relation_to_prior="escalation",
            entities=["Iran", "Israel"],
            sources=[{"url": "https://reuters.com/1", "outlet": "reuters",
                      "affiliation": "private", "country": "US", "language": "en"}],
        ),
        Event(
            date=date(2026, 3, 5),
            summary="US Treasury expands sanctions on Iranian shipping",
            significance=7,
            relation_to_prior="response",
            entities=["United States", "Iran"],
            sources=[{"url": "https://bbc.com/1", "outlet": "bbc",
                      "affiliation": "public", "country": "GB", "language": "en"}],
        ),
        Event(
            date=date(2026, 3, 10),
            summary="Beijing proposes ceasefire framework via back channels",
            significance=6,
            relation_to_prior="diplomatic",
            entities=["China", "Iran"],
            sources=[{"url": "https://scmp.com/1", "outlet": "scmp",
                      "affiliation": "private", "country": "HK", "language": "en"}],
        ),
    ]
    return await store.add_events(events, "iran-us-relations")


async def _seed_relationships(store: KnowledgeStore, entities: dict, event_ids: list[int]):
    """Insert test relationships."""
    rels = [
        {
            "source_entity_id": entities["Iran"],
            "target_entity_id": entities["Israel"],
            "relation_type": "threatens",
            "evidence_text": "Iran launches missile barrage targeting Haifa",
            "source_event_id": event_ids[0],
            "strength": 0.9,
            "valid_from": "2026-03-01",
        },
        {
            "source_entity_id": entities["United States"],
            "target_entity_id": entities["Iran"],
            "relation_type": "sanctions",
            "evidence_text": "US Treasury expands sanctions on Iranian shipping",
            "source_event_id": event_ids[1],
            "strength": 0.85,
            "valid_from": "2026-03-05",
        },
        {
            "source_entity_id": entities["Iran"],
            "target_entity_id": entities["China"],
            "relation_type": "negotiates_with",
            "evidence_text": "Beijing proposes ceasefire framework",
            "source_event_id": event_ids[2],
            "strength": 0.4,
            "valid_from": "2026-03-10",
        },
        {
            "source_entity_id": entities["Israel"],
            "target_entity_id": entities["United States"],
            "relation_type": "allies_with",
            "evidence_text": "Long-standing alliance",
            "source_event_id": event_ids[0],
            "strength": 0.95,
            "valid_from": "2026-01-01",
        },
        {
            "source_entity_id": entities["Iran"],
            "target_entity_id": entities["Hamas"],
            "relation_type": "allies_with",
            "evidence_text": "Axis of resistance partnership",
            "source_event_id": event_ids[0],
            "strength": 0.7,
            "valid_from": "2025-11-20",
        },
    ]
    ids = []
    for rel in rels:
        rid = await store.save_entity_relationship(rel)
        ids.append(rid)
    return ids


# ── Store CRUD ──────────────────────────────────────────────────────


class TestSaveAndQueryRelationships:

    async def test_save_entity_relationship_returns_id(self, store):
        entities = await _seed_entities(store)
        event_ids = await _seed_events(store)
        rid = await store.save_entity_relationship({
            "source_entity_id": entities["Iran"],
            "target_entity_id": entities["Israel"],
            "relation_type": "threatens",
            "evidence_text": "missile strike",
            "source_event_id": event_ids[0],
            "strength": 0.9,
            "valid_from": "2026-03-01",
        })
        assert isinstance(rid, int)
        assert rid > 0

    async def test_duplicate_relationship_idempotent(self, store):
        """UNIQUE constraint on (source, target, type, event) prevents duplicates."""
        entities = await _seed_entities(store)
        event_ids = await _seed_events(store)
        rel = {
            "source_entity_id": entities["Iran"],
            "target_entity_id": entities["Israel"],
            "relation_type": "threatens",
            "evidence_text": "missile strike",
            "source_event_id": event_ids[0],
            "strength": 0.9,
            "valid_from": "2026-03-01",
        }
        id1 = await store.save_entity_relationship(rel)
        id2 = await store.save_entity_relationship(rel)
        assert id1 == id2  # Same row, not a new one

    async def test_get_active_relationships_for_entity(self, store):
        entities = await _seed_entities(store)
        event_ids = await _seed_events(store)
        await _seed_relationships(store, entities, event_ids)

        rels = await store.get_active_relationships_for_entity(entities["Iran"])
        # Iran has: threatens Israel, negotiates_with China, allies_with Hamas
        # Plus: US sanctions Iran (Iran is target)
        assert len(rels) >= 3
        rel_types = {r["relation_type"] for r in rels}
        assert "threatens" in rel_types
        assert "negotiates_with" in rel_types

    async def test_get_active_relationships_respects_as_of(self, store):
        entities = await _seed_entities(store)
        event_ids = await _seed_events(store)
        await _seed_relationships(store, entities, event_ids)

        # As of March 3, only the March 1 relationships should exist
        rels = await store.get_active_relationships_for_entity(
            entities["Iran"], as_of=date(2026, 3, 3)
        )
        rel_types = {r["relation_type"] for r in rels}
        assert "threatens" in rel_types
        # negotiates_with is from March 10, should NOT appear
        assert "negotiates_with" not in rel_types

    async def test_get_relationships_between(self, store):
        entities = await _seed_entities(store)
        event_ids = await _seed_events(store)
        await _seed_relationships(store, entities, event_ids)

        rels = await store.get_relationships_between(
            entities["Iran"], entities["Israel"]
        )
        assert len(rels) == 1
        assert rels[0]["relation_type"] == "threatens"
        assert rels[0]["strength"] == 0.9


# ── Invalidation ────────────────────────────────────────────────────


class TestInvalidation:

    async def test_invalidate_relationship_sets_valid_until(self, store):
        entities = await _seed_entities(store)
        event_ids = await _seed_events(store)
        rel_ids = await _seed_relationships(store, entities, event_ids)

        # Invalidate Iran allies_with Hamas (last relationship in seed)
        iran_hamas_id = rel_ids[4]
        await store.invalidate_relationship(iran_hamas_id, date(2026, 3, 8))

        # Active relationships for Iran should no longer include Hamas
        rels = await store.get_active_relationships_for_entity(entities["Iran"])
        rel_targets = {
            (r["relation_type"], r.get("target_entity_name") or r.get("target_entity_id"))
            for r in rels
        }
        # Check Hamas relationship is gone from active
        hamas_rels = [
            r for r in rels
            if r.get("target_entity_id") == entities["Hamas"]
            or r.get("source_entity_id") == entities["Hamas"]
        ]
        assert len(hamas_rels) == 0

    async def test_invalidated_relationship_visible_before_cutoff(self, store):
        """A relationship invalidated on March 8 should be visible as_of March 7."""
        entities = await _seed_entities(store)
        event_ids = await _seed_events(store)
        rel_ids = await _seed_relationships(store, entities, event_ids)

        await store.invalidate_relationship(rel_ids[4], date(2026, 3, 8))

        # As of March 7 (before invalidation), Hamas relationship should be active
        rels = await store.get_active_relationships_for_entity(
            entities["Iran"], as_of=date(2026, 3, 7)
        )
        hamas_rels = [
            r for r in rels
            if r.get("target_entity_id") == entities["Hamas"]
            or r.get("source_entity_id") == entities["Hamas"]
        ]
        assert len(hamas_rels) == 1


# ── Graph Queries ───────────────────────────────────────────────────


class TestEntityNeighborhood:

    async def test_one_hop_returns_direct_neighbors(self, store):
        entities = await _seed_entities(store)
        event_ids = await _seed_events(store)
        await _seed_relationships(store, entities, event_ids)

        result = await store.get_entity_neighborhood(entities["Iran"], hops=1)
        neighbor_ids = {e["id"] for e in result["entities"]}
        # Iran → Israel (threatens), Iran → China (negotiates), Iran → Hamas (allies)
        # US → Iran (sanctions) — so US is also a neighbor
        assert entities["Israel"] in neighbor_ids
        assert entities["China"] in neighbor_ids
        assert entities["Hamas"] in neighbor_ids
        assert entities["United States"] in neighbor_ids
        # Iran itself should not be in the neighbor list
        assert entities["Iran"] not in neighbor_ids

    async def test_two_hop_returns_extended_neighbors(self, store):
        entities = await _seed_entities(store)
        event_ids = await _seed_events(store)
        await _seed_relationships(store, entities, event_ids)

        result = await store.get_entity_neighborhood(entities["Iran"], hops=2)
        neighbor_ids = {e["id"] for e in result["entities"]}
        # Hop 1: Israel, China, Hamas, US
        # Hop 2 from Israel: US (allies_with) — already in hop 1
        # All 4 direct neighbors should be present
        assert entities["Israel"] in neighbor_ids
        assert entities["United States"] in neighbor_ids
        assert len(result["relationships"]) >= 4

    async def test_neighborhood_includes_relationship_details(self, store):
        entities = await _seed_entities(store)
        event_ids = await _seed_events(store)
        await _seed_relationships(store, entities, event_ids)

        result = await store.get_entity_neighborhood(entities["Iran"], hops=1)
        # Each relationship should have type, strength, evidence
        for rel in result["relationships"]:
            assert "relation_type" in rel
            assert "strength" in rel
            assert "valid_from" in rel

    async def test_neighborhood_respects_as_of(self, store):
        entities = await _seed_entities(store)
        event_ids = await _seed_events(store)
        await _seed_relationships(store, entities, event_ids)

        # As of March 3, only March 1 relationships exist
        result = await store.get_entity_neighborhood(
            entities["Iran"], hops=1, as_of=date(2026, 3, 3)
        )
        neighbor_ids = {e["id"] for e in result["entities"]}
        assert entities["Israel"] in neighbor_ids
        assert entities["Hamas"] in neighbor_ids
        # China relationship is from March 10 — should not appear
        assert entities["China"] not in neighbor_ids


class TestRelationshipTimeline:

    async def test_timeline_shows_recent_changes(self, store):
        entities = await _seed_entities(store)
        event_ids = await _seed_events(store)
        rel_ids = await _seed_relationships(store, entities, event_ids)

        # Invalidate Hamas relationship
        await store.invalidate_relationship(rel_ids[4], date(2026, 3, 8))

        timeline = await store.get_relationship_timeline(entities["Iran"], days=30)
        # Should include both new relationships and the invalidation
        assert len(timeline) >= 3
        # Check that we see the invalidated relationship
        invalidated = [t for t in timeline if t.get("valid_until") is not None]
        assert len(invalidated) >= 1

    async def test_timeline_limited_to_days(self, store):
        entities = await _seed_entities(store)
        event_ids = await _seed_events(store)
        await _seed_relationships(store, entities, event_ids)

        # Only last 5 days from a reference date of March 12
        timeline = await store.get_relationship_timeline(
            entities["Iran"], days=5, reference_date=date(2026, 3, 12)
        )
        # Should only include relationships from March 7+
        # That means: negotiates_with China (Mar 10), sanctions from US (Mar 5 is outside)
        for entry in timeline:
            assert entry["valid_from"] >= "2026-03-07"


# ── Extraction (mock LLM) ──────────────────────────────────────────


class TestExtraction:

    async def test_extract_relationships_returns_typed_pairs(self):
        from nexus.engine.knowledge.relationships import (
            extract_relationships_from_event,
            ExtractedRelationship,
        )
        from nexus.engine.knowledge.events import Event

        event = Event(
            date=date(2026, 3, 1),
            summary="Iran launches missile barrage targeting Haifa, Israel",
            significance=9,
            relation_to_prior="escalation",
            entities=["Iran", "Israel"],
            sources=[],
        )

        mock_llm = AsyncMock()
        mock_llm.complete.return_value = json.dumps({
            "relationships": [
                {
                    "source_entity": "Iran",
                    "target_entity": "Israel",
                    "relation_type": "threatens",
                    "evidence_text": "Iran launches missile barrage targeting Haifa",
                    "strength": 0.9,
                },
            ]
        })

        result = await extract_relationships_from_event(
            mock_llm, event, existing_relationships=[]
        )
        assert len(result) == 1
        assert isinstance(result[0], ExtractedRelationship)
        assert result[0].source_entity == "Iran"
        assert result[0].target_entity == "Israel"
        assert result[0].relation_type == "threatens"
        assert result[0].strength == 0.9

    async def test_extract_relationships_handles_empty_response(self):
        from nexus.engine.knowledge.relationships import extract_relationships_from_event
        from nexus.engine.knowledge.events import Event

        event = Event(
            date=date(2026, 3, 1),
            summary="Minor administrative update",
            significance=2,
            relation_to_prior="routine",
            entities=["IAEA"],
            sources=[],
        )

        mock_llm = AsyncMock()
        mock_llm.complete.return_value = json.dumps({"relationships": []})

        result = await extract_relationships_from_event(
            mock_llm, event, existing_relationships=[]
        )
        assert result == []

    async def test_extract_relationships_filters_invalid_types(self):
        from nexus.engine.knowledge.relationships import extract_relationships_from_event
        from nexus.engine.knowledge.events import Event

        event = Event(
            date=date(2026, 3, 1),
            summary="Some event",
            significance=5,
            relation_to_prior="continuation",
            entities=["Iran", "Israel"],
            sources=[],
        )

        mock_llm = AsyncMock()
        mock_llm.complete.return_value = json.dumps({
            "relationships": [
                {
                    "source_entity": "Iran",
                    "target_entity": "Israel",
                    "relation_type": "threatens",
                    "evidence_text": "valid",
                    "strength": 0.9,
                },
                {
                    "source_entity": "Iran",
                    "target_entity": "Israel",
                    "relation_type": "hugs_warmly",  # invalid type
                    "evidence_text": "invalid",
                    "strength": 0.5,
                },
            ]
        })

        result = await extract_relationships_from_event(
            mock_llm, event, existing_relationships=[]
        )
        assert len(result) == 1
        assert result[0].relation_type == "threatens"

    async def test_extract_handles_llm_failure(self):
        from nexus.engine.knowledge.relationships import extract_relationships_from_event
        from nexus.engine.knowledge.events import Event

        event = Event(
            date=date(2026, 3, 1),
            summary="Some event",
            significance=5,
            relation_to_prior="continuation",
            entities=["Iran", "Israel"],
            sources=[],
        )

        mock_llm = AsyncMock()
        mock_llm.complete.side_effect = Exception("API error")

        result = await extract_relationships_from_event(
            mock_llm, event, existing_relationships=[]
        )
        assert result == []


# ── Contradiction Detection ─────────────────────────────────────────


class TestContradictionDetection:

    async def test_invalidate_contradicted_relationships(self, store):
        from nexus.engine.knowledge.relationships import (
            invalidate_contradicted_relationships,
            ExtractedRelationship,
        )

        entities = await _seed_entities(store)
        event_ids = await _seed_events(store)
        await _seed_relationships(store, entities, event_ids)

        # New relationship: Iran opposes Hamas (contradicts allies_with)
        new_rels = [
            ExtractedRelationship(
                source_entity="Iran",
                target_entity="Hamas",
                relation_type="opposes",
                evidence_text="Leadership change causes split",
                strength=0.6,
                valid_from=date(2026, 3, 8),
            ),
        ]

        count = await invalidate_contradicted_relationships(
            store, new_rels, date(2026, 3, 8)
        )
        assert count >= 1

        # The old allies_with should now be invalidated
        rels = await store.get_active_relationships_for_entity(entities["Iran"])
        hamas_allies = [
            r for r in rels
            if r.get("target_entity_id") == entities["Hamas"]
            and r["relation_type"] == "allies_with"
        ]
        assert len(hamas_allies) == 0
