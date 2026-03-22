"""Tests for thread repair/backfill utilities."""

from datetime import date

import pytest

from nexus.engine.knowledge.events import Event
from nexus.engine.knowledge.store import KnowledgeStore
from nexus.engine.synthesis.knowledge import NarrativeThread, TopicSynthesis
from nexus.engine.synthesis.repair import repair_thread_hygiene


@pytest.fixture
async def store(tmp_path):
    s = KnowledgeStore(tmp_path / "repair.db")
    await s.initialize()
    yield s
    await s.close()


async def test_repair_thread_hygiene_merges_duplicates_and_hydrates_syntheses(store):
    event_a = Event(
        date=date(2026, 3, 8),
        summary="Iran sanctions tightened",
        significance=8,
        entities=["Iran", "IAEA", "US"],
        sources=[],
    )
    event_b = Event(
        date=date(2026, 3, 10),
        summary="Iran sanctions response",
        significance=7,
        entities=["Iran", "IAEA", "EU"],
        sources=[],
    )
    ids = await store.add_events([event_a, event_b], "iran-us")

    for entity_name in ["Iran", "IAEA", "US", "EU"]:
        entity_id = await store.upsert_entity(entity_name, "country")
        for event_id in ids:
            await store.link_event_entities(event_id, [entity_id])

    keep_id = await store.upsert_thread("iran-sanctions-a", "Iran Sanctions Push", 8, "active")
    absorb_id = await store.upsert_thread("iran-sanctions-b", "Iran Sanctions Drive", 5, "active")
    await store.link_thread_topic(keep_id, "iran-us")
    await store.link_thread_topic(absorb_id, "iran-us")
    await store.link_thread_events(keep_id, [ids[0]])
    await store.link_thread_events(absorb_id, [ids[1]])

    synthesis = TopicSynthesis(
        topic_name="Iran-US",
        threads=[NarrativeThread(
            headline="Iran Sanctions Push",
            events=[
                Event(date=date(2026, 3, 8), summary="Iran sanctions tightened", significance=8, entities=["Iran"]),
                Event(date=date(2026, 3, 10), summary="Iran sanctions response", significance=7, entities=["Iran"]),
            ],
            significance=8,
        )],
    )
    await store.save_synthesis(synthesis.model_dump(mode="json"), "iran-us", date(2026, 3, 10))

    result = await repair_thread_hygiene(store, None, run_date=date(2026, 3, 21))

    visible_threads = await store.get_all_threads()
    merged_threads = await store.get_all_threads(status="merged")
    repaired = await store.get_synthesis("iran-us", date(2026, 3, 10))
    repaired_thread = repaired["threads"][0]

    assert result["merged_pairs"] >= 1
    assert len(visible_threads) == 1
    assert visible_threads[0]["id"] == keep_id
    assert len(merged_threads) == 1
    assert merged_threads[0]["id"] == absorb_id
    assert repaired_thread["thread_id"] == keep_id
    assert repaired_thread["slug"] == "iran-sanctions-a"
    assert await store.get_thread_snapshots(keep_id)
