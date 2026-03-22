"""Case-scoped thread persistence tests."""

from datetime import date
from unittest.mock import AsyncMock

import pytest

from nexus.engine.knowledge.events import Event
from nexus.engine.knowledge.store import KnowledgeStore
from nexus.engine.synthesis.knowledge import NarrativeThread, TopicSynthesis, _persist_threads


@pytest.mark.asyncio
async def test_persist_threads_links_case_threads_without_leaking_into_topic_queries(tmp_path):
    store = KnowledgeStore(tmp_path / "knowledge.db")
    await store.initialize()

    case_id = await store.upsert_case(
        "mh370",
        "MH370",
        "What happened to MH370?",
        time_bounds={"start": "2014-03-08", "end": "2026-03-21"},
        build_defaults={"max_documents": 12},
        monitoring_enabled=True,
    )

    event = Event(
        date=date(2014, 3, 8),
        summary="Investigators narrowed the likely end-state to the southern Indian Ocean.",
        entities=["MH370", "ATSB", "Southern Indian Ocean"],
        sources=[{"url": "https://example.com/report", "outlet": "ATSB", "affiliation": "official"}],
        significance=8,
    )
    await store.add_events([event], "__case__:mh370", case_id=case_id)

    synthesis = TopicSynthesis(
        topic_name="MH370",
        threads=[
            NarrativeThread(
                headline="The Search for MH370 in the Southern Indian Ocean",
                events=[event],
                key_entities=["MH370", "ATSB", "Southern Indian Ocean"],
                significance=9,
                convergence=[{"fact": "Multiple evidence lines point south.", "confirmed_by": ["ATSB", "independent analysis"]}],
                divergence=[
                    {
                        "shared_event": "Search-area confidence",
                        "source_a": "ATSB",
                        "framing_a": "The official zone remains the leading area.",
                        "source_b": "Independent analysis",
                        "framing_b": "The broad southern corridor is right, but the specific zone is off.",
                    }
                ],
            )
        ],
    )

    await _persist_threads(store, AsyncMock(), synthesis, [event], case_id=case_id)

    case_threads = await store.get_threads_for_case(case_id)
    visible_topic_threads = await store.get_all_threads()
    convergence = await store.get_convergence_for_thread(case_threads[0]["id"])
    divergence = await store.get_divergence_for_thread(case_threads[0]["id"])

    assert len(case_threads) == 1
    assert case_threads[0]["headline"] == "The Search for MH370 in the Southern Indian Ocean"
    assert visible_topic_threads == []
    assert convergence[0]["fact_text"] == "Multiple evidence lines point south."
    assert divergence[0]["shared_event"] == "Search-area confidence"

    await store.close()

