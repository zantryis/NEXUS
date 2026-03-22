"""Case-scoped persistence tests for the shared knowledge store."""

from datetime import date

import pytest

from nexus.casefiles.models import CaseAssessment
from nexus.engine.knowledge.events import Event
from nexus.engine.knowledge.store import KnowledgeStore


@pytest.fixture
async def store(tmp_path):
    s = KnowledgeStore(tmp_path / "knowledge.db")
    await s.initialize()
    yield s
    await s.close()


def _case_event(summary: str, *, entities: list[str] | None = None) -> Event:
    return Event(
        date=date(2014, 3, 8),
        summary=summary,
        significance=8,
        relation_to_prior="case development",
        entities=entities or ["MH370", "Malaysia"],
        sources=[
            {
                "url": "https://example.com/mh370",
                "outlet": "ATSB",
                "affiliation": "official",
                "country": "AU",
                "language": "en",
            }
        ],
    )


@pytest.mark.asyncio
async def test_case_scope_events_and_stats_are_isolated_from_topics(store):
    case_id = await store.upsert_case(
        "mh370",
        "MH370",
        "What happened to MH370?",
        time_bounds={"start": "2014-03-08", "end": "2026-03-21"},
        build_defaults={"max_documents": 12},
        monitoring_enabled=True,
    )
    await store.add_events([_case_event("MH370 vanishes from radar")], "__case__:mh370", case_id=case_id)
    await store.add_events(
        [
            Event(
                date=date(2026, 3, 20),
                summary="Topic event",
                significance=5,
                relation_to_prior="topic development",
                entities=["OpenAI"],
                sources=[
                    {
                        "url": "https://example.com/topic",
                        "outlet": "ArXiv",
                        "affiliation": "academic",
                        "country": "US",
                        "language": "en",
                    }
                ],
            )
        ],
        "ai-ml",
    )

    case_events = await store.get_events(case_id=case_id)
    assert [event.summary for event in case_events] == ["MH370 vanishes from radar"]

    global_events = await store.get_all_events()
    assert [event.summary for event in global_events] == ["Topic event"]

    topic_stats = await store.get_topic_stats()
    assert [row["topic_slug"] for row in topic_stats] == ["ai-ml"]

    global_sources = await store.get_source_stats()
    assert {row["outlet"] for row in global_sources} == {"ArXiv"}
    case_sources = await store.get_source_stats(case_id=case_id)
    assert {row["outlet"] for row in case_sources} == {"ATSB"}


@pytest.mark.asyncio
async def test_case_documents_evidence_hypotheses_and_questions_roundtrip(store):
    case_id = await store.upsert_case(
        "mh370",
        "MH370",
        "What happened to MH370?",
        time_bounds={"start": "2014-03-08", "end": "2026-03-21"},
        build_defaults={"max_documents": 12},
        monitoring_enabled=True,
    )

    await store.replace_case_documents(
        case_id,
        [
            {
                "id": "doc-1",
                "title": "Official MH370 Report",
                "url": "https://example.com/report",
                "canonical_url": "https://example.com/report",
                "kind": "report",
                "role": "primary",
                "source_class": "official",
                "source_label": "ATSB",
                "priority": 10,
                "summary": "Official report summary",
            }
        ],
    )
    await store.replace_case_evidence(
        case_id,
        [
            {
                "id": "E001",
                "claim": "Satellite data points south.",
                "stance": "supports",
                "quality_label": "high",
                "summary": "Ping arc analysis favored the southern corridor.",
                "document_id": "doc-1",
                "document_title": "Official MH370 Report",
                "document_url": "https://example.com/report",
                "source_label": "ATSB",
                "source_class": "official",
                "related_hypotheses": ["H01"],
            }
        ],
    )
    await store.replace_case_hypotheses(
        case_id,
        [
            {
                "id": "H01",
                "title": "Deliberate diversion",
                "summary": "Manual deviation followed by southern Indian Ocean impact.",
                "confidence_label": "Leading",
                "evidence_for": ["E001"],
                "evidence_against": [],
                "unresolved_gaps": ["Who acted and why?"],
                "what_would_change_my_mind": ["A verified systems-failure chain."],
            }
        ],
    )
    await store.replace_case_open_questions(case_id, ["Where is the main fuselage?"])

    assessment = CaseAssessment(
        id="A-h01",
        target_hypothesis_id="H01",
        mode="posterior",
        question="What is the probability of deliberate diversion?",
        probability=0.68,
        confidence="medium",
        rationale="The evidence lines cluster around a controlled southbound end-state.",
        counterarguments=["No direct cockpit evidence survives."],
        evidence_ids=["E001"],
        evidence_thread_ids=[1],
        signposts=["Recovery of additional debris with trace evidence."],
    )
    await store.replace_case_assessments(case_id, [assessment.model_dump(mode="json")])

    documents = await store.get_case_documents(case_id)
    evidence = await store.get_case_evidence(case_id)
    hypotheses = await store.get_case_hypotheses(case_id)
    questions = await store.get_case_open_questions(case_id)
    assessments = await store.get_case_assessments(case_id)

    assert documents[0]["id"] == "doc-1"
    assert evidence[0]["id"] == "E001"
    assert hypotheses[0]["id"] == "H01"
    assert questions == ["Where is the main fuselage?"]
    assert assessments[0]["id"] == "A-h01"
    assert assessments[0]["probability"] == pytest.approx(0.68)


@pytest.mark.asyncio
async def test_case_threads_graph_and_divergence_are_scoped(store):
    case_id = await store.upsert_case(
        "mh370",
        "MH370",
        "What happened to MH370?",
        time_bounds={"start": "2014-03-08", "end": "2026-03-21"},
        build_defaults={"max_documents": 12},
        monitoring_enabled=True,
    )

    case_event = _case_event(
        "Investigators focus on the southern Indian Ocean",
        entities=["MH370", "ATSB", "Southern Indian Ocean"],
    )
    topic_event = Event(
        date=date(2026, 3, 20),
        summary="AI benchmark ships",
        significance=5,
        relation_to_prior="topic development",
        entities=["OpenAI", "Google"],
        sources=[{"url": "https://example.com/ai", "outlet": "ArXiv", "affiliation": "academic"}],
    )
    case_event_id = (await store.add_events([case_event], "__case__:mh370", case_id=case_id))[0]
    topic_event_id = (await store.add_events([topic_event], "ai-ml"))[0]

    mh370_id = await store.upsert_entity("MH370", "concept")
    atsb_id = await store.upsert_entity("ATSB", "org")
    sio_id = await store.upsert_entity("Southern Indian Ocean", "concept")
    openai_id = await store.upsert_entity("OpenAI", "org")
    google_id = await store.upsert_entity("Google", "org")
    await store.link_event_entities(case_event_id, [mh370_id, atsb_id, sio_id])
    await store.link_event_entities(topic_event_id, [openai_id, google_id])

    case_thread_id = await store.upsert_thread(
        "southern-indian-ocean-search",
        "The Search for MH370 in the Southern Indian Ocean",
        9,
        "active",
    )
    await store.link_thread_case(case_thread_id, case_id, relevance=0.9, role="search")
    await store.link_thread_events(case_thread_id, [case_event_id])
    await store.add_divergence(
        case_thread_id,
        "Search-area confidence",
        "ATSB",
        "The priority area remains the strongest candidate zone.",
        "Independent analysis",
        "The priority area is materially mislocated.",
    )

    topic_thread_id = await store.upsert_thread("ai-benchmarks", "AI Benchmarks", 5, "active")
    await store.link_thread_topic(topic_thread_id, "ai-ml")
    await store.link_thread_events(topic_thread_id, [topic_event_id])

    case_threads = await store.get_threads_for_case(case_id)
    visible_threads = await store.get_all_threads()
    divergence = await store.get_case_divergence(case_id)
    graph = await store.get_case_graph_data(case_id)

    assert [thread["slug"] for thread in case_threads] == ["southern-indian-ocean-search"]
    assert [thread["slug"] for thread in visible_threads] == ["ai-benchmarks"]
    assert divergence[0]["shared_event"] == "Search-area confidence"

    node_names = {node["name"] for node in graph["nodes"]}
    assert node_names == {"MH370", "ATSB", "Southern Indian Ocean"}
    assert graph["links"]


@pytest.mark.asyncio
async def test_reset_case_scope_tolerates_threads_referenced_by_merged_threads(store):
    case_id = await store.upsert_case(
        "jfk-assassination",
        "JFK Assassination",
        "What happened?",
        time_bounds={"start": "1963-11-22", "end": "2026-03-21"},
        build_defaults={"max_documents": 12},
        monitoring_enabled=True,
    )

    event_id = (await store.add_events([_case_event("Dealey Plaza evidence review")], "__case__:jfk-assassination", case_id=case_id))[0]
    keep_id = await store.upsert_thread("ballistics-thread", "Ballistics Thread", 8, "active")
    absorb_id = await store.upsert_thread("acoustics-thread", "Acoustics Thread", 6, "merged")
    await store.link_thread_case(keep_id, case_id, relevance=0.8, role="ballistics")
    await store.link_thread_events(keep_id, [event_id])
    await store.db.execute("UPDATE threads SET merged_into_id = ? WHERE id = ?", (keep_id, absorb_id))
    await store.db.commit()

    await store.reset_case_scope(case_id)

    case_threads = await store.get_threads_for_case(case_id)
    assert case_threads == []
