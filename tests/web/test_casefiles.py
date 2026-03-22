"""Tests for the casefiles web routes."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml
from httpx import ASGITransport, AsyncClient

from nexus.casefiles.models import (
    CaseAssessment,
    CaseHypothesis,
    CaseMetadata,
    CaseOverview,
    CaseReview,
    CasefileBundle,
    EvidenceItem,
    ExtractedDocument,
)
from nexus.engine.knowledge.events import Event
from nexus.engine.knowledge.store import KnowledgeStore
from nexus.web.app import create_app


def _sample_bundle() -> CasefileBundle:
    return CasefileBundle(
        metadata=CaseMetadata(
            slug="mh370",
            title="MH370",
            question="What happened to MH370?",
            generated_at="2026-03-21T00:00:00+00:00",
            last_updated="2026-03-21T00:00:00+00:00",
            document_count=3,
            evidence_count=2,
            source_count=3,
            presentable=True,
        ),
        overview=CaseOverview(
            best_current_account="Draft best account",
            confidence_label="Moderate",
            reading_levels={
                "short": "Short",
                "standard": "Standard account with enough detail to render.",
                "deep": "Deep account",
            },
        ),
        evidence=[
            EvidenceItem(
                id="E001",
                claim="Evidence one",
                stance="supports",
                quality_label="high",
                summary="Summary one",
                document_id="doc-1",
                document_title="Doc One",
                document_url="https://example.com/1",
                source_label="Doc One",
                source_class="official",
            ),
            EvidenceItem(
                id="E002",
                claim="Evidence two",
                stance="refutes",
                quality_label="medium",
                summary="Summary two",
                document_id="doc-2",
                document_title="Doc Two",
                document_url="https://example.com/2",
                source_label="Doc Two",
                source_class="analysis",
            ),
        ],
        documents=[
            ExtractedDocument(
                id="doc-1",
                title="Doc One",
                url="https://example.com/1",
                canonical_url="https://example.com/1",
                kind="report",
                role="primary",
                source_class="official",
                source_label="Doc One",
                summary="Doc summary one",
            )
        ],
        review=CaseReview(presentable=True, verdict="presentable"),
    )


async def _seed_db_case_state(app) -> int:
    store = app.state.store
    case_id = await store.upsert_case(
        "mh370",
        "MH370",
        "What happened to MH370?",
        time_bounds={"start": "2014-03-08", "end": "2026-03-21"},
        build_defaults={"max_documents": 12},
        monitoring_enabled=True,
    )
    bundle = _sample_bundle()
    await store.replace_case_documents(case_id, [item.model_dump(mode="json") for item in bundle.documents])
    await store.replace_case_evidence(case_id, [item.model_dump(mode="json") for item in bundle.evidence])
    await store.replace_case_hypotheses(
        case_id,
        [
            CaseHypothesis(
                id="H01",
                title="Deliberate diversion",
                summary="Primary working hypothesis.",
                confidence_label="Leading",
                evidence_for=["E001"],
                evidence_against=["E002"],
                unresolved_gaps=["Who acted and why?"],
                what_would_change_my_mind=["A verified systems-failure chain."],
            ).model_dump(mode="json")
        ],
    )
    await store.replace_case_assessments(
        case_id,
        [
            CaseAssessment(
                id="A-h01",
                target_hypothesis_id="H01",
                mode="posterior",
                question="What is the probability of deliberate diversion?",
                probability=0.71,
                confidence="medium",
                rationale="The available evidence clusters around a controlled deviation into the southern Indian Ocean.",
                counterarguments=["No direct cockpit recording survives."],
                evidence_ids=["E001", "E002"],
                evidence_thread_ids=[],
                signposts=["Recovery of additional high-confidence wreckage."],
            ).model_dump(mode="json")
        ],
    )
    await store.replace_case_open_questions(case_id, ["Where is the main wreckage field?"])

    event = Event(
        date="2014-03-08",
        summary="Investigators narrowed the end-state to the southern Indian Ocean.",
        significance=8,
        relation_to_prior="case development",
        entities=["MH370", "ATSB", "Southern Indian Ocean"],
        sources=[{"url": "https://example.com/report", "outlet": "ATSB", "affiliation": "official"}],
    )
    event_id = (await store.add_events([event], "__case__:mh370", case_id=case_id))[0]
    mh370_id = await store.upsert_entity("MH370", "concept")
    atsb_id = await store.upsert_entity("ATSB", "org")
    sio_id = await store.upsert_entity("Southern Indian Ocean", "concept")
    await store.link_event_entities(event_id, [mh370_id, atsb_id, sio_id])

    thread_id = await store.upsert_thread(
        "southern-indian-ocean-search",
        "The Search for MH370 in the Southern Indian Ocean",
        9,
        "active",
    )
    await store.link_thread_case(thread_id, case_id, relevance=0.9, role="search")
    await store.link_thread_events(thread_id, [event_id])
    await store.add_convergence(
        thread_id,
        "Multiple evidence lines point toward a southern end-state.",
        ["ATSB", "independent analysis"],
    )
    await store.add_divergence(
        thread_id,
        "Search-area confidence",
        "ATSB",
        "The official search area remains the most likely zone.",
        "Independent analysis",
        "The search was materially mislocated within the broader southern corridor.",
    )
    return thread_id


@pytest.fixture
async def casefile_app(tmp_path):
    data_dir = tmp_path / "data"
    case_dir = data_dir / "casefiles" / "mh370"
    case_dir.mkdir(parents=True)
    (data_dir / "config.yaml").write_text(
        yaml.dump(
            {
                "preset": "balanced",
                "user": {"name": "Tester", "timezone": "UTC", "output_language": "en"},
                "topics": [{"name": "AI", "priority": "high"}],
            },
            sort_keys=False,
        )
    )
    (case_dir / "case.yaml").write_text(
        yaml.dump(
            {
                "slug": "mh370",
                "title": "MH370",
                "question": "What happened to MH370?",
                "time_bounds": {"start": "2014-03-08", "end": "2026-03-21"},
                "hypothesis_seeds": ["Hypothesis A", "Hypothesis B", "Hypothesis C"],
                "reading_levels": ["short", "standard", "deep"],
            },
            sort_keys=False,
        )
    )
    (case_dir / "casefile.json").write_text(_sample_bundle().model_dump_json(indent=2))

    app = create_app(data_dir / "knowledge.db", data_dir=data_dir)
    store = KnowledgeStore(data_dir / "knowledge.db")
    await store.initialize()
    app.state.store = store
    app.state.llm = MagicMock()

    yield app
    await store.close()


async def test_casefiles_index_renders(casefile_app):
    transport = ASGITransport(app=casefile_app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/casefiles/")
    assert resp.status_code == 200
    assert "MH370" in resp.text
    assert "Casefiles" in resp.text


async def test_casefile_detail_renders_bundle(casefile_app):
    transport = ASGITransport(app=casefile_app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/casefiles/mh370")
    assert resp.status_code == 200
    assert "Best Current Account" in resp.text
    assert "Draft best account" in resp.text


async def test_casefile_graph_page_renders(casefile_app):
    transport = ASGITransport(app=casefile_app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/casefiles/mh370/graph")
    assert resp.status_code == 200
    assert "Graph" in resp.text
    assert "/api/casefiles/mh370/graph" in resp.text


async def test_casefile_status_endpoint_reports_last_build(casefile_app):
    transport = ASGITransport(app=casefile_app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/casefiles/mh370/status")
    assert resp.status_code == 200
    assert "Last built" in resp.text


async def test_casefile_rebuild_endpoint_starts_background_job(casefile_app):
    async def fake_build(*_args, **_kwargs):
        return _sample_bundle()

    transport = ASGITransport(app=casefile_app, raise_app_exceptions=False)
    with patch("nexus.web.routes.casefiles.build_casefile", new=AsyncMock(side_effect=fake_build)):
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/casefiles/mh370/rebuild")
            assert resp.status_code == 200
            assert "Starting casefile rebuild" in resp.text
        tasks = tuple(getattr(casefile_app.state, "background_tasks", ()))
        if tasks:
            await asyncio.gather(*tasks)
    status = casefile_app.state.casefile_statuses["mh370"]
    assert status.status == "completed"


async def test_casefile_chat_endpoint_returns_grounded_answer(casefile_app):
    transport = ASGITransport(app=casefile_app, raise_app_exceptions=False)
    with patch(
        "nexus.web.routes.casefiles.answer_case_question",
        new=AsyncMock(return_value={"answer": "Grounded answer", "citations": ["E001"]}),
    ):
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/casefiles/mh370/chat", json={"question": "What happened?"})
    assert resp.status_code == 200
    assert resp.json()["answer"] == "Grounded answer"
    assert resp.json()["citations"] == ["E001"]


async def test_casefile_chat_requires_question(casefile_app):
    transport = ASGITransport(app=casefile_app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/api/casefiles/mh370/chat", json={"question": ""})
    assert resp.status_code == 400


async def test_casefile_graph_endpoint_returns_case_scoped_data(casefile_app):
    await _seed_db_case_state(casefile_app)
    transport = ASGITransport(app=casefile_app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/casefiles/mh370/graph")
    assert resp.status_code == 200
    payload = resp.json()
    assert {node["name"] for node in payload["nodes"]} == {"MH370", "ATSB", "Southern Indian Ocean"}
    assert payload["links"]


async def test_casefile_assessments_endpoint_returns_items(casefile_app):
    await _seed_db_case_state(casefile_app)
    transport = ASGITransport(app=casefile_app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/casefiles/mh370/assessments")
    assert resp.status_code == 200
    items = resp.json()["items"]
    assert len(items) == 1
    assert items[0]["id"] == "A-h01"
    assert items[0]["target_hypothesis_id"] == "H01"


async def test_casefile_updates_endpoint_uses_db_backed_overlay(casefile_app):
    await _seed_db_case_state(casefile_app)
    transport = ASGITransport(app=casefile_app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/casefiles/mh370/updates")
    assert resp.status_code == 200
    items = resp.json()["items"]
    assert any("active case threads" in item for item in items)
    assert any("Current leading assessed account" in item for item in items)


async def test_casefile_thread_detail_renders_only_for_linked_case_thread(casefile_app):
    thread_id = await _seed_db_case_state(casefile_app)
    transport = ASGITransport(app=casefile_app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/casefiles/mh370/threads/southern-indian-ocean-search")
    assert resp.status_code == 200
    assert "The Search for MH370 in the Southern Indian Ocean" in resp.text
    assert str(thread_id) in resp.text or "Search-area confidence" in resp.text
