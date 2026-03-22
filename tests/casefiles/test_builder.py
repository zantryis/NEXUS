"""Tests for casefile builder gating and fixture integration."""

import json

import pytest
import yaml

from nexus.casefiles.builder import apply_readiness_gate, build_casefile, extract_document, _coerce_hypotheses
from nexus.casefiles.models import (
    CaseMetadata,
    CaseOverview,
    CaseReview,
    CasefileBundle,
    EvidenceItem,
    ExtractedClaim,
    ExtractedDocument,
    FetchedDocument,
)
from nexus.casefiles.storage import load_case_definition


class StubLLM:
    async def complete(self, config_key, _system_prompt, user_prompt, **_kwargs):
        payload = json.loads(user_prompt)
        if config_key == "knowledge_summary":
            doc = payload["document"]
            title = doc["title"]
            return json.dumps(
                {
                    "summary": f"Summary for {title}",
                    "published_at": "2026-03-21",
                    "quality_label": "high" if doc["role"] == "primary" else "medium",
                    "time_anchors": ["2014-03-08"],
                    "entities": [
                        {"name": "MH370", "type": "flight", "description": "Malaysia Airlines Flight 370"}
                    ],
                    "claims": [
                        {
                            "claim": f"{title} supports the southern Indian Ocean theory.",
                            "stance": "supports",
                            "related_hypotheses": [payload["case"]["hypothesis_seeds"][0]],
                            "importance": "high",
                            "excerpt": "supports the southern Indian Ocean theory",
                            "why_it_matters": f"{title} matters to the core route hypothesis.",
                        }
                    ],
                }
            )
        if config_key == "synthesis":
            return json.dumps(
                {
                    "overview": {
                        "best_current_account": "The case points most strongly toward a deliberate diversion and southern Indian Ocean crash.",
                        "confidence_label": "Moderate",
                        "reading_levels": {
                            "short": "Deliberate diversion remains the leading explanation.",
                            "standard": "The evidence most strongly supports a deliberate diversion into the southern Indian Ocean, while exact actor and final crash point remain disputed.",
                            "deep": "A deeper draft summary of the case.",
                        },
                        "key_judgments": ["The route evidence is stronger than the motive evidence."],
                        "recent_updates": ["No new updates in the local fixture corpus."],
                    },
                    "hypotheses": [
                        {
                            "id": "H01",
                            "title": payload["case"]["hypothesis_seeds"][0],
                            "summary": "Leading hypothesis from the local corpus.",
                            "confidence_label": "Leading",
                            "evidence_for": ["E001"],
                            "evidence_against": [],
                            "unresolved_gaps": ["Actor identity remains unresolved."],
                            "what_would_change_my_mind": ["Recovery of decisive primary evidence."],
                        },
                        {
                            "id": "H02",
                            "title": payload["case"]["hypothesis_seeds"][1],
                            "summary": "Secondary hypothesis.",
                            "confidence_label": "Weak",
                            "evidence_for": ["E002"],
                            "evidence_against": ["E001"],
                            "unresolved_gaps": ["Needs stronger technical support."],
                            "what_would_change_my_mind": ["A direct systems-failure record."],
                        },
                        {
                            "id": "H03",
                            "title": payload["case"]["hypothesis_seeds"][2],
                            "summary": "Alternative hypothesis.",
                            "confidence_label": "Weak",
                            "evidence_for": ["E003"],
                            "evidence_against": [],
                            "unresolved_gaps": ["No direct evidence in local corpus."],
                            "what_would_change_my_mind": ["A corroborated third-party interference trace."],
                        },
                    ],
                    "timeline": [
                        {"id": "T01", "label": "Disappearance", "date": "2014-03-08", "description": "The flight disappears.", "evidence_ids": ["E001"]},
                        {"id": "T02", "label": "Search", "date": "2014-03-09", "description": "Search begins.", "evidence_ids": ["E002"]},
                        {"id": "T03", "label": "Debris", "date": "2015-07-29", "description": "Debris is found.", "evidence_ids": ["E003"]},
                        {"id": "T04", "label": "Review", "date": "2016-11", "description": "Review narrows the area.", "evidence_ids": ["E001"]},
                    ],
                    "entities": [{"id": "mh370", "name": "MH370", "type": "flight", "description": "Flight"}],
                    "relations": [],
                    "open_questions": ["Where is the main fuselage?"],
                }
            )
        return json.dumps(
            {
                "scores": {
                    "coherence": 4,
                    "evidentiary_grounding": 4,
                    "source_diversity": 4,
                    "separation_of_hypotheses": 4,
                    "uncertainty_honesty": 4,
                    "non_obvious_synthesis": 4,
                },
                "flags": [],
                "notes": ["Looks grounded enough for a fixture build."],
            }
        )


def test_apply_readiness_gate_blocks_shallow_bundle():
    bundle = CasefileBundle(
        metadata=CaseMetadata(
            slug="sample",
            title="Sample",
            question="What happened?",
            generated_at="2026-03-21T00:00:00+00:00",
            last_updated="2026-03-21T00:00:00+00:00",
        ),
        overview=CaseOverview(
            best_current_account="Thin draft",
            confidence_label="Low",
            reading_levels={"short": "Thin", "standard": "Too short", "deep": "Thin"},
        ),
        evidence=[
            EvidenceItem(
                id="E001",
                claim="Only one claim",
                stance="context",
                quality_label="low",
                summary="Sparse evidence",
                document_id="doc-1",
                document_title="Doc",
                document_url="https://example.com",
                source_label="Doc",
                source_class="media",
            )
        ],
        documents=[
            ExtractedDocument(
                id="doc-1",
                title="Doc",
                url="https://example.com",
                canonical_url="https://example.com",
                kind="article",
                role="secondary",
                source_class="media",
                source_label="Doc",
                summary="Doc summary",
                claims=[ExtractedClaim(claim="Only claim")],
            )
        ],
        review=CaseReview(flags=["shallow_connections"]),
    )
    gated = apply_readiness_gate(bundle)
    assert gated.review.presentable is False
    assert gated.review.blocker_reasons


@pytest.mark.asyncio
async def test_build_casefile_with_local_fixture_corpus(tmp_path, monkeypatch):
    case_root = tmp_path / "casefiles" / "fixture"
    case_root.mkdir(parents=True)

    doc_official = tmp_path / "official.html"
    doc_analysis = tmp_path / "analysis.html"
    doc_media = tmp_path / "media.html"
    for path, text in [
        (doc_official, "<html><body><h1>Official report</h1><p>MH370 disappeared on 2014-03-08.</p><p>Satellite data pointed south.</p></body></html>"),
        (doc_analysis, "<html><body><h1>Independent analysis</h1><p>Debris drift supports the southern Indian Ocean theory.</p></body></html>"),
        (doc_media, "<html><body><h1>Media report</h1><p>Searches continued years later.</p></body></html>"),
    ]:
        path.write_text(text)

    (case_root / "case.yaml").write_text(
        yaml.dump(
            {
                "slug": "fixture",
                "title": "Fixture Case",
                "question": "What happened?",
                "time_bounds": {"start": "2014-03-08", "end": "2026-03-21"},
                "hypothesis_seeds": ["Hypothesis A", "Hypothesis B", "Hypothesis C"],
                "reading_levels": ["short", "standard", "deep"],
            },
            sort_keys=False,
        )
    )
    (case_root / "seed.yaml").write_text(
        yaml.dump(
            {
                "sources": [
                    {
                        "id": "official-doc",
                        "label": "Official doc",
                        "url": str(doc_official),
                        "kind": "report",
                        "role": "primary",
                        "source_class": "official",
                        "priority": 10,
                    },
                    {
                        "id": "analysis-doc",
                        "label": "Analysis doc",
                        "url": str(doc_analysis),
                        "kind": "article",
                        "role": "secondary",
                        "source_class": "analysis",
                        "priority": 8,
                    },
                    {
                        "id": "media-doc",
                        "label": "Media doc",
                        "url": str(doc_media),
                        "kind": "article",
                        "role": "secondary",
                        "source_class": "media",
                        "priority": 7,
                    },
                ]
            },
            sort_keys=False,
        )
    )

    async def fake_search_candidates(_case, *, search_fn=None):
        return [], []

    monkeypatch.setattr("nexus.casefiles.acquisition.search_candidates", fake_search_candidates)

    bundle = await build_casefile(case_root, llm=StubLLM())
    assert bundle.metadata.document_count == 3
    assert len(bundle.evidence) >= 3
    assert len(bundle.hypotheses) >= 3
    assert (case_root / "casefile.json").exists()


@pytest.mark.asyncio
async def test_extract_document_tolerates_string_entities_and_claims(tmp_path):
    case_root = tmp_path / "casefiles" / "fixture"
    case_root.mkdir(parents=True)
    (case_root / "case.yaml").write_text(
        yaml.dump(
            {
                "slug": "fixture",
                "title": "Fixture Case",
                "question": "What happened?",
                "time_bounds": {"start": "2014-03-08", "end": "2026-03-21"},
                "hypothesis_seeds": ["Hypothesis A", "Hypothesis B", "Hypothesis C"],
                "reading_levels": ["short", "standard", "deep"],
            },
            sort_keys=False,
        )
    )
    definition = load_case_definition(case_root)

    class StringyLLM:
        async def complete(self, *_args, **_kwargs):
            return json.dumps(
                {
                    "summary": "A compact summary.",
                    "quality_label": "medium",
                    "time_anchors": ["2020-01"],
                    "entities": ["Florence Debarre", "Huanan Market"],
                    "claims": ["This document is mainly contextual."],
                }
            )

    document = FetchedDocument(
        id="doc-1",
        label="Doc",
        url="https://example.com",
        canonical_url="https://example.com",
        kind="article",
        role="secondary",
        source_class="analysis",
        title="Doc",
        raw_text="Useful context.",
    )

    extracted = await extract_document(StringyLLM(), definition.case, document)
    assert [entity.name for entity in extracted.entities] == ["Florence Debarre", "Huanan Market"]
    assert extracted.claims[0].claim == "This document is mainly contextual."


def test_coerce_hypotheses_tolerates_scalar_list_fields():
    hypotheses = _coerce_hypotheses(
        [
            {
                "id": "H01",
                "title": "Oswald alone",
                "summary": "Test hypothesis",
                "confidence_label": "Contested",
                "evidence_for": "E001",
                "evidence_against": ["E002"],
                "unresolved_gaps": "Ballistics still disputed",
                "what_would_change_my_mind": "A direct contradiction in the primary record",
            }
        ],
        {"E001", "E002"},
        ["Seed A", "Seed B", "Seed C"],
    )
    assert hypotheses[0].evidence_for == ["E001"]
    assert hypotheses[0].unresolved_gaps == ["Ballistics still disputed"]
    assert hypotheses[0].what_would_change_my_mind == ["A direct contradiction in the primary record"]
