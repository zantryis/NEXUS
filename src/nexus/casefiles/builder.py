"""Casefile build pipeline: acquire, extract, synthesize, review, persist."""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from nexus.agent.websearch import web_search
from nexus.casefiles.acquisition import acquire_case_documents
from nexus.casefiles.models import (
    AcquisitionResult,
    CaseAssessment,
    CaseConfig,
    CaseDivergence,
    CaseEntity,
    CaseGraphSummary,
    CaseHypothesis,
    CaseMetadata,
    CaseOverview,
    CaseRelation,
    CaseReview,
    CaseThread,
    CasefileBundle,
    EvidenceItem,
    ExtractedClaim,
    ExtractedDocument,
    ExtractedEntity,
    FetchedDocument,
    ReviewScores,
    TimelineEntry,
)
from nexus.casefiles.runtime import (
    build_case_divergence_models,
    build_case_graph_summary,
    build_case_thread_models,
    build_recent_changes,
    generate_case_assessments,
    persist_case_operational_state,
)
from nexus.casefiles.storage import load_case_definition, save_casefile

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[str], None] | None

EXTRACTION_SYSTEM_PROMPT = (
    "You are extracting evidence for a controversial long-horizon incident casefile. "
    "Work only from the supplied document. Be literal, sober, and explicit about uncertainty. "
    "Return valid JSON only."
)

SYNTHESIS_SYSTEM_PROMPT = (
    "You are building a living casefile for a controversial incident. "
    "Your job is not to sound authoritative; it is to produce a coherent, evidence-backed account "
    "that separates what is established from what is disputed. "
    "Use only the provided evidence and document summaries. "
    "Prefer concrete causal or mechanistic hypotheses over meta-hypotheses about uncertainty, missing records, or epistemic deadlock. "
    "If the record is incomplete, express that through confidence labels, unresolved gaps, and open questions rather than letting 'uncertainty' become the leading explanation unless the evidence truly cannot distinguish any causal account. "
    "Return valid JSON only."
)

REVIEW_SYSTEM_PROMPT = (
    "You are a strict internal reviewer for a casefile intelligence product. "
    "Score the bundle harshly on evidentiary grounding, source diversity, hypothesis separation, "
    "uncertainty honesty, and non-obvious synthesis. "
    "Flag shallow official recaps aggressively. Return valid JSON only."
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _notify(progress: ProgressCallback, message: str) -> None:
    if progress:
        progress(message)


def _strip_json_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def _repair_json_text(text: str) -> str:
    repaired = text.replace("\x00", "")
    repaired = re.sub(r'\\u(?![0-9a-fA-F]{4})', r'\\\\u', repaired)
    repaired = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', repaired)
    return repaired


def _parse_json_payload(text: str) -> dict:
    cleaned = _strip_json_fences(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        repaired = _repair_json_text(cleaned)
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            pass
        for candidate in (cleaned, repaired):
            start = candidate.find("{")
            end = candidate.rfind("}")
            if start >= 0 and end > start:
                snippet = candidate[start : end + 1]
                try:
                    return json.loads(snippet)
                except json.JSONDecodeError:
                    continue
        raise


def _coerce_mapping(payload) -> dict:
    if isinstance(payload, dict):
        return payload
    raise ValueError("Expected a JSON object payload.")


def _fallback_document_summary(text: str) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", " ".join(text.split()))
    return " ".join(sentences[:3])[:500].strip() or "No useful summary extracted."


def _fallback_claims(text: str, hypotheses: list[str]) -> list[ExtractedClaim]:
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", " ".join(text.split())) if s.strip()]
    claims: list[ExtractedClaim] = []
    for sentence in sentences[:3]:
        claims.append(
            ExtractedClaim(
                claim=sentence[:260],
                stance="context",
                related_hypotheses=hypotheses[:1],
                importance="medium",
                excerpt=sentence[:120],
                why_it_matters="Fallback extraction from document text.",
            )
        )
    return claims


def _normalize_entity_payload(entity) -> dict | None:
    if isinstance(entity, dict):
        return entity
    if isinstance(entity, str):
        text = entity.strip()
        if text:
            return {"name": text, "type": "other"}
    return None


def _normalize_claim_payload(claim, hypotheses: list[str]) -> dict | None:
    if isinstance(claim, dict):
        return claim
    if isinstance(claim, str):
        text = claim.strip()
        if text:
            return {
                "claim": text[:260],
                "stance": "context",
                "related_hypotheses": hypotheses[:1],
                "importance": "medium",
                "excerpt": text[:120],
                "why_it_matters": "Model returned an unstructured claim string.",
            }
    return None


async def extract_document(
    llm,
    case: CaseConfig,
    document: FetchedDocument,
) -> ExtractedDocument:
    """Run case-specific document extraction, with heuristic fallback."""
    prompt = {
        "case": {
            "title": case.title,
            "question": case.question,
            "hypothesis_seeds": case.hypothesis_seeds,
        },
        "document": {
            "title": document.title,
            "url": document.canonical_url,
            "kind": document.kind,
            "role": document.role,
            "source_class": document.source_class,
            "notes": document.notes,
            "published_at": document.published_at,
            "excerpt": document.excerpt,
            "text": document.raw_text,
        },
        "output_schema": {
            "summary": "2-4 sentence sober summary",
            "published_at": "ISO date or null",
            "quality_label": "high|medium|low",
            "time_anchors": ["dates or time periods"],
            "entities": [
                {"name": "entity name", "type": "person|org|place|flight|system|document|other", "description": "optional"}
            ],
            "claims": [
                {
                    "claim": "verifiable claim or evidentiary takeaway",
                    "stance": "supports|refutes|context",
                    "related_hypotheses": ["hypothesis seed labels"],
                    "importance": "high|medium|low",
                    "excerpt": "very short quote or paraphrase under 20 words",
                    "why_it_matters": "why this matters to the case",
                }
            ],
        },
        "constraints": [
            "Use only facts present in the document text.",
            "Keep claim count to at most 4.",
            "If the document is mainly descriptive or contextual, use stance=context.",
            "Prefer exact dates where the document gives them.",
        ],
    }

    try:
        raw = await llm.complete(
            "knowledge_summary",
            EXTRACTION_SYSTEM_PROMPT,
            json.dumps(prompt, ensure_ascii=True),
            json_response=True,
            timeout_s=120.0,
        )
        data = _coerce_mapping(_parse_json_payload(raw))
    except Exception as exc:
        logger.warning("Casefile document extraction failed for %s: %s", document.canonical_url, exc)
        data = {
            "summary": _fallback_document_summary(document.raw_text),
            "published_at": document.published_at,
            "quality_label": "high" if document.role == "primary" else "medium",
            "time_anchors": [],
            "entities": [],
            "claims": [
                claim.model_dump()
                for claim in _fallback_claims(document.raw_text, case.hypothesis_seeds)
            ],
        }

    claims = []
    for raw_claim in (data.get("claims") or [])[:4]:
        normalized = _normalize_claim_payload(raw_claim, case.hypothesis_seeds)
        if normalized is None:
            continue
        try:
            claims.append(ExtractedClaim.model_validate(normalized))
        except Exception:
            continue

    entities = []
    for raw_entity in (data.get("entities") or [])[:10]:
        normalized = _normalize_entity_payload(raw_entity)
        if normalized is None:
            continue
        try:
            entities.append(ExtractedEntity.model_validate(normalized))
        except Exception:
            continue

    if not claims:
        claims = _fallback_claims(document.raw_text, case.hypothesis_seeds)

    return ExtractedDocument(
        id=document.id,
        title=document.title,
        url=document.url,
        canonical_url=document.canonical_url,
        kind=document.kind,
        role=document.role,
        source_class=document.source_class,
        source_label=document.label,
        priority=document.priority,
        notes=document.notes,
        discovered_via=document.discovered_via,
        published_at=data.get("published_at") or document.published_at,
        quality_label=data.get("quality_label") or ("high" if document.role == "primary" else "medium"),
        summary=(data.get("summary") or _fallback_document_summary(document.raw_text))[:800],
        time_anchors=[str(anchor) for anchor in (data.get("time_anchors") or [])[:8]],
        entities=entities,
        claims=claims,
        excerpt=document.excerpt,
        ingestion_status=document.ingestion_status,
        ingestion_error=document.ingestion_error,
    )


def build_evidence(documents: list[ExtractedDocument], *, max_evidence: int) -> list[EvidenceItem]:
    """Convert extracted document claims into evidence ledger items."""
    evidence: list[EvidenceItem] = []
    for document in documents:
        for claim in document.claims:
            evidence_id = f"E{len(evidence) + 1:03d}"
            evidence.append(
                EvidenceItem(
                    id=evidence_id,
                    claim=claim.claim,
                    stance=claim.stance,
                    quality_label=claim.importance,
                    summary=claim.why_it_matters or claim.claim,
                    document_id=document.id,
                    document_title=document.title,
                    document_url=document.canonical_url,
                    source_label=document.source_label,
                    source_class=document.source_class,
                    related_hypotheses=claim.related_hypotheses[:3],
                    excerpt=claim.excerpt or document.excerpt,
                    time_anchors=document.time_anchors[:4],
                )
            )
            if len(evidence) >= max_evidence:
                return evidence
    return evidence


def _normalize_entity_id(name: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return normalized or "entity"


def _source_distribution(documents: list[ExtractedDocument]) -> dict[str, int]:
    counts = Counter(document.source_class for document in documents)
    return {
        "official": counts.get("official", 0),
        "investigation": counts.get("investigation", 0),
        "analysis": counts.get("analysis", 0),
        "media": counts.get("media", 0),
    }


def _sanitize_ids(items: list[str], valid_ids: set[str]) -> list[str]:
    return [item for item in items if item in valid_ids]


def _ensure_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _matching_evidence_ids(
    evidence: list[EvidenceItem],
    *,
    hypothesis_title: str,
    hypothesis_seed: str | None,
    stance: str,
) -> list[str]:
    title = (hypothesis_title or "").lower()
    seed = (hypothesis_seed or "").lower()
    matches: list[str] = []
    for item in evidence:
        if item.stance != stance:
            continue
        related = " | ".join(item.related_hypotheses).lower()
        claim = item.claim.lower()
        haystack = f"{related} | {claim}"
        for term in (title, seed):
            if not term:
                continue
            needle = term[:28]
            if needle and needle in haystack:
                matches.append(item.id)
                break
    return matches


def _coerce_hypotheses(
    data: list[dict],
    valid_evidence_ids: set[str],
    seeds: list[str],
    evidence: list[EvidenceItem],
) -> list[CaseHypothesis]:
    hypotheses: list[CaseHypothesis] = []
    for index, item in enumerate(data, start=1):
        payload = dict(item)
        payload["id"] = payload.get("id") or f"H{index:02d}"
        payload["evidence_for"] = _sanitize_ids(_ensure_list(payload.get("evidence_for")), valid_evidence_ids)
        payload["evidence_against"] = _sanitize_ids(_ensure_list(payload.get("evidence_against")), valid_evidence_ids)
        payload["unresolved_gaps"] = [str(item) for item in _ensure_list(payload.get("unresolved_gaps")) if str(item).strip()]
        payload["what_would_change_my_mind"] = [
            str(item)
            for item in _ensure_list(payload.get("what_would_change_my_mind"))
            if str(item).strip()
        ]
        seed = seeds[index - 1] if index - 1 < len(seeds) else None
        title = str(payload.get("title") or payload.get("summary") or seed or "")
        if not payload["evidence_for"]:
            payload["evidence_for"] = _matching_evidence_ids(
                evidence,
                hypothesis_title=title,
                hypothesis_seed=seed,
                stance="supports",
            )[:4]
        if not payload["evidence_against"]:
            payload["evidence_against"] = _matching_evidence_ids(
                evidence,
                hypothesis_title=title,
                hypothesis_seed=seed,
                stance="refutes",
            )[:3]
        hypotheses.append(CaseHypothesis.model_validate(payload))

    while len(hypotheses) < 3:
        seed = seeds[len(hypotheses)] if len(hypotheses) < len(seeds) else f"Alternative hypothesis {len(hypotheses) + 1}"
        hypotheses.append(
            CaseHypothesis(
                id=f"H{len(hypotheses) + 1:02d}",
                title=seed,
                summary="This hypothesis remains insufficiently developed in the current evidence bundle.",
                confidence_label="Weak",
                unresolved_gaps=["The current bundle does not separate this hypothesis cleanly enough."],
                what_would_change_my_mind=["Higher-quality direct evidence that distinguishes this path from the leading account."],
            )
        )
    return hypotheses


def _coerce_timeline(data: list[dict], valid_evidence_ids: set[str]) -> list[TimelineEntry]:
    timeline: list[TimelineEntry] = []
    for index, item in enumerate(data, start=1):
        payload = dict(item)
        payload["id"] = payload.get("id") or f"T{index:02d}"
        payload["evidence_ids"] = _sanitize_ids(payload.get("evidence_ids", []), valid_evidence_ids)
        timeline.append(TimelineEntry.model_validate(payload))
    return timeline


def _coerce_entities(data: list[dict]) -> list[CaseEntity]:
    entities: list[CaseEntity] = []
    seen: set[str] = set()
    for item in data:
        payload = dict(item)
        payload["id"] = payload.get("id") or _normalize_entity_id(payload.get("name", "entity"))
        if payload["id"] in seen:
            continue
        seen.add(payload["id"])
        entities.append(CaseEntity.model_validate(payload))
    return entities


def _coerce_relations(data: list[dict], valid_evidence_ids: set[str], entities: list[CaseEntity]) -> list[CaseRelation]:
    entity_ids = {entity.id for entity in entities}
    relations: list[CaseRelation] = []
    for item in data:
        payload = dict(item)
        if payload.get("source_entity_id") not in entity_ids or payload.get("target_entity_id") not in entity_ids:
            continue
        payload["evidence_ids"] = _sanitize_ids(payload.get("evidence_ids", []), valid_evidence_ids)
        relations.append(CaseRelation.model_validate(payload))
    return relations


def _fallback_hypotheses(case: CaseConfig, evidence: list[EvidenceItem]) -> list[CaseHypothesis]:
    hypotheses: list[CaseHypothesis] = []
    for index in range(3):
        title = case.hypothesis_seeds[index] if index < len(case.hypothesis_seeds) else f"Alternative hypothesis {index + 1}"
        seed_text = title.lower()
        matching = [
            item for item in evidence
            if any(seed_text[:24] in hypothesis.lower() for hypothesis in item.related_hypotheses)
            or seed_text[:24] in item.claim.lower()
        ]
        support_ids = [item.id for item in matching[:3]]
        counter_ids = [item.id for item in matching[3:5] if item.stance == "refutes"]
        hypotheses.append(
            CaseHypothesis(
                id=f"H{index + 1:02d}",
                title=title,
                summary="This draft hypothesis summary was built from the extracted evidence ledger and still needs a cleaner synthesis pass.",
                confidence_label="Contested" if index == 0 else "Weak",
                evidence_for=support_ids,
                evidence_against=counter_ids,
                unresolved_gaps=["The current build needs a stronger synthesis pass to separate this hypothesis cleanly."],
                what_would_change_my_mind=["Higher-quality direct evidence that either narrows or rules out this path."],
            )
        )
    return hypotheses


def _fallback_timeline(documents: list[ExtractedDocument], evidence: list[EvidenceItem]) -> list[TimelineEntry]:
    timeline: list[TimelineEntry] = []
    evidence_by_doc: dict[str, list[str]] = {}
    for item in evidence:
        evidence_by_doc.setdefault(item.document_id, []).append(item.id)

    for index, document in enumerate(documents[:6], start=1):
        date_hint = document.time_anchors[0] if document.time_anchors else (document.published_at or "Undated")
        timeline.append(
            TimelineEntry(
                id=f"T{index:02d}",
                label=document.title[:120],
                date=date_hint,
                description=document.summary,
                evidence_ids=evidence_by_doc.get(document.id, [])[:3],
            )
        )
    return timeline


def _fallback_entities(documents: list[ExtractedDocument]) -> list[CaseEntity]:
    entities: list[CaseEntity] = []
    seen: set[str] = set()
    for document in documents:
        for item in document.entities:
            entity_id = _normalize_entity_id(item.name)
            if entity_id in seen:
                continue
            seen.add(entity_id)
            entities.append(
                CaseEntity(
                    id=entity_id,
                    name=item.name,
                    type=item.type,
                    description=item.description,
                )
            )
            if len(entities) >= 12:
                return entities
    return entities


def _fallback_bundle(
    case: CaseConfig,
    acquisition: AcquisitionResult,
    documents: list[ExtractedDocument],
    evidence: list[EvidenceItem],
) -> CasefileBundle:
    best_account = (
        "The strongest current account in this draft is that MH370 was deliberately diverted and ended in the southern Indian Ocean, "
        "but the exact actor, mechanism, and final crash location remain disputed."
    )
    generated_at = _now_iso()
    return CasefileBundle(
        metadata=CaseMetadata(
            slug=case.slug,
            title=case.title,
            question=case.question,
            generated_at=generated_at,
            last_updated=generated_at,
            time_bounds=case.time_bounds,
            reading_levels=case.reading_levels,
            source_count=len(acquisition.candidates),
            document_count=len(documents),
            evidence_count=len(evidence),
            build_defaults=case.build,
        ),
        overview=CaseOverview(
            best_current_account=best_account,
            confidence_label="Contested",
            reading_levels={
                "short": best_account,
                "standard": best_account + " This fallback bundle still needs a full synthesis pass before presentation.",
                "deep": "\n\n".join(document.summary for document in documents[:3]) or best_account,
            },
            key_judgments=[
                "Official and independent evidence still point most strongly toward a southern Indian Ocean end state.",
                "The evidentiary base remains much stronger on route geometry than on motive or actor.",
                "Search-area uncertainty is still a live issue, especially in light of renewed search efforts.",
            ],
            recent_updates=[document.summary for document in documents[:2]],
        ),
        hypotheses=_fallback_hypotheses(case, evidence),
        timeline=_fallback_timeline(documents, evidence),
        evidence=evidence,
        documents=documents,
        entities=_fallback_entities(documents),
        relations=[],
        open_questions=[
            "Who, if anyone, executed the diversion inside the aircraft?",
            "How much evidentiary weight should be given to alternate tracking methods outside the satcom baseline?",
            "Is the remaining uncertainty mostly about cause, crash dynamics, or search-area placement?",
        ],
        review=CaseReview(),
    )


async def synthesize_casefile(
    llm,
    case: CaseConfig,
    acquisition: AcquisitionResult,
    documents: list[ExtractedDocument],
    evidence: list[EvidenceItem],
    *,
    thread_context: dict | None = None,
) -> CasefileBundle:
    """Synthesize the casefile bundle from extracted documents and evidence."""
    evidence_dump = [item.model_dump() for item in evidence]
    document_dump = [
        {
            "id": document.id,
            "title": document.title,
            "url": document.canonical_url,
            "role": document.role,
            "source_class": document.source_class,
            "published_at": document.published_at,
            "quality_label": document.quality_label,
            "summary": document.summary,
            "time_anchors": document.time_anchors,
            "claims": [claim.model_dump() for claim in document.claims],
        }
        for document in documents
    ]

    prompt = {
        "case": {
            "slug": case.slug,
            "title": case.title,
            "question": case.question,
            "time_bounds": case.time_bounds.model_dump(),
            "hypothesis_seeds": case.hypothesis_seeds,
            "reading_levels": case.reading_levels,
        },
        "source_distribution": _source_distribution(documents),
        "queries": [query.model_dump() for query in acquisition.queries],
        "documents": document_dump,
        "evidence": evidence_dump,
        "thread_context": thread_context or {},
        "output_schema": {
            "overview": {
                "best_current_account": "plain-English answer to what most likely happened",
                "confidence_label": "qualitative confidence",
                "reading_levels": {
                    "short": "tight TLDR",
                    "standard": "2-4 paragraph account",
                    "deep": "deeper briefing",
                },
                "key_judgments": ["short evidence-backed judgments"],
                "recent_updates": ["new developments or current search status"],
            },
            "hypotheses": [
                {
                    "id": "H01",
                    "title": "hypothesis title",
                    "summary": "coherent summary",
                    "confidence_label": "Leading|Plausible|Weak|Poorly supported",
                    "evidence_for": ["E001"],
                    "evidence_against": ["E010"],
                    "unresolved_gaps": ["what remains unresolved"],
                    "what_would_change_my_mind": ["specific falsifier or confirmer"],
                }
            ],
            "timeline": [
                {
                    "id": "T01",
                    "label": "event label",
                    "date": "date or time window",
                    "description": "what happened",
                    "evidence_ids": ["E001"],
                }
            ],
            "entities": [
                {"id": "malaysia-airlines-flight-370", "name": "Malaysia Airlines Flight 370", "type": "flight", "description": "optional"}
            ],
            "relations": [
                {
                    "source_entity_id": "entity-a",
                    "target_entity_id": "entity-b",
                    "relationship": "describe the relationship",
                    "evidence_ids": ["E001"],
                }
            ],
            "open_questions": ["the most important unresolved questions"],
        },
        "constraints": [
            "Order hypotheses from strongest to weakest support.",
            "Do not invent evidence IDs or documents.",
            "Make uncertainty explicit.",
            "Every hypothesis should cite both supporting evidence and, where the ledger contains it, explicit counterevidence or limiting evidence.",
            "Do not leave evidence_against empty when the bundle contains meaningful evidence that weakens the hypothesis.",
            "The best current account can be nuanced, but it must answer the question directly.",
            "At least 3 hypotheses are required.",
            "Prefer ranking a causal explanation first if one has materially better support than the others.",
            "Do not let a meta-hypothesis like uncertainty, compromised records, or data deadlock outrank all causal hypotheses unless the evidence genuinely makes causal ranking impossible.",
            "Use unresolved_gaps and open_questions to capture missing records, withheld data, or contaminated evidence rather than turning those into the main theory by default.",
        ],
    }

    try:
        raw = await llm.complete(
            "synthesis",
            SYNTHESIS_SYSTEM_PROMPT,
            json.dumps(prompt, ensure_ascii=True),
            json_response=True,
            timeout_s=180.0,
        )
        data = _coerce_mapping(_parse_json_payload(raw))
    except Exception as exc:
        logger.warning("Casefile synthesis failed, using fallback bundle: %s", exc)
        return _fallback_bundle(case, acquisition, documents, evidence)

    overview_payload = data.get("overview") or {}
    overview_levels = dict(overview_payload.get("reading_levels") or {})
    best_current_account = overview_payload.get("best_current_account") or "The current evidence bundle does not yet support a confident best account."
    overview_levels.setdefault("short", best_current_account)
    overview_levels.setdefault("standard", best_current_account)
    overview_levels.setdefault("deep", overview_levels.get("standard", best_current_account))

    evidence_ids = {item.id for item in evidence}
    hypotheses = _coerce_hypotheses(data.get("hypotheses") or [], evidence_ids, case.hypothesis_seeds, evidence)
    timeline = _coerce_timeline(data.get("timeline") or [], evidence_ids)
    entities = _coerce_entities(data.get("entities") or [])
    relations = _coerce_relations(data.get("relations") or [], evidence_ids, entities)

    generated_at = _now_iso()
    metadata = CaseMetadata(
        slug=case.slug,
        title=case.title,
        question=case.question,
        generated_at=generated_at,
        last_updated=generated_at,
        time_bounds=case.time_bounds,
        reading_levels=case.reading_levels,
        source_count=len(acquisition.candidates),
        document_count=len(documents),
        evidence_count=len(evidence),
        build_defaults=case.build,
    )

    return CasefileBundle(
        metadata=metadata,
        overview=CaseOverview(
            best_current_account=best_current_account,
            confidence_label=overview_payload.get("confidence_label") or "Contested",
            reading_levels=overview_levels,
            key_judgments=[str(item) for item in (overview_payload.get("key_judgments") or [])[:6]],
            recent_updates=[str(item) for item in (overview_payload.get("recent_updates") or [])[:4]],
        ),
        hypotheses=hypotheses,
        timeline=timeline,
        evidence=evidence,
        documents=documents,
        entities=entities,
        relations=relations,
        threads=[],
        divergence=[],
        assessments=[],
        graph=CaseGraphSummary(),
        recent_changes=[],
        open_questions=[str(item) for item in (data.get("open_questions") or [])[:8]],
        review=CaseReview(),
    )


def _default_review(documents: list[ExtractedDocument]) -> CaseReview:
    scores = ReviewScores(
        coherence=2,
        evidentiary_grounding=2,
        source_diversity=2,
        separation_of_hypotheses=2,
        uncertainty_honesty=3,
        non_obvious_synthesis=2,
    )
    flags: list[str] = []
    if not any(document.role == "primary" for document in documents):
        flags.append("weak_primary_docs")
    return CaseReview(
        scores=scores,
        flags=flags,
        notes=["Fallback review used because the critique call failed."],
        verdict="draft",
        presentable=False,
    )


async def review_casefile(
    llm,
    bundle: CasefileBundle,
) -> CaseReview:
    """Run a strict critique pass over the synthesized casefile."""
    prompt = {
        "metadata": bundle.metadata.model_dump(),
        "overview": bundle.overview.model_dump(),
        "hypotheses": [item.model_dump() for item in bundle.hypotheses],
        "timeline_count": len(bundle.timeline),
        "evidence_count": len(bundle.evidence),
        "thread_count": len(bundle.threads),
        "divergence_count": len(bundle.divergence),
        "assessment_count": len(bundle.assessments),
        "source_distribution": _source_distribution(bundle.documents),
        "document_samples": [
            {
                "id": document.id,
                "role": document.role,
                "source_class": document.source_class,
                "quality_label": document.quality_label,
                "summary": document.summary,
            }
            for document in bundle.documents[:8]
        ],
        "output_schema": {
            "scores": {
                "coherence": "1-5",
                "evidentiary_grounding": "1-5",
                "source_diversity": "1-5",
                "separation_of_hypotheses": "1-5",
                "uncertainty_honesty": "1-5",
                "non_obvious_synthesis": "1-5",
            },
            "flags": [
                "official_narrative_only",
                "insufficient_counterevidence",
                "weak_primary_docs",
                "shallow_connections",
            ],
            "notes": ["specific reviewer notes"],
        },
    }

    try:
        raw = await llm.complete(
            "agent",
            REVIEW_SYSTEM_PROMPT,
            json.dumps(prompt, ensure_ascii=True),
            json_response=True,
            timeout_s=120.0,
        )
        data = _coerce_mapping(_parse_json_payload(raw))
    except Exception as exc:
        logger.warning("Casefile review failed: %s", exc)
        return _default_review(bundle.documents)

    return CaseReview(
        scores=ReviewScores.model_validate(data.get("scores") or {}),
        flags=[str(flag) for flag in (data.get("flags") or [])[:8]],
        notes=[str(note) for note in _ensure_list(data.get("notes"))[:8]],
        verdict="draft",
        presentable=False,
    )


def apply_readiness_gate(bundle: CasefileBundle) -> CasefileBundle:
    """Apply deterministic readiness rules on top of the review output."""
    review = bundle.review
    blockers = list(review.blocker_reasons)

    standard_tldr = bundle.overview.reading_levels.get("standard", "").strip()
    if len(standard_tldr) < 180:
        blockers.append("Standard TLDR is too thin.")
    if len(bundle.timeline) < 4:
        blockers.append("Timeline is not yet coherent enough.")
    if len(bundle.evidence) < 8:
        blockers.append("Evidence ledger is too sparse.")
    if len(bundle.hypotheses) < 3:
        blockers.append("Fewer than three ranked hypotheses were produced.")
    if not any(document.role == "primary" for document in bundle.documents):
        blockers.append("The bundle lacks primary or near-primary source material.")

    flag_blockers = {
        "official_narrative_only": "Critic flagged the casefile as an official-narrative recap.",
        "insufficient_counterevidence": "Critic found too little counterevidence across hypotheses.",
        "weak_primary_docs": "Critic found the primary-document base too weak.",
        "shallow_connections": "Critic found the synthesis too shallow.",
    }
    for flag in review.flags:
        if flag in flag_blockers:
            blockers.append(flag_blockers[flag])

    deduped_blockers: list[str] = []
    for blocker in blockers:
        if blocker not in deduped_blockers:
            deduped_blockers.append(blocker)

    review.blocker_reasons = deduped_blockers
    review.presentable = not deduped_blockers
    review.verdict = "presentable" if review.presentable else "draft"
    bundle.metadata.presentable = review.presentable
    bundle.review = review
    return bundle


async def build_casefile(
    case_path: Path,
    *,
    llm,
    store=None,
    search_fn=None,
    progress: ProgressCallback = None,
) -> CasefileBundle:
    """Build and persist one casefile bundle from its on-disk contract."""
    if llm is None:
        raise RuntimeError("Casefile build requires an initialized LLM client.")

    _notify(progress, "Loading case definition")
    definition = load_case_definition(case_path)

    _notify(progress, "Planning retrieval")
    acquisition = await acquire_case_documents(
        definition.case,
        seeds=definition.seeds.sources if definition.seeds else None,
        search_fn=search_fn or web_search,
    )
    if not acquisition.documents:
        raise RuntimeError("Casefile acquisition produced no usable documents.")

    _notify(progress, "Extracting documents")
    extracted_documents: list[ExtractedDocument] = []
    for document in acquisition.documents:
        extracted_documents.append(await extract_document(llm, definition.case, document))

    _notify(progress, "Building evidence ledger")
    evidence = build_evidence(extracted_documents, max_evidence=definition.case.build.max_evidence)
    if not evidence:
        raise RuntimeError("Casefile build produced no evidence items.")

    operational: dict | None = None
    if store is not None:
        _notify(progress, "Persisting case scope")
        operational = await persist_case_operational_state(
            store,
            llm,
            definition.case,
            acquisition,
            extracted_documents,
            evidence,
            progress=progress,
        )

    _notify(progress, "Synthesizing casefile")
    bundle = await synthesize_casefile(
        llm,
        definition.case,
        acquisition,
        extracted_documents,
        evidence,
        thread_context={
            "threads": operational.get("threads", []) if operational else [],
            "divergence": operational.get("divergence", []) if operational else [],
            "convergence": operational.get("convergence", []) if operational else [],
            "graph_summary": build_case_graph_summary(operational.get("graph", {})).model_dump(mode="json")
            if operational
            else {},
        },
    )

    if store is not None and operational is not None:
        case_id = operational["case_id"]
        _notify(progress, "Generating case assessments")
        assessments = await generate_case_assessments(
            store,
            llm,
            case_id,
            definition.case,
            bundle.hypotheses,
            bundle.evidence,
        )
        await store.replace_case_hypotheses(case_id, [item.model_dump(mode="json") for item in bundle.hypotheses])
        await store.replace_case_open_questions(case_id, bundle.open_questions)
        bundle.threads = build_case_thread_models(
            operational.get("threads", []),
            operational.get("divergence", []),
            operational.get("convergence", []),
        )
        bundle.divergence = build_case_divergence_models(operational.get("divergence", []))
        bundle.assessments = assessments
        bundle.graph = build_case_graph_summary(operational.get("graph", {}))
        bundle.recent_changes = build_recent_changes(
            definition.case,
            assessments,
            bundle.threads,
            bundle.divergence,
        )

    _notify(progress, "Running readiness review")
    bundle.review = await review_casefile(llm, bundle)
    bundle = apply_readiness_gate(bundle)

    _notify(progress, "Saving bundle")
    save_casefile(case_path, bundle)
    return bundle
