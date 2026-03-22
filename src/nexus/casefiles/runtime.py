"""Case runtime helpers bridging case-native data to the shared knowledge store."""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import Callable

from nexus.casefiles.models import (
    AcquisitionResult,
    CaseAssessment,
    CaseConfig,
    CaseDivergence,
    CaseGraphSummary,
    CaseHypothesis,
    CaseThread,
    EvidenceItem,
    ExtractedDocument,
)
from nexus.config.models import TopicConfig
from nexus.engine.knowledge.entities import resolve_entities
from nexus.engine.knowledge.events import Event
from nexus.engine.knowledge.relationships import (
    extract_relationships_from_event,
    invalidate_contradicted_relationships,
)
from nexus.engine.projection.evidence import EvidencePackage
from nexus.engine.projection.models import ThreadSnapshot
from nexus.engine.projection.structural_engine import predict_structural
from nexus.engine.sources.polling import ContentItem
from nexus.engine.synthesis.knowledge import synthesize_topic

ProgressCallback = Callable[[str], None] | None

CASE_TOPIC_PREFIX = "__case__:"


def case_topic_slug(slug: str) -> str:
    """Hidden topic slug used only to satisfy the shared events table contract."""
    return f"{CASE_TOPIC_PREFIX}{slug}"


def _notify(progress: ProgressCallback, message: str) -> None:
    if progress:
        progress(message)


def _parse_date_hint(value: str | None) -> date | None:
    if not value:
        return None
    text = value.strip()
    for pattern in (r"\b(\d{4}-\d{2}-\d{2})\b", r"\b(\d{4}-\d{2})\b", r"\b(\d{4})\b"):
        match = re.search(pattern, text)
        if not match:
            continue
        token = match.group(1)
        try:
            if len(token) == 10:
                return date.fromisoformat(token)
            if len(token) == 7:
                return date.fromisoformat(token + "-01")
            if len(token) == 4:
                return date(int(token), 1, 1)
        except ValueError:
            continue
    return None


def _event_significance(document: ExtractedDocument) -> int:
    base = 5
    if document.role == "primary":
        base += 1
    if document.quality_label == "high":
        base += 2
    elif document.quality_label == "medium":
        base += 1
    if document.priority >= 8:
        base += 1
    return max(3, min(base, 9))


def build_case_events(documents: list[ExtractedDocument]) -> list[Event]:
    """Distill extracted documents into dated events for shared thread/graph machinery."""
    events: list[Event] = []
    seen: set[tuple[str, date]] = set()

    for document in documents:
        event_date = _parse_date_hint(document.time_anchors[0]) if document.time_anchors else None
        if event_date is None and document.published_at:
            event_date = _parse_date_hint(document.published_at)
        if event_date is None:
            event_date = date.today()

        summary = document.summary.strip()
        if not summary and document.claims:
            summary = document.claims[0].claim.strip()
        if not summary:
            continue
        summary = re.sub(r"\s+", " ", summary)[:260]
        key = (summary.lower(), event_date)
        if key in seen:
            continue
        seen.add(key)

        events.append(
            Event(
                date=event_date,
                summary=summary,
                sources=[
                    {
                        "url": document.canonical_url,
                        "language": "en",
                        "outlet": document.source_label,
                        "affiliation": document.source_class,
                        "country": "",
                        "framing": f"[{document.source_class}] {document.summary[:120]}",
                    }
                ],
                entities=[entity.name for entity in document.entities[:8]],
                raw_entities=[entity.name for entity in document.entities[:8]],
                relation_to_prior=document.claims[0].why_it_matters if document.claims else "",
                significance=_event_significance(document),
            )
        )

    events.sort(key=lambda item: (item.date, item.significance))
    return events


def build_case_articles(documents: list[ExtractedDocument], acquisition: AcquisitionResult) -> list[ContentItem]:
    """Build synthetic ContentItems for thread synthesis from fetched case documents."""
    fetched_by_id = {document.id: document for document in acquisition.documents}
    articles: list[ContentItem] = []
    for document in documents:
        fetched = fetched_by_id.get(document.id)
        published = None
        if document.published_at:
            try:
                published = datetime.fromisoformat(str(document.published_at).replace("Z", "+00:00"))
            except ValueError:
                published = None
        articles.append(
            ContentItem(
                title=document.title,
                url=document.canonical_url,
                source_id=document.source_label,
                snippet=document.summary[:240],
                published=published,
                full_text=fetched.raw_text if fetched else document.summary,
                source_language="en",
                source_affiliation=document.source_class,
                source_country="",
                source_tier="casefile",
            )
        )
    return articles


def build_case_topic(case: CaseConfig) -> TopicConfig:
    """Create a synthetic TopicConfig so the old synthesis path can be reused."""
    scope = "medium"
    if len(case.hypothesis_seeds) >= 4:
        scope = "broad"
    return TopicConfig(
        name=case.title,
        subtopics=case.hypothesis_seeds[:6],
        scope=scope,
        projection_eligible=False,
    )


async def persist_case_operational_state(
    store,
    llm,
    case: CaseConfig,
    acquisition: AcquisitionResult,
    documents: list[ExtractedDocument],
    evidence: list[EvidenceItem],
    *,
    progress: ProgressCallback = None,
) -> dict:
    """Persist documents, events, entities, relationships, and threads for a case."""
    case_id = await store.upsert_case(
        case.slug,
        case.title,
        case.question,
        time_bounds=case.time_bounds.model_dump(),
        build_defaults=case.build.model_dump(),
        monitoring_enabled=True,
    )

    await store.reset_case_scope(case_id)
    await store.replace_case_documents(case_id, [document.model_dump(mode="json") for document in documents])
    await store.replace_case_evidence(case_id, [item.model_dump(mode="json") for item in evidence])

    events = build_case_events(documents)
    if not events:
        return {"case_id": case_id, "event_count": 0, "threads": [], "graph": {"nodes": [], "links": []}}

    _notify(progress, "Resolving case entities")
    all_raw = sorted({name for event in events for name in event.raw_entities})
    known = await store.get_all_entities(case_id=case_id)
    resolutions = await resolve_entities(llm, all_raw, known)
    resolve_map: dict[str, tuple[int, str]] = {}
    latest_date = max(event.date for event in events)
    for resolution in resolutions:
        aliases = [resolution.raw] if resolution.raw != resolution.canonical else []
        entity_id = await store.upsert_entity(
            resolution.canonical,
            resolution.entity_type,
            aliases,
            observation_date=latest_date,
        )
        resolve_map[resolution.raw] = (entity_id, resolution.canonical)

    for event in events:
        canonical_entities: list[str] = []
        seen_entities: set[str] = set()
        for raw in event.raw_entities or event.entities:
            canonical = resolve_map.get(raw, (None, raw))[1]
            key = canonical.lower()
            if key in seen_entities:
                continue
            seen_entities.add(key)
            canonical_entities.append(canonical)
        event.entities = canonical_entities

    _notify(progress, "Persisting case events")
    synthetic_slug = case_topic_slug(case.slug)
    event_ids = await store.add_events(events, synthetic_slug, case_id=case_id)
    for event_id, event in zip(event_ids, events):
        event.event_id = event_id
        entity_ids = [
            resolve_map[name][0]
            for name in (event.raw_entities or event.entities)
            if name in resolve_map
        ]
        if entity_ids:
            await store.link_event_entities(event_id, entity_ids)

    _notify(progress, "Extracting entity relationships")
    rel_count = 0
    for event in events:
        if not event.event_id:
            continue
        existing_rels: list[dict] = []
        seen_entity_ids: set[int] = set()
        for name in event.raw_entities or event.entities:
            if name in resolve_map:
                entity_id = resolve_map[name][0]
                if entity_id in seen_entity_ids:
                    continue
                seen_entity_ids.add(entity_id)
                existing_rels.extend(await store.get_active_relationships_for_entity(entity_id))
        extracted = await extract_relationships_from_event(llm, event, existing_relationships=existing_rels)
        if not extracted:
            continue
        await invalidate_contradicted_relationships(store, extracted, event.date)
        for relationship in extracted:
            source_id = resolve_map.get(relationship.source_entity, (None,))[0]
            target_id = resolve_map.get(relationship.target_entity, (None,))[0]
            if source_id and target_id:
                await store.save_entity_relationship(
                    {
                        "source_entity_id": source_id,
                        "target_entity_id": target_id,
                        "relation_type": relationship.relation_type,
                        "evidence_text": relationship.evidence_text,
                        "source_event_id": event.event_id,
                        "strength": relationship.strength,
                        "valid_from": relationship.valid_from.isoformat(),
                    }
                )
                rel_count += 1

    _notify(progress, "Synthesizing case threads")
    articles = build_case_articles(documents, acquisition)
    topic = build_case_topic(case)
    synthesis = await synthesize_topic(
        llm,
        topic,
        events=events,
        articles=articles,
        weekly_summaries=[],
        monthly_summaries=[],
        store=store,
        case_id=case_id,
    )

    today = date.today()
    thread_rows = await store.get_threads_for_case(case_id)
    for thread in thread_rows:
        events_for_thread = await store.get_events_for_thread(thread["id"])
        latest_event_date = max((event.date for event in events_for_thread), default=today)
        await store.upsert_thread_snapshot(
            ThreadSnapshot(
                thread_id=thread["id"],
                snapshot_date=today,
                status=thread.get("status", "emerging"),
                significance=thread.get("significance", 5),
                event_count=len(events_for_thread),
                latest_event_date=latest_event_date,
            )
        )

    graph = await store.get_case_graph_data(case_id)
    return {
        "case_id": case_id,
        "event_count": len(events),
        "relationship_count": rel_count,
        "synthesis": synthesis,
        "threads": await store.get_threads_for_case(case_id),
        "divergence": await store.get_case_divergence(case_id),
        "convergence": await store.get_case_convergence(case_id),
        "graph": graph,
    }


async def build_case_evidence_package(
    store,
    case_id: int,
    question: str,
    *,
    as_of: date,
) -> EvidencePackage:
    """Assemble structural evidence for a case assessment."""
    pkg = EvidencePackage(question=question, as_of=as_of)
    entities = await store.get_all_entities(case_id=case_id)
    threads = await store.get_threads_for_case(case_id)
    convergence = await store.get_case_convergence(case_id)
    divergence = await store.get_case_divergence(case_id)
    events = await store.get_all_events(case_id=case_id)

    pkg.entities = [
        {
            "name": entity.get("canonical_name", ""),
            "entity_id": entity.get("id"),
            "entity_type": entity.get("entity_type", ""),
        }
        for entity in entities[:12]
    ]
    pkg.threads = [
        {
            "thread_id": thread["id"],
            "headline": thread["headline"],
            "status": thread.get("status", "unknown"),
            "trajectory_label": thread.get("trajectory_label"),
            "momentum_score": thread.get("momentum_score"),
            "velocity_7d": thread.get("velocity_7d"),
            "acceleration_7d": thread.get("acceleration_7d"),
            "significance": thread.get("significance"),
        }
        for thread in threads[:6]
    ]
    pkg.convergence = convergence[:10]
    pkg.divergence = divergence[:8]
    pkg.recent_events = [
        {
            "event_id": event.event_id,
            "date": event.date.isoformat(),
            "summary": event.summary,
            "significance": event.significance,
            "entities": event.entities,
        }
        for event in sorted(events, key=lambda item: (item.date, item.significance), reverse=True)[:12]
    ]

    relationships: list[dict] = []
    for entity in entities[:8]:
        relationships.extend(await store.get_active_relationships_for_entity(entity["id"], as_of=as_of))
    deduped_relationships: list[dict] = []
    seen_relationships: set[int] = set()
    for relationship in relationships:
        rel_id = relationship.get("id")
        if rel_id in seen_relationships:
            continue
        seen_relationships.add(rel_id)
        deduped_relationships.append(relationship)
    pkg.relationships = deduped_relationships[:15]
    pkg.coverage = {
        "entities_found": len(pkg.entities),
        "threads_found": len(pkg.threads),
        "events_found": len(pkg.recent_events),
        "convergence_found": len(pkg.convergence),
        "divergence_found": len(pkg.divergence),
        "relationships_found": len(pkg.relationships),
    }
    return pkg


async def generate_case_assessments(
    store,
    llm,
    case_id: int,
    case: CaseConfig,
    hypotheses: list[CaseHypothesis],
    evidence_items: list[EvidenceItem],
) -> list[CaseAssessment]:
    """Generate posterior and forecast assessments using the structural engine."""
    assessments: list[CaseAssessment] = []
    today = date.today()
    threads = await store.get_threads_for_case(case_id)
    top_thread_ids = [thread["id"] for thread in threads[:4]]

    for hypothesis in hypotheses[:4]:
        question = f"What is the best current probability that {hypothesis.title}?"
        pkg = await build_case_evidence_package(store, case_id, question, as_of=today)
        result = await predict_structural(llm, pkg, config_key="knowledge_summary")
        evidence_ids = hypothesis.evidence_for[:3] + hypothesis.evidence_against[:2]
        assessments.append(
            CaseAssessment(
                id=f"A-{hypothesis.id.lower()}",
                target_hypothesis_id=hypothesis.id,
                mode="posterior",
                question=question,
                probability=result.implied_probability,
                confidence=result.confidence,
                rationale=result.reasoning,
                counterarguments=[result.contrarian_view, *result.key_uncertainties][:4],
                evidence_ids=evidence_ids,
                evidence_thread_ids=top_thread_ids[:3],
                signposts=result.signposts[:4],
            )
        )

    if hypotheses:
        forecast_question = (
            f"Will materially new primary-source evidence change the current ranking of "
            f"{case.title} hypotheses within the next 12 months?"
        )
        pkg = await build_case_evidence_package(store, case_id, forecast_question, as_of=today)
        result = await predict_structural(llm, pkg, config_key="knowledge_summary")
        assessments.append(
            CaseAssessment(
                id="A-forecast-leading",
                target_hypothesis_id=hypotheses[0].id,
                mode="forecast",
                question=forecast_question,
                probability=result.implied_probability,
                confidence=result.confidence,
                rationale=result.reasoning,
                counterarguments=[result.contrarian_view, *result.key_uncertainties][:4],
                evidence_ids=hypotheses[0].evidence_for[:3] + hypotheses[0].evidence_against[:2],
                evidence_thread_ids=top_thread_ids[:3],
                signposts=result.signposts[:4],
            )
        )

    await store.replace_case_assessments(
        case_id,
        [assessment.model_dump(mode="json") for assessment in assessments],
    )
    return assessments


def build_case_thread_models(threads: list[dict], divergence: list[dict], convergence: list[dict]) -> list[CaseThread]:
    """Convert persisted thread rows into bundle models."""
    divergence_counts: dict[int, int] = {}
    convergence_counts: dict[int, int] = {}
    for item in divergence:
        divergence_counts[item["thread_id"]] = divergence_counts.get(item["thread_id"], 0) + 1
    for item in convergence:
        convergence_counts[item["thread_id"]] = convergence_counts.get(item["thread_id"], 0) + 1

    results: list[CaseThread] = []
    for thread in threads:
        results.append(
            CaseThread(
                thread_id=thread["id"],
                slug=thread["slug"],
                headline=thread["headline"],
                status=thread.get("status", "emerging"),
                significance=thread.get("significance", 5),
                key_entities=thread.get("key_entities", []),
                event_count=thread.get("event_count", 0),
                convergence_count=convergence_counts.get(thread["id"], 0),
                divergence_count=divergence_counts.get(thread["id"], 0),
                trajectory_label=thread.get("trajectory_label"),
                momentum_score=thread.get("momentum_score"),
                velocity_7d=thread.get("velocity_7d"),
                acceleration_7d=thread.get("acceleration_7d"),
                snapshot_count=thread.get("snapshot_count"),
                created_at=thread.get("created_at"),
                updated_at=thread.get("updated_at"),
            )
        )
    return results


def build_case_divergence_models(items: list[dict]) -> list[CaseDivergence]:
    """Convert persisted divergence rows into bundle models."""
    return [CaseDivergence.model_validate(item) for item in items]


def build_case_graph_summary(graph_data: dict) -> CaseGraphSummary:
    """Create a compact graph summary for the case bundle."""
    nodes = graph_data.get("nodes", [])
    links = graph_data.get("links", [])
    top_entities = [node.get("name", "") for node in sorted(nodes, key=lambda item: item.get("event_count", 0), reverse=True)[:6]]
    return CaseGraphSummary(
        node_count=len(nodes),
        edge_count=len(links),
        top_entities=[name for name in top_entities if name],
    )


def build_recent_changes(
    case: CaseConfig,
    assessments: list[CaseAssessment],
    threads: list[CaseThread],
    divergence: list[CaseDivergence],
) -> list[str]:
    """Build compact update bullets for the case bundle."""
    changes: list[str] = []
    if threads:
        changes.append(f"{len(threads)} active case threads are now tracked inside {case.title}.")
    if divergence:
        changes.append(f"{len(divergence)} explicit divergence records are mapped across case threads.")
    posterior = [item for item in assessments if item.mode == "posterior"]
    if posterior:
        leading = sorted(posterior, key=lambda item: item.probability, reverse=True)[0]
        changes.append(
            f"Current leading assessed account: {leading.target_hypothesis_id} at roughly {round(leading.probability * 100)}%."
        )
    return changes[:4]
