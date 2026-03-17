"""GraphRAG-enhanced prediction engine — actual graph traversal + structured reasoning.

Key difference from actor engine: instead of reasoning about actors independently,
this engine traverses the knowledge graph (relationships, causal chains, cross-topic
signals) to build a holistic evidence picture, then reasons about the system as a whole.

Pipeline (2 LLM calls):
1. Entity extraction from question (1 LLM call)
2. Multi-hop graph traversal + evidence ranking (0 LLM calls)
3. Graph-informed reasoning (1 LLM call)
4. Calibrate
"""

from __future__ import annotations

import json
import logging
import re
from datetime import date, timedelta

from nexus.engine.knowledge.store import KnowledgeStore
from nexus.engine.projection.forecasting import (
    ForecastEngineInput,
    _clip_probability,
)
from nexus.engine.projection.models import ForecastQuestion, ForecastRun
from nexus.engine.projection.swarm import anchor_blend, extremize

logger = logging.getLogger(__name__)


# ── Prompts ───────────────────────────────────────────────────────────

ENTITY_EXTRACTION_SYSTEM = (
    "You extract entity names from questions. Return JSON only."
)

ENTITY_EXTRACTION_PROMPT = (
    'Question: {question}\n\n'
    'Extract the key entities (countries, organizations, people, technologies) '
    'mentioned or implied in this question.\n\n'
    'Return JSON: {{"entities": ["entity1", "entity2", ...]}}'
)

GRAPH_REASONING_SYSTEM = (
    "You are an analyst using structured intelligence to predict real-world outcomes. "
    "The following evidence was collected from our automated news monitoring system. "
    "Some of this information may be more recent than your training data. "
    "When our monitored intelligence conflicts with your prior knowledge, "
    "treat our data as ground truth — it is more recent.\n\n"
    "Use both the graph structure (who is connected to whom, how relationships "
    "have changed) and the event timeline to reason about the question."
)

GRAPH_REASONING_PROMPT = """\
Question: {question}

Today's date: {as_of}

== ENTITY MAP ==
{entity_section}

== RELATIONSHIPS (active as of {as_of}) ==
{relationship_section}

== RECENT EVENTS (most relevant first) ==
{event_section}

== CAUSAL CHAINS ==
{causal_section}

{market_section}

Based on the graph structure and evidence above, what is the probability that \
the answer to this question is YES?

Consider:
- How entities are connected and what those connections imply
- The direction relationships have been changing (new vs. invalidated)
- Causal chains between events
- What is NOT present in the data (absence of evidence)

Return JSON only:
{{"probability": <float 0.05-0.95>, "reasoning": "...", \
"key_uncertainties": ["..."], "signposts": ["..."]}}
"""


# ── Entity Extraction ─────────────────────────────────────────────────


async def extract_entities_from_question(
    store: KnowledgeStore,
    llm,
    question: str,
    *,
    max_entities: int = 6,
) -> list[dict]:
    """Extract entities from question text and resolve against the store.

    Uses LLM for semantic extraction when available, falls back to keyword matching.
    Returns list of {name, entity_id, match_source}.
    """
    entity_names: list[str] = []

    # LLM-based extraction
    if llm is not None:
        try:
            response = await llm.complete(
                config_key="knowledge_summary",
                system_prompt=ENTITY_EXTRACTION_SYSTEM,
                user_prompt=ENTITY_EXTRACTION_PROMPT.format(question=question),
                json_response=True,
            )
            data = json.loads(response)
            entity_names = data.get("entities", [])[:max_entities]
        except Exception as exc:
            logger.warning("Entity extraction LLM failed: %s", exc)

    # Resolve against store
    results: list[dict] = []
    seen_ids: set[int] = set()

    for name in entity_names:
        entity = await store.find_entity(name)
        if entity and entity["id"] not in seen_ids:
            seen_ids.add(entity["id"])
            results.append({
                "name": entity["canonical_name"],
                "entity_id": entity["id"],
                "match_source": "llm+store",
            })
        else:
            results.append({
                "name": name,
                "entity_id": None,
                "match_source": "llm_only",
            })

    # Keyword fallback if no LLM results
    if not entity_names:
        all_entities = await store.get_all_entities()
        for entity in all_entities:
            name = entity.get("canonical_name", "")
            if (
                name
                and re.search(r"\b" + re.escape(name) + r"\b", question, re.IGNORECASE)
                and entity["id"] not in seen_ids
            ):
                seen_ids.add(entity["id"])
                results.append({
                    "name": name,
                    "entity_id": entity["id"],
                    "match_source": "keyword",
                })
                if len(results) >= max_entities:
                    break

    return results[:max_entities]


# ── Graph Traversal ───────────────────────────────────────────────────


async def gather_graph_evidence(
    store: KnowledgeStore,
    entity_ids: list[int],
    *,
    as_of: date | None = None,
    event_days: int = 30,
    max_hops: int = 2,
) -> dict:
    """Traverse the knowledge graph from seed entities, collecting evidence.

    Returns {events, relationships, neighbors, causal_chains}.
    """
    if not entity_ids:
        return {"events": [], "relationships": [], "neighbors": [], "causal_chains": []}

    all_events: list[dict] = []
    all_relationships: list[dict] = []
    all_neighbors: list[dict] = []
    all_causal: list[dict] = []
    seen_event_ids: set[int] = set()

    for entity_id in entity_ids:
        # Get events for this entity (returns Event pydantic objects)
        try:
            events = await store.get_events_for_entity(entity_id)
            for event in events:
                eid = getattr(event, "event_id", None)
                if eid and eid not in seen_event_ids:
                    seen_event_ids.add(eid)
                    all_events.append({
                        "id": eid,
                        "date": str(event.date),
                        "summary": event.summary,
                        "significance": event.significance,
                        "entities": event.entities,
                    })
        except Exception:
            pass

        # Get relationships
        try:
            rels = await store.get_active_relationships_for_entity(entity_id, as_of=as_of)
            all_relationships.extend(rels)
        except Exception:
            pass

        # Get multi-hop neighborhood
        try:
            neighborhood = await store.get_entity_neighborhood(
                entity_id, hops=max_hops, as_of=as_of, limit=30
            )
            for neighbor in neighborhood.get("entities", []):
                all_neighbors.append(neighbor)
        except Exception:
            pass

    # Get causal links for discovered events
    event_ids = list(seen_event_ids)
    if event_ids:
        try:
            causal_links = await store.get_causal_links_for_events(event_ids)
            all_causal.extend(causal_links)
        except Exception:
            pass

    return {
        "events": all_events,
        "relationships": all_relationships,
        "neighbors": all_neighbors,
        "causal_chains": all_causal,
    }


# ── Evidence Ranking ──────────────────────────────────────────────────


def rank_evidence(
    evidence: dict,
    *,
    as_of: date | None = None,
    max_events: int = 15,
    max_relationships: int = 20,
) -> dict:
    """Rank and cap evidence for prompt inclusion.

    Events ranked by: recency × significance.
    Relationships ranked by: strength.
    """
    ref_date = as_of or date.today()

    # Score events: recency × significance
    scored_events = []
    for event in evidence.get("events", []):
        event_date_str = event.get("date", "")
        try:
            event_date = date.fromisoformat(str(event_date_str))
            days_ago = max(1, (ref_date - event_date).days)
        except (ValueError, TypeError):
            days_ago = 30

        significance = float(event.get("significance", 5))
        score = significance / days_ago  # higher sig + more recent = better
        scored_events.append((score, event))

    scored_events.sort(key=lambda x: x[0], reverse=True)

    # Relationships: sort by strength if available
    rels = evidence.get("relationships", [])
    rels_sorted = sorted(rels, key=lambda r: float(r.get("strength", 0.5)), reverse=True)

    return {
        "events": [e for _, e in scored_events[:max_events]],
        "relationships": rels_sorted[:max_relationships],
        "neighbors": evidence.get("neighbors", [])[:10],
        "causal_chains": evidence.get("causal_chains", [])[:10],
    }


# ── Prompt Formatting ─────────────────────────────────────────────────


def _format_entity_section(entities: list[dict]) -> str:
    if not entities:
        return "(no entities identified)"
    lines = []
    for e in entities:
        eid = e.get("entity_id", "?")
        lines.append(f"- {e['name']} (id={eid}, source={e.get('match_source', '?')})")
    return "\n".join(lines)


def _format_relationship_section(relationships: list[dict]) -> str:
    if not relationships:
        return "(no relationships found)"
    lines = []
    for r in relationships:
        src = r.get("source_name", r.get("source_id", "?"))
        tgt = r.get("target_name", r.get("target_id", "?"))
        rtype = r.get("relation_type", "related_to")
        detail = r.get("detail", "")
        lines.append(f"- {src} --[{rtype}]--> {tgt}: {detail}")
    return "\n".join(lines[:20])


def _format_event_section(events: list[dict]) -> str:
    if not events:
        return "(no events found)"
    lines = []
    for e in events:
        d = e.get("date", "?")
        summary = e.get("summary", "")
        sig = e.get("significance", "?")
        lines.append(f"- [{d}] (sig={sig}) {summary}")
    return "\n".join(lines[:15])


def _format_causal_section(causal_chains: list[dict]) -> str:
    if not causal_chains:
        return "(no causal chains found)"
    lines = []
    for c in causal_chains:
        src = c.get("source_event_id", "?")
        tgt = c.get("target_event_id", "?")
        rtype = c.get("relation_type", "?")
        strength = c.get("strength", "?")
        lines.append(f"- Event {src} --[{rtype}, strength={strength}]--> Event {tgt}")
    return "\n".join(lines[:10])


# ── Main Engine ───────────────────────────────────────────────────────


class GraphRAGBenchmarkEngine:
    """Benchmark engine: graph traversal + structured reasoning."""

    engine_name = "graphrag"

    async def predict_probability(
        self,
        question: str,
        *,
        llm=None,
        store=None,
        market_prob: float | None = None,
        as_of: date | None = None,
    ) -> float:
        as_of = as_of or date.today()

        if llm is None:
            return 0.50

        # Step 1: Extract entities (1 LLM call)
        entities = []
        if store is not None:
            entities = await extract_entities_from_question(store, llm, question)

        # Step 2: Traverse graph (0 LLM calls)
        entity_ids = [e["entity_id"] for e in entities if e.get("entity_id") is not None]
        evidence: dict = {"events": [], "relationships": [], "neighbors": [], "causal_chains": []}

        if store is not None and entity_ids:
            evidence = await gather_graph_evidence(store, entity_ids, as_of=as_of)
            evidence = rank_evidence(evidence, as_of=as_of)

        # Step 3: Graph-informed reasoning (1 LLM call)
        market_section = ""
        if market_prob is not None:
            market_section = f"Market consensus probability: {market_prob:.1%}"

        user_prompt = GRAPH_REASONING_PROMPT.format(
            question=question,
            as_of=as_of.isoformat(),
            entity_section=_format_entity_section(entities),
            relationship_section=_format_relationship_section(evidence["relationships"]),
            event_section=_format_event_section(evidence["events"]),
            causal_section=_format_causal_section(evidence.get("causal_chains", [])),
            market_section=market_section,
        )

        try:
            response = await llm.complete(
                config_key="knowledge_summary",
                system_prompt=GRAPH_REASONING_SYSTEM,
                user_prompt=user_prompt,
                json_response=True,
            )
            data = json.loads(response)
            raw_prob = float(data.get("probability", 0.50))
        except Exception as exc:
            logger.warning("GraphRAG reasoning LLM failed: %s", exc)
            raw_prob = 0.50

        # Step 4: Calibrate
        calibrated = extremize(raw_prob, gamma=0.8)
        if market_prob is not None:
            calibrated = anchor_blend(calibrated, market_prob, swarm_weight=0.45)
        return _clip_probability(calibrated)


class GraphRAGForecastEngine:
    """Adapter for the full ForecastEngine protocol."""

    engine_name = "graphrag"

    async def generate(
        self,
        llm,
        payload: ForecastEngineInput,
        *,
        critic_pass: bool = True,
        max_questions: int = 4,
        calibration_data: list[dict] | None = None,
    ) -> ForecastRun:
        benchmark = GraphRAGBenchmarkEngine()
        questions: list[ForecastQuestion] = []

        # Access store from LLM if available
        store = getattr(llm, "_store", None) if llm else None

        for thread in payload.threads[:max_questions]:
            question_text = (
                f"Will the narrative '{thread.headline}' see significant "
                f"developments in the next 14 days?"
            )
            prob = await benchmark.predict_probability(
                question_text,
                llm=llm,
                store=store,
                as_of=payload.run_date,
            )
            questions.append(
                ForecastQuestion(
                    question=question_text,
                    forecast_type="binary",
                    target_variable="thread_development",
                    probability=prob,
                    base_rate=0.50,
                    resolution_criteria=f"Significant events in thread '{thread.headline}'",
                    resolution_date=payload.run_date + timedelta(days=14),
                    horizon_days=14,
                    signpost=f"Watch for events in: {thread.headline}",
                    signals_cited=["engine:graphrag"],
                )
            )

        return ForecastRun(
            topic_slug=payload.topic_slug,
            topic_name=payload.topic_name,
            engine="graphrag",
            generated_for=payload.run_date,
            summary=f"GraphRAG engine: {len(questions)} questions.",
            questions=questions,
            metadata={"engine": "graphrag"},
        )
