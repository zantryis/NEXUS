"""Graph-informed prediction engine.

Uses entity-relationship graph (Phase 1 SQLite layer) to give the LLM
structurally different information than flat text. Adopts key patterns
from MiroFish source analysis:
  - Sub-query decomposition (InsightForge): multi-query graph retrieval
  - "Forbidden to use own knowledge": forces grounding in graph evidence
  - Active/historical fact separation (PanoramaSearch): shows what changed
  - Minimum 3 evidence queries before synthesis (ReAct agent)

The deterministic trajectory probability is the anchor. The LLM can only
adjust it based on specific cited relationships from the graph.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import date, timedelta

from nexus.engine.projection.forecasting import (
    ForecastEngineInput,
    ForecastQuestion,
    ForecastRun,
    _build_candidate_catalog,
    _clip_probability,
)
from nexus.engine.projection.swarm import anchor_blend
from nexus.llm.client import LLMClient

logger = logging.getLogger(__name__)

# Max adjustment the LLM can make from the deterministic anchor (pp)
MAX_ADJUSTMENT_PP = 0.20
# Graph engine blending weight tiers (based on active relationship count)
# Sparse graph = less LLM influence to protect well-calibrated trajectory
BLEND_WEIGHT_TIERS = [
    (10, 0.35),  # 10+ active rels → full weight
    (4, 0.25),   # 4-9 active rels → moderate
    (1, 0.15),   # 1-3 active rels → conservative
]
BLEND_WEIGHT_DEFAULT = 0.0  # 0 active rels → skip (keep trajectory)


# ---------------------------------------------------------------------------
# Graph context builder
# ---------------------------------------------------------------------------


async def build_graph_context(
    store,
    question: ForecastQuestion,
    payload: ForecastEngineInput,
) -> dict:
    """Build structured graph context for a forecast question.

    Multi-query retrieval inspired by MiroFish InsightForge decomposition.
    Minimum 3 store queries enforced per question.

    Returns dict with:
      - active_relationships: currently valid edges
      - historical_relationships: recently invalidated edges
      - relationship_chains: 2-hop textual paths
      - stats: entity/relationship counts
    """
    # 1. Extract anchor entities from question metadata + thread entities
    anchor_names: list[str] = list(
        question.target_metadata.get("anchor_entities", [])
    )
    # Also pull entities from payload threads
    for thread in payload.threads:
        for entity in thread.key_entities or []:
            if entity not in anchor_names:
                anchor_names.append(entity)
    # Also from recent events
    for event in payload.recent_events:
        for entity in event.entities or []:
            if entity not in anchor_names:
                anchor_names.append(entity)

    # Cap at 6 to avoid excessive queries
    anchor_names = anchor_names[:6]

    if not anchor_names:
        return {
            "active_relationships": [],
            "historical_relationships": [],
            "relationship_chains": [],
            "stats": {
                "entities_in_neighborhood": 0,
                "active_count": 0,
                "invalidated_this_week": 0,
                "new_this_week": 0,
            },
        }

    # 2. Resolve anchor entities to IDs
    resolved: list[dict] = []
    for name in anchor_names:
        entity = await store.find_entity(name)
        if entity:
            resolved.append(entity)

    if not resolved:
        return {
            "active_relationships": [],
            "historical_relationships": [],
            "relationship_chains": [],
            "stats": {
                "entities_in_neighborhood": 0,
                "active_count": 0,
                "invalidated_this_week": 0,
                "new_this_week": 0,
            },
        }

    # 3. Sub-query decomposition — minimum 3 store queries
    all_relationships: list[dict] = []
    all_entities: list[dict] = []
    timeline_rels: list[dict] = []
    seen_rel_ids: set[int] = set()
    seen_entity_ids: set[int] = set()

    as_of = payload.run_date

    for entity in resolved:
        eid = entity["id"]

        # Query 1: entity neighborhood (2-hop BFS)
        neighborhood = await store.get_entity_neighborhood(
            eid, hops=2, as_of=as_of, limit=50,
        )
        for rel in neighborhood.get("relationships", []):
            rid = rel.get("id")
            if rid and rid not in seen_rel_ids:
                seen_rel_ids.add(rid)
                all_relationships.append(rel)
        for ent in neighborhood.get("entities", []):
            nid = ent.get("id")
            if nid and nid not in seen_entity_ids:
                seen_entity_ids.add(nid)
                all_entities.append(ent)

        # Query 2: relationship timeline (recent changes)
        timeline = await store.get_relationship_timeline(
            eid, days=14, reference_date=as_of,
        )
        for rel in timeline:
            rid = rel.get("id")
            if rid and rid not in seen_rel_ids:
                seen_rel_ids.add(rid)
                timeline_rels.append(rel)
            elif rid in seen_rel_ids:
                # Still add to timeline for active/historical separation
                timeline_rels.append(rel)

    # 4. Separate active from historical (MiroFish PanoramaSearch pattern)
    active_relationships: list[dict] = []
    historical_relationships: list[dict] = []

    for rel in all_relationships:
        if rel.get("valid_until") is not None:
            historical_relationships.append(rel)
        else:
            active_relationships.append(rel)

    # Also check timeline for historical rels not in neighborhood
    for rel in timeline_rels:
        if rel.get("valid_until") is not None:
            # Avoid duplicates
            if not any(h.get("id") == rel.get("id") for h in historical_relationships):
                historical_relationships.append(rel)

    # 5. Build relationship chains (2-hop textual paths)
    chains: list[str] = []
    # Build from active relationships: find 2-hop paths through shared entities
    entity_name_map: dict[int, str] = {}
    for ent in all_entities:
        entity_name_map[ent["id"]] = ent.get("name") or ent.get("canonical_name", "?")
    for ent in resolved:
        entity_name_map[ent["id"]] = ent.get("name") or ent.get("canonical_name", "?")

    # Index active rels by source
    rels_by_source: dict[int, list[dict]] = {}
    for rel in active_relationships:
        src = rel.get("source_entity_id")
        if src:
            rels_by_source.setdefault(src, []).append(rel)

    for rel1 in active_relationships:
        target1 = rel1.get("target_entity_id")
        if target1 and target1 in rels_by_source:
            for rel2 in rels_by_source[target1]:
                src_name = entity_name_map.get(
                    rel1.get("source_entity_id", 0), "?"
                )
                mid_name = entity_name_map.get(target1, "?")
                end_name = entity_name_map.get(
                    rel2.get("target_entity_id", 0), "?"
                )
                chain = (
                    f"{src_name} \u2192 {rel1.get('relation_type', '?')} \u2192 "
                    f"{mid_name} \u2192 {rel2.get('relation_type', '?')} \u2192 {end_name}"
                )
                if chain not in chains:
                    chains.append(chain)

    # 6. Compute stats
    week_ago = as_of - timedelta(days=7)
    week_ago_str = week_ago.isoformat()
    new_this_week = sum(
        1 for r in active_relationships
        if (r.get("valid_from") or "") >= week_ago_str
    )
    invalidated_this_week = sum(
        1 for r in historical_relationships
        if (r.get("valid_until") or "") >= week_ago_str
    )

    return {
        "active_relationships": active_relationships,
        "historical_relationships": historical_relationships,
        "relationship_chains": chains[:10],
        "stats": {
            "entities_in_neighborhood": len(all_entities) + len(resolved),
            "active_count": len(active_relationships),
            "invalidated_this_week": invalidated_this_week,
            "new_this_week": new_this_week,
        },
    }


# ---------------------------------------------------------------------------
# Prompt renderer
# ---------------------------------------------------------------------------


def render_graph_prompt(
    question: str,
    deterministic_probability: float,
    base_rate: float,
    graph_context: dict,
    run_date: date,
) -> str:
    """Render the structured graph-informed prediction prompt.

    Adopts MiroFish ReAct constraints:
    - "FORBIDDEN from using your own knowledge"
    - Each adjustment must cite a specific relationship
    - Max adjustment capped
    """
    # Active relationships section
    active_lines = []
    for rel in graph_context.get("active_relationships", []):
        src = rel.get("source_entity_name") or rel.get("source", "?")
        tgt = rel.get("target_entity_name") or rel.get("target", "?")
        rtype = rel.get("relation_type", "?")
        strength = rel.get("strength", "?")
        since = rel.get("valid_from", "?")
        evidence = rel.get("evidence_text") or rel.get("evidence", "")
        new_tag = ""
        if since and since >= (run_date - timedelta(days=3)).isoformat():
            new_tag = " (NEW)"
        active_lines.append(
            f"- {src} --[{rtype}]--> {tgt} (strength {strength}, since {since}{new_tag})"
        )
        if evidence:
            active_lines.append(f'  Evidence: "{evidence}"')

    active_section = "\n".join(active_lines) if active_lines else "No active relationships found."

    # Historical relationships section
    hist_lines = []
    for rel in graph_context.get("historical_relationships", []):
        src = rel.get("source_entity_name") or rel.get("source", "?")
        tgt = rel.get("target_entity_name") or rel.get("target", "?")
        rtype = rel.get("relation_type", "?")
        valid_from = rel.get("valid_from", "?")
        valid_until = rel.get("valid_until", "?")
        evidence = rel.get("evidence_text") or rel.get("evidence", "")
        hist_lines.append(
            f"- {src} --[{rtype}]--> {tgt} (INVALIDATED {valid_until})"
        )
        hist_lines.append(f"  Was valid since: {valid_from}")
        if evidence:
            hist_lines.append(f'  Evidence: "{evidence}"')

    hist_section = "\n".join(hist_lines) if hist_lines else "No recently invalidated relationships."

    # Relationship chains
    chains = graph_context.get("relationship_chains", [])
    chains_section = "\n".join(f"- {c}" for c in chains) if chains else "No multi-hop paths found."

    # Stats
    stats = graph_context.get("stats", {})

    return f"""\
CRITICAL: You are FORBIDDEN from using your own knowledge about world events.
You must reason ONLY from the entity relationship graph provided below.
If the graph does not contain evidence for an adjustment, do not make one.

## Forecast Question
{question}

## Deterministic Anchor: {deterministic_probability:.2f} (from trajectory + base rates)
## Base Rate: {base_rate:.2f}

## Active Relationships (current as of {run_date.isoformat()})
{active_section}

## Historical Relationships (recently invalidated)
{hist_section}

## Relationship Chains (2-hop paths)
{chains_section}

## Graph Statistics
- Entities within 2 hops: {stats.get('entities_in_neighborhood', 0)} | \
Active: {stats.get('active_count', 0)} | \
Invalidated this week: {stats.get('invalidated_this_week', 0)} | \
New: {stats.get('new_this_week', 0)}

## Your Task
Each adjustment must cite a specific relationship or change above. Max \u00b1{int(MAX_ADJUSTMENT_PP * 100)}pp.
Return JSON: {{"reasoning": "...", "adjustments": [{{"direction": "up|down", \
"amount": 0.XX, "evidence": "relationship citation"}}], "probability": 0.XX}}"""


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class GraphForecastEngine:
    """Graph-informed prediction engine.

    Uses entity-relationship graph to provide structurally different
    context than flat text engines. The deterministic trajectory
    probability is the anchor; the LLM adjusts based on graph evidence.
    """

    engine_name = "graph"

    async def generate(
        self,
        llm: LLMClient | None,
        payload: ForecastEngineInput,
        *,
        critic_pass: bool = True,
        max_questions: int = 4,
        calibration_data: list[dict] | None = None,
        store=None,
    ) -> ForecastRun:
        # 1. Build deterministic candidates (trajectory variant for anchor)
        candidates = _build_candidate_catalog(
            payload, "trajectory", max_questions,
            calibration_data=calibration_data,
        )

        if llm is not None and store is not None:
            for candidate in candidates:
                try:
                    # 2. Build graph context via multi-query retrieval
                    graph_ctx = await build_graph_context(
                        store, candidate, payload,
                    )

                    # 3. Compute adaptive blend weight from graph density
                    active_count = len(graph_ctx.get("active_relationships", []))
                    blend_weight = BLEND_WEIGHT_DEFAULT
                    for threshold, weight in BLEND_WEIGHT_TIERS:
                        if active_count >= threshold:
                            blend_weight = weight
                            break

                    if blend_weight == 0.0:
                        # No graph evidence — keep trajectory, skip LLM call
                        candidate.signals_cited = (candidate.signals_cited or []) + [
                            "graph:skipped=no_active_rels",
                        ]
                        continue

                    # 4. Render graph-informed prompt
                    prompt = render_graph_prompt(
                        question=candidate.question,
                        deterministic_probability=candidate.probability,
                        base_rate=candidate.base_rate,
                        graph_context=graph_ctx,
                        run_date=payload.run_date,
                    )

                    # 5. One smart LLM call per question
                    raw = await llm.complete(
                        config_key="synthesis",
                        system_prompt=(
                            "You are a graph-informed forecasting engine. "
                            "You adjust probabilities based ONLY on entity "
                            "relationship evidence. Never use your own knowledge."
                        ),
                        user_prompt=prompt,
                        json_response=True,
                    )

                    # 6. Parse LLM response with JSON repair
                    cleaned = re.sub(r"```json\s*|\s*```", "", raw)
                    cleaned = re.sub(
                        r"<think>.*?</think>", "", cleaned, flags=re.DOTALL,
                    )
                    data = json.loads(cleaned)

                    llm_prob = float(data.get("probability", candidate.probability))
                    llm_prob = max(0.05, min(0.95, llm_prob))

                    # 7. Anchor blend with adaptive weight
                    blended = anchor_blend(
                        llm_prob,
                        candidate.probability,
                        swarm_weight=blend_weight,
                    )
                    candidate.probability = _clip_probability(blended)

                    # Annotate with graph metadata
                    adjustments = data.get("adjustments", [])
                    candidate.signals_cited = (candidate.signals_cited or []) + [
                        f"graph:llm_raw={llm_prob:.3f}",
                        f"graph:blended={blended:.3f}",
                        f"graph:blend_weight={blend_weight:.2f}",
                        f"graph:active_rels={active_count}",
                        f"graph:hist_rels={len(graph_ctx['historical_relationships'])}",
                    ] + [
                        f"graph:adj:{a.get('direction','?')}:{a.get('amount',0):.2f}"
                        for a in adjustments[:3]
                    ]

                except Exception as exc:
                    logger.warning(
                        "Graph engine failed for question '%s': %s",
                        candidate.question[:60], exc,
                    )
                    # Keep deterministic probability as fallback

        return ForecastRun(
            topic_slug=payload.topic_slug,
            topic_name=payload.topic_name,
            engine=self.engine_name,
            generated_for=payload.run_date,
            summary=f"Graph-informed forecast for {payload.topic_name}.",
            questions=candidates,
            metadata={
                "graph_engine": True,
                "blend_weight_tiers": BLEND_WEIGHT_TIERS,
            },
        )
