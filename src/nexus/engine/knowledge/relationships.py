"""Entity-entity relationship extraction and invalidation.

Extracts typed, temporal relationships from events using LLM.
Implements bi-temporal edge invalidation (Graphiti pattern):
when new info contradicts an existing relationship, the old
one gets valid_until set rather than deleted.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import date

from nexus.engine.knowledge.events import Event
from nexus.llm.client import LLMClient

logger = logging.getLogger(__name__)

RELATION_TYPES = [
    "threatens",
    "allies_with",
    "sanctions",
    "partners_with",
    "competes_with",
    "leads",
    "funds",
    "opposes",
    "negotiates_with",
    "deploys_to",
    "regulates",
    "acquires",
    "supplies",
]

# Contradiction pairs: if we see a new relationship of type A between
# two entities, and there's an existing active relationship of type B,
# the old one should be invalidated.
CONTRADICTION_PAIRS: dict[str, set[str]] = {
    "threatens": {"allies_with", "partners_with", "negotiates_with"},
    "allies_with": {"threatens", "opposes", "sanctions"},
    "sanctions": {"allies_with", "partners_with", "funds"},
    "partners_with": {"competes_with", "opposes", "sanctions", "threatens"},
    "competes_with": {"partners_with", "allies_with"},
    "opposes": {"allies_with", "partners_with", "funds"},
    "funds": {"sanctions", "opposes"},
    "negotiates_with": {"threatens"},
}

EXTRACTION_SYSTEM_PROMPT = (
    "You extract typed entity-entity relationships from news events. "
    "Return a JSON object with a 'relationships' array. Each relationship has: "
    "source_entity (string), target_entity (string), relation_type (one of: "
    + ", ".join(RELATION_TYPES)
    + "), evidence_text (brief quote from the event), strength (0.1-1.0). "
    "Only extract relationships explicitly supported by the event text. "
    "Return at most 5 relationships. If no clear relationships exist, "
    "return {\"relationships\": []}."
)


@dataclass
class ExtractedRelationship:
    source_entity: str
    target_entity: str
    relation_type: str
    evidence_text: str
    strength: float
    valid_from: date


async def extract_relationships_from_event(
    llm: LLMClient,
    event: Event,
    *,
    existing_relationships: list[dict],
) -> list[ExtractedRelationship]:
    """Extract entity-entity relationships from a single event.

    Uses gemini-3-flash (config_key='knowledge_summary') for cost.
    Returns 0-5 typed relationships. Prompt includes existing active
    relationships involving the same entities so the LLM can flag
    contradictions.
    """
    entities = event.raw_entities or event.entities
    if len(entities) < 2:
        return []

    # Build context about existing relationships for contradiction awareness
    existing_context = ""
    if existing_relationships:
        lines = []
        for r in existing_relationships[:10]:
            src = r.get("source_entity_name", "?")
            tgt = r.get("target_entity_name", "?")
            lines.append(f"  {src} --[{r['relation_type']}]--> {tgt} (since {r['valid_from']})")
        existing_context = (
            "\n\nExisting active relationships involving these entities:\n"
            + "\n".join(lines)
        )

    user_prompt = (
        f"Event date: {event.date}\n"
        f"Event: {event.summary}\n"
        f"Entities mentioned: {', '.join(entities)}"
        f"{existing_context}\n\n"
        f"Extract relationships. Valid types: {', '.join(RELATION_TYPES)}"
    )

    try:
        raw = await llm.complete(
            config_key="knowledge_summary",
            system_prompt=EXTRACTION_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            json_response=True,
        )
        # JSON repair: strip markdown fences, think tags
        cleaned = re.sub(r"```json\s*|\s*```", "", raw)
        cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL)
        data = json.loads(cleaned)
    except Exception as exc:
        logger.warning("Relationship extraction failed for event '%s': %s",
                       event.summary[:50], exc)
        return []

    results = []
    for item in data.get("relationships", []):
        rel_type = item.get("relation_type", "")
        if rel_type not in RELATION_TYPES:
            continue
        strength = max(0.1, min(1.0, float(item.get("strength", 0.5))))
        results.append(ExtractedRelationship(
            source_entity=item.get("source_entity", ""),
            target_entity=item.get("target_entity", ""),
            relation_type=rel_type,
            evidence_text=item.get("evidence_text", ""),
            strength=strength,
            valid_from=event.date if isinstance(event.date, date) else date.fromisoformat(str(event.date)),
        ))

    return results


async def invalidate_contradicted_relationships(
    store,  # KnowledgeStore — avoid circular import
    new_relationships: list[ExtractedRelationship],
    event_date: date,
) -> int:
    """Mark old relationships as invalid when new ones contradict them.

    E.g., if 'Iran allies_with Hamas' exists and we extract 'Iran opposes Hamas',
    set valid_until=event_date on the old allies_with.

    Returns count of invalidated relationships.
    """
    count = 0
    for new_rel in new_relationships:
        contradicts = CONTRADICTION_PAIRS.get(new_rel.relation_type, set())
        if not contradicts:
            continue

        # Find the source and target entities in the store
        src = await store.find_entity(new_rel.source_entity)
        tgt = await store.find_entity(new_rel.target_entity)
        if not src or not tgt:
            continue

        # Get active relationships between these two entities
        existing = await store.get_relationships_between(
            src["id"], tgt["id"], as_of=event_date
        )

        for old_rel in existing:
            if old_rel["relation_type"] in contradicts:
                await store.invalidate_relationship(old_rel["id"], event_date)
                logger.info(
                    "Invalidated: %s -[%s]-> %s (contradicted by %s)",
                    new_rel.source_entity,
                    old_rel["relation_type"],
                    new_rel.target_entity,
                    new_rel.relation_type,
                )
                count += 1

    return count
