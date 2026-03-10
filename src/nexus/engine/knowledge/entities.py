"""Entity resolution — canonicalize raw entity strings into graph nodes."""

import json
import logging
from dataclasses import dataclass

from nexus.llm.client import LLMClient

logger = logging.getLogger(__name__)


@dataclass
class EntityResolution:
    """Result of resolving a raw entity string."""
    raw: str
    canonical: str
    entity_type: str  # person|org|country|treaty|concept|unknown
    is_new: bool


RESOLVE_SYSTEM_PROMPT = (
    "You are an entity resolution engine. Given a list of raw entity names "
    "extracted from news articles and a list of known canonical entities, "
    "map each raw name to its canonical form.\n\n"
    "Rules:\n"
    "- If a raw name matches a known entity (same entity, different spelling), "
    "map it to that entity's canonical_name.\n"
    "- If a raw name is a new entity not in the known list, assign it a clean "
    "canonical name and classify its type.\n"
    "- Entity types: person, org, country, treaty, concept, unknown.\n"
    "- Be conservative: only merge entities you are CERTAIN are the same. "
    "'Iran' (country) and 'Iran Air' (org) are different entities.\n"
    "- Use the most formal/complete canonical form: "
    "'US Treasury' → 'US Department of the Treasury', "
    "'Khamenei' → 'Ali Khamenei'.\n\n"
    "Respond with a JSON array:\n"
    '[{"raw": "...", "canonical": "...", "type": "...", "is_new": true/false}]'
)


async def resolve_entities(
    llm: LLMClient,
    raw_entity_names: list[str],
    known_entities: list[dict],
) -> list[EntityResolution]:
    """Resolve raw entity strings to canonical entities via LLM.

    Args:
        llm: LLM client for resolution calls.
        raw_entity_names: Unique entity names from today's events.
        known_entities: Existing entities from the store
            [{canonical_name, entity_type, aliases}].

    Returns:
        List of EntityResolution objects mapping raw → canonical.
    """
    if not raw_entity_names:
        return []

    # Build known entities context
    known_lines = []
    for e in known_entities[:200]:  # Cap to avoid prompt overflow
        aliases = e.get("aliases", [])
        alias_str = f" (aliases: {', '.join(aliases)})" if aliases else ""
        known_lines.append(
            f"- {e['canonical_name']} [{e.get('entity_type', 'unknown')}]{alias_str}"
        )

    known_context = "\n".join(known_lines) if known_lines else "None yet"

    user_prompt = (
        f"Known entities:\n{known_context}\n\n"
        f"Raw entity names to resolve:\n"
        + "\n".join(f"- {name}" for name in raw_entity_names)
    )

    try:
        response = await llm.complete(
            config_key="knowledge_summary",
            system_prompt=RESOLVE_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            json_response=True,
        )
        data = json.loads(response)
        if not isinstance(data, list):
            data = [data]

        resolutions = []
        for item in data:
            resolutions.append(EntityResolution(
                raw=item["raw"],
                canonical=item.get("canonical", item["raw"]),
                entity_type=item.get("type", "unknown"),
                is_new=item.get("is_new", True),
            ))
        return resolutions

    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logger.warning(f"Entity resolution failed: {e}. Using raw names as-is.")
        # Fallback: treat each raw name as its own canonical entity
        return [
            EntityResolution(raw=name, canonical=name, entity_type="unknown", is_new=True)
            for name in raw_entity_names
        ]
