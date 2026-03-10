"""Knowledge layer — per-topic event logging and persistence."""

import json
import logging
from datetime import date
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field

from nexus.config.models import TopicConfig
from nexus.engine.sources.polling import ContentItem
from nexus.llm.client import LLMClient

logger = logging.getLogger(__name__)


class Event(BaseModel):
    date: date
    summary: str
    sources: list[dict] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    relation_to_prior: str = ""
    significance: int = 5


def load_events(path: Path) -> list[Event]:
    """Load events from a YAML file. Returns empty list if file missing."""
    if not path.exists():
        return []
    raw = yaml.safe_load(path.read_text())
    if not raw:
        return []
    return [Event(**e) for e in raw]


def save_events(path: Path, events: list[Event]) -> None:
    """Write events to a YAML file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [e.model_dump(mode="json") for e in events]
    path.write_text(yaml.dump(data, default_flow_style=False, allow_unicode=True))


def append_events(path: Path, new_events: list[Event]) -> None:
    """Append new events to an existing events file."""
    existing = load_events(path)
    existing.extend(new_events)
    save_events(path, existing)


def is_duplicate_event(new: "Event", existing: "Event", entity_threshold: float = 0.6) -> bool:
    """Check if new event is a duplicate of existing based on entity overlap + date proximity.

    Returns True if >entity_threshold entity overlap on same date (±1 day).
    """
    if not new.entities or not existing.entities:
        return False

    # Date proximity: same day or adjacent
    date_diff = abs((new.date - existing.date).days)
    if date_diff > 1:
        return False

    # Entity overlap
    new_set = {e.lower() for e in new.entities}
    existing_set = {e.lower() for e in existing.entities}
    if not new_set or not existing_set:
        return False

    overlap = len(new_set & existing_set)
    max_possible = min(len(new_set), len(existing_set))
    overlap_ratio = overlap / max_possible if max_possible > 0 else 0

    return overlap_ratio >= entity_threshold


def merge_events(target: "Event", source: "Event") -> "Event":
    """Merge source event into target: combine sources and entities."""
    # Add source's sources
    existing_urls = {s.get("url") for s in target.sources}
    for s in source.sources:
        if s.get("url") not in existing_urls:
            target.sources.append(s)

    # Merge entities (deduplicated)
    existing_entities = {e.lower() for e in target.entities}
    for e in source.entities:
        if e.lower() not in existing_entities:
            target.entities.append(e)
            existing_entities.add(e.lower())

    # Keep higher significance
    target.significance = max(target.significance, source.significance)

    return target


EXTRACT_SYSTEM_PROMPT = (
    "You extract structured event data from news articles. "
    "Given an article and topic context, output JSON with: "
    "date (YYYY-MM-DD), summary (1-2 sentences in the user's language), "
    "entities (key actors/organizations), relation_to_prior (how this connects to recent events), "
    "significance (1-10).\n\n"
    "CRITICAL DATE RULES:\n"
    "- The date field is WHEN THE EVENT HAPPENED, not when it might happen in the future.\n"
    "- Today's date is {current_date}. The event date MUST be on or before today.\n"
    "- If the article discusses future plans or speculation, use the article's publication date.\n"
    "- If you cannot determine the exact date, use the article's publication date.\n"
    "- NEVER output a date after {current_date}."
)


async def extract_event(
    llm: LLMClient,
    item: ContentItem,
    topic: TopicConfig,
    existing_events: list[Event],
    current_date: date | None = None,
) -> Optional[Event]:
    """Extract a structured event from a content item via LLM.

    Args:
        current_date: The processing date (e.g. backtest day). Defaults to today.
            Used to anchor the LLM's date extraction and clamp future dates.
    """
    processing_date = current_date or date.today()

    # Wider event window: last 7 days, up to 30 events
    recent_context = ""
    if existing_events:
        from datetime import timedelta
        cutoff = processing_date - timedelta(days=7)
        recent = [e for e in existing_events if e.date >= cutoff][-30:]
        recent_context = "\n".join(
            f"- [{e.date}] {e.summary}" for e in recent
        )

    # Language and source metadata for context
    article_lang = item.detected_language or item.source_language or "unknown"
    source_meta = ""
    if item.source_affiliation or item.source_country:
        source_meta = f"\nSource affiliation: {item.source_affiliation or 'unknown'}, Country: {item.source_country or 'unknown'}"

    # Article publication date as fallback anchor
    pub_date = ""
    if item.published:
        pub_day = item.published.date() if hasattr(item.published, "date") else item.published
        pub_date = f"\nArticle publication date: {pub_day}"

    user_prompt = (
        f"Topic: {topic.name}\n"
        f"Subtopics: {', '.join(topic.subtopics)}\n\n"
        f"Today's date: {processing_date}\n"
        f"Recent events:\n{recent_context or 'None yet'}\n\n"
        f"Article language: {article_lang}{source_meta}{pub_date}\n"
        f"Write the summary in English.\n\n"
        f"Article title: {item.title}\n"
        f"Article text: {(item.full_text or '')[:3000]}"
    )

    system_prompt = EXTRACT_SYSTEM_PROMPT.format(current_date=processing_date)

    try:
        response = await llm.complete(
            config_key="knowledge_summary",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            json_response=True,
        )
        data = json.loads(response)
        if isinstance(data, list):
            data = data[0] if data else {}

        event_date = date.fromisoformat(data["date"])

        # Hard clamp: never allow future dates
        if event_date > processing_date:
            logger.warning(
                f"Clamped future date {event_date} → {processing_date} "
                f"for article: {item.url}"
            )
            event_date = processing_date

        return Event(
            date=event_date,
            summary=data["summary"],
            sources=[{
                "url": item.url,
                "language": item.detected_language or item.source_language or "en",
                "outlet": item.source_id,
                "affiliation": item.source_affiliation or "",
                "country": item.source_country or "",
            }],
            entities=data.get("entities", []),
            relation_to_prior=data.get("relation_to_prior", ""),
            significance=int(data.get("significance", 5)),
        )
    except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
        logger.warning(f"Failed to extract event from {item.url}: {e}")
        return None
