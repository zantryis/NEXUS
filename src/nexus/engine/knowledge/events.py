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


EXTRACT_SYSTEM_PROMPT = (
    "You extract structured event data from news articles. "
    "Given an article and topic context, output JSON with: "
    "date (YYYY-MM-DD), summary (1-2 sentences in the user's language), "
    "entities (key actors/organizations), relation_to_prior (how this connects to recent events), "
    "significance (1-10)."
)


async def extract_event(
    llm: LLMClient,
    item: ContentItem,
    topic: TopicConfig,
    existing_events: list[Event],
) -> Optional[Event]:
    """Extract a structured event from a content item via LLM."""
    recent_context = ""
    if existing_events:
        recent = existing_events[-10:]
        recent_context = "\n".join(
            f"- [{e.date}] {e.summary}" for e in recent
        )

    user_prompt = (
        f"Topic: {topic.name}\n"
        f"Subtopics: {', '.join(topic.subtopics)}\n\n"
        f"Recent events:\n{recent_context or 'None yet'}\n\n"
        f"Article title: {item.title}\n"
        f"Article text: {(item.full_text or '')[:3000]}"
    )

    try:
        response = await llm.complete(
            config_key="knowledge_summary",
            system_prompt=EXTRACT_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            json_response=True,
        )
        data = json.loads(response)
        return Event(
            date=date.fromisoformat(data["date"]),
            summary=data["summary"],
            sources=[{
                "url": item.url,
                "language": item.language or "en",
                "outlet": item.source_id,
            }],
            entities=data.get("entities", []),
            relation_to_prior=data.get("relation_to_prior", ""),
            significance=int(data.get("significance", 5)),
        )
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        logger.warning(f"Failed to extract event from {item.url}: {e}")
        return None
