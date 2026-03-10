"""Knowledge layer compression — weekly and monthly rollup summaries."""

import logging
from collections import defaultdict
from datetime import date
from pathlib import Path

import yaml
from pydantic import BaseModel

from nexus.engine.knowledge.events import Event
from nexus.llm.client import LLMClient

logger = logging.getLogger(__name__)


class Summary(BaseModel):
    period_start: date
    period_end: date
    text: str
    event_count: int


def group_events_by_week(events: list[Event]) -> dict[tuple[int, int], list[Event]]:
    """Group events by ISO week number. Returns {(year, week): [events]}."""
    groups: dict[tuple[int, int], list[Event]] = defaultdict(list)
    for event in events:
        iso = event.date.isocalendar()
        groups[(iso.year, iso.week)].append(event)
    return dict(groups)


async def compress_to_weekly(
    llm: LLMClient, events: list[Event], topic_name: str
) -> list[Summary]:
    """Compress a list of events into weekly summaries via LLM."""
    groups = group_events_by_week(events)
    summaries = []

    for (_year, _week), week_events in sorted(groups.items()):
        dates = [e.date for e in week_events]
        event_text = "\n".join(f"- [{e.date}] {e.summary}" for e in week_events)

        response = await llm.complete(
            config_key="knowledge_summary",
            system_prompt=(
                "Summarize these events into a concise weekly narrative paragraph. "
                "Preserve key facts, entities, and causal connections."
            ),
            user_prompt=f"Topic: {topic_name}\n\nEvents:\n{event_text}",
        )

        summaries.append(Summary(
            period_start=min(dates),
            period_end=max(dates),
            text=response,
            event_count=len(week_events),
        ))

    return summaries


def load_summaries(path: Path) -> list[Summary]:
    if not path.exists():
        return []
    raw = yaml.safe_load(path.read_text())
    if not raw:
        return []
    return [Summary(**s) for s in raw]


def save_summaries(path: Path, summaries: list[Summary]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [s.model_dump(mode="json") for s in summaries]
    path.write_text(yaml.dump(data, default_flow_style=False, allow_unicode=True))
