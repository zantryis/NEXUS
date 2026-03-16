"""Historical state helpers for leakage-safe replay and benchmarking."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from nexus.engine.knowledge.events import Event
from nexus.engine.projection.models import CrossTopicSignal
from nexus.engine.synthesis.knowledge import TopicSynthesis

SIGNAL_RICH_KEYWORDS = {
    "ceasefire",
    "command",
    "contract",
    "court",
    "department",
    "filing",
    "guidance",
    "lawsuit",
    "launch",
    "ministry",
    "naval",
    "partnership",
    "permit",
    "policy",
    "price",
    "regulatory",
    "sanction",
    "statement",
    "strike",
}


def is_signal_rich_events(events: list[Event]) -> bool:
    """Return whether a recent event set looks useful for hard-resolution benchmarking."""
    haystack = " ".join(event.summary.lower() for event in events)
    return any(keyword in haystack for keyword in SIGNAL_RICH_KEYWORDS)


@dataclass
class HistoricalTopicState:
    """Topic state frozen at a historical cutoff."""

    topic_slug: str
    topic_name: str
    cutoff: date
    synthesis: TopicSynthesis
    recent_events: list[Event] = field(default_factory=list)
    cross_topic_signals: list[CrossTopicSignal] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
