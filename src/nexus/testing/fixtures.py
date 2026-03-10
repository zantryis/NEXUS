"""Fixture capture and replay for multi-day pipeline simulation.

Capture mode: run real pipeline, save all intermediate data to fixtures dir.
Replay mode: load captured data and run pipeline without network/LLM calls.
"""

import json
import logging
from datetime import date
from pathlib import Path
from typing import Optional

import yaml

from nexus.engine.sources.polling import ContentItem
from nexus.engine.knowledge.events import Event

logger = logging.getLogger(__name__)


class FixtureCapture:
    """Captures intermediate pipeline data for replay testing."""

    def __init__(self, fixture_dir: Path, topic_slug: str, day_label: str = ""):
        self.dir = fixture_dir / topic_slug / (day_label or date.today().isoformat())
        self.dir.mkdir(parents=True, exist_ok=True)

    def save_polled(self, items: list[ContentItem]) -> None:
        data = [item.model_dump(mode="json") for item in items]
        (self.dir / "polled.json").write_text(json.dumps(data, indent=2, default=str))

    def save_ingested(self, items: list[ContentItem]) -> None:
        data = [item.model_dump(mode="json") for item in items]
        (self.dir / "ingested.json").write_text(json.dumps(data, indent=2, default=str))

    def save_filtered(self, items: list[ContentItem]) -> None:
        data = [item.model_dump(mode="json") for item in items]
        (self.dir / "filtered.json").write_text(json.dumps(data, indent=2, default=str))

    def save_events(self, events: list[Event]) -> None:
        data = [e.model_dump(mode="json") for e in events]
        (self.dir / "events.json").write_text(json.dumps(data, indent=2, default=str))

    def save_llm_responses(self, responses: list[dict]) -> None:
        (self.dir / "llm_responses.json").write_text(json.dumps(responses, indent=2, default=str))


class FixtureReplay:
    """Loads captured fixture data for replay testing."""

    def __init__(self, fixture_dir: Path, topic_slug: str, day_label: str):
        self.dir = fixture_dir / topic_slug / day_label
        if not self.dir.exists():
            raise FileNotFoundError(f"Fixture not found: {self.dir}")

    def load_polled(self) -> list[ContentItem]:
        path = self.dir / "polled.json"
        if not path.exists():
            return []
        data = json.loads(path.read_text())
        return [ContentItem(**item) for item in data]

    def load_ingested(self) -> list[ContentItem]:
        path = self.dir / "ingested.json"
        if not path.exists():
            return []
        data = json.loads(path.read_text())
        return [ContentItem(**item) for item in data]

    def load_filtered(self) -> list[ContentItem]:
        path = self.dir / "filtered.json"
        if not path.exists():
            return []
        data = json.loads(path.read_text())
        return [ContentItem(**item) for item in data]

    def load_events(self) -> list[Event]:
        path = self.dir / "events.json"
        if not path.exists():
            return []
        data = json.loads(path.read_text())
        return [Event(**e) for e in data]


def list_captured_days(fixture_dir: Path, topic_slug: str) -> list[str]:
    """List all captured day labels for a topic."""
    topic_dir = fixture_dir / topic_slug
    if not topic_dir.exists():
        return []
    return sorted(d.name for d in topic_dir.iterdir() if d.is_dir())
