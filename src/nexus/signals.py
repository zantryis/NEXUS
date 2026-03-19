"""Lightweight async signal bus for cross-module event notification.

Pre-release foundation: emits signals at key points (breaking alerts, odds
updates, pipeline completion). Post-release, handlers will be registered to
trigger reprice/repredict on new signals.
"""

import enum
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Coroutine

logger = logging.getLogger(__name__)


class SignalType(enum.Enum):
    BREAKING_ALERT = "breaking_alert"
    KALSHI_ODDS_UPDATED = "kalshi_odds_updated"
    PIPELINE_COMPLETE = "pipeline_complete"
    REPRICE_COMPLETE = "reprice_complete"
    NEW_EVENTS_INGESTED = "new_events_ingested"


@dataclass
class Signal:
    type: SignalType
    payload: dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


Handler = Callable[[Signal], Coroutine[Any, Any, None]]


class SignalBus:
    """In-process async pub/sub."""

    def __init__(self) -> None:
        self._handlers: dict[SignalType, list[Handler]] = {}
        self._history: list[Signal] = []

    def on(self, signal_type: SignalType, handler: Handler) -> None:
        """Register a handler for a signal type."""
        self._handlers.setdefault(signal_type, []).append(handler)

    async def emit(self, signal: Signal) -> None:
        """Emit a signal, calling all registered handlers."""
        self._history.append(signal)
        if len(self._history) > 100:
            self._history = self._history[-50:]
        for handler in self._handlers.get(signal.type, []):
            try:
                await handler(signal)
            except Exception as e:
                logger.warning("Signal handler error for %s: %s", signal.type, e)


# Module-level singleton
bus = SignalBus()
