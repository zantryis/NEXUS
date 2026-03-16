"""Quantified forecast engines and projection rendering adapters."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Protocol

from nexus.engine.knowledge.events import Event
from nexus.engine.projection.models import (
    CrossTopicSignal,
    ForecastQuestion,
    ForecastRun,
    GraphSnapshot,
    TopicProjection,
    confidence_from_probability,
)
from nexus.engine.synthesis.knowledge import NarrativeThread
from nexus.llm.client import LLMClient

logger = logging.getLogger(__name__)


@dataclass
class ForecastEngineInput:
    """Normalized topic state passed into forecast engines."""

    topic_slug: str
    topic_name: str
    run_date: date
    threads: list[NarrativeThread] = field(default_factory=list)
    recent_events: list[Event] = field(default_factory=list)
    cross_topic_signals: list[CrossTopicSignal] = field(default_factory=list)
    graph_snapshot: GraphSnapshot | None = None
    metadata: dict = field(default_factory=dict)


class ForecastEngine(Protocol):
    """Common contract for quantified forecast engines."""

    engine_name: str

    async def generate(
        self,
        llm: LLMClient | None,
        payload: ForecastEngineInput,
        *,
        critic_pass: bool = True,
        max_questions: int = 4,
        calibration_data: list[dict] | None = None,
    ) -> ForecastRun:
        """Produce a structured forecast run."""


def _clip_probability(value: float) -> float:
    return round(max(0.05, min(0.95, value)), 3)


def _empirical_adjusted_base_rate(
    hardcoded_rate: float,
    historical_rows: list[dict],
    *,
    min_samples: int = 10,
) -> float:
    """Bayesian shrinkage: blend hardcoded prior with empirical hit rate.

    Below min_samples, returns prior unchanged.
    Otherwise, weight = min(n / 50, 0.7) and blended = weight * empirical + (1 - weight) * prior.
    """
    n = len(historical_rows)
    if n < min_samples:
        return hardcoded_rate
    empirical = sum(1 for r in historical_rows if r["resolved_bool"]) / n
    weight = min(n / 50.0, 0.7)
    return round(weight * empirical + (1.0 - weight) * hardcoded_rate, 4)


def get_forecast_engine(engine_name: str) -> ForecastEngine:
    """Resolve a quantified forecast engine by name."""
    normalized = engine_name.strip().lower()
    if normalized in {"actor", "native"}:
        from nexus.engine.projection.actor_engine import ActorForecastEngine
        return ActorForecastEngine()
    raise ValueError(f"Unknown forecast engine: {engine_name}")


def projection_from_forecast_run(run: ForecastRun, cross_topic_signals: list[CrossTopicSignal]) -> TopicProjection:
    """Render the user-facing projection artifact from a structured forecast run."""
    return TopicProjection(
        topic_slug=run.topic_slug,
        topic_name=run.topic_name,
        engine=run.engine,
        generated_for=run.generated_for,
        status="ready" if run.questions else "insufficient_history",
        summary=run.summary,
        items=[
            {
                "claim": question.question,
                "confidence": confidence_from_probability(question.probability),
                "horizon_days": question.horizon_days,
                "signpost": question.signpost,
                "signals_cited": question.signals_cited,
                "evidence_event_ids": question.evidence_event_ids,
                "evidence_thread_ids": question.evidence_thread_ids,
                "review_after": question.resolution_date,
                "external_ref": question.external_ref,
            }
            for question in run.questions
        ],
        cross_topic_signals=cross_topic_signals[:5],
        metadata={"forecast_engine": run.engine, "forecast_question_count": len(run.questions), **run.metadata},
    )
