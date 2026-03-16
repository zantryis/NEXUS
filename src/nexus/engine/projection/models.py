"""Projection data models shared by the native runtime and sidecar experiments."""

import hashlib
import json
from datetime import date
from typing import Literal

from pydantic import BaseModel, Field


ConfidenceLevel = Literal["low", "medium", "high"]
TrajectoryLabel = Literal["accelerating", "steady", "decelerating", "about_to_break"]
ProjectionStatus = Literal["ready", "insufficient_history", "unavailable"]
ForecastType = Literal["binary", "directional"]
ForecastQuestionStatus = Literal["open", "resolved", "unresolved", "invalid"]
ForecastResolutionStatus = Literal["pending", "resolved", "unresolved", "invalid"]


class ThreadSnapshot(BaseModel):
    """Daily state snapshot for a persistent thread."""

    thread_id: int
    snapshot_date: date
    status: str
    significance: int
    event_count: int = 0
    latest_event_date: date | None = None
    velocity_7d: float = 0.0
    acceleration_7d: float = 0.0
    significance_trend_7d: float = 0.0
    momentum_score: float = 0.0
    trajectory_label: TrajectoryLabel = "steady"


class CausalLink(BaseModel):
    """Structured relation between two persisted events."""

    source_event_id: int
    target_event_id: int
    relation_type: str = "follow_on"
    evidence_text: str = ""
    strength: float = 0.5


class CrossTopicSignal(BaseModel):
    """Shared signal linking activity across topics."""

    signal_id: int | None = None
    topic_slug: str
    related_topic_slug: str
    shared_entity: str
    signal_type: str = "entity_bridge"
    observed_at: date
    event_ids: list[int] = Field(default_factory=list)
    related_event_ids: list[int] = Field(default_factory=list)
    note: str = ""


class ProjectionItem(BaseModel):
    """Single forward-looking claim with explicit evidence."""

    claim: str
    confidence: ConfidenceLevel = "medium"
    horizon_days: Literal[3, 7, 14] = 7
    signpost: str
    signals_cited: list[str] = Field(default_factory=list)
    evidence_event_ids: list[int] = Field(default_factory=list)
    evidence_thread_ids: list[int] = Field(default_factory=list)
    review_after: date
    external_ref: str | None = None


class TopicProjection(BaseModel):
    """Topic-level projection artifact."""

    topic_slug: str
    topic_name: str
    engine: str = "native"
    generated_for: date
    status: ProjectionStatus = "ready"
    summary: str = ""
    items: list[ProjectionItem] = Field(default_factory=list)
    cross_topic_signals: list[CrossTopicSignal] = Field(default_factory=list)
    convergence_signals: list[dict] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)


class ProjectionOutcome(BaseModel):
    """Outcome review for a stored projection item."""

    projection_item_id: int
    outcome_status: Literal["pending", "hit", "miss", "mixed"] = "pending"
    score: float | None = None
    notes: str = ""
    reviewed_at: date | None = None
    external_ref: str | None = None


class ForecastQuestion(BaseModel):
    """Structured, replayable forecast unit."""

    question_id: int | None = None
    forecast_key: str | None = None
    question: str
    forecast_type: ForecastType = "binary"
    target_variable: str
    target_metadata: dict = Field(default_factory=dict)
    probability: float = Field(default=0.5, ge=0.05, le=0.95)
    base_rate: float = Field(default=0.5, ge=0.0, le=1.0)
    resolution_criteria: str
    resolution_date: date
    horizon_days: Literal[3, 7, 14] = 7
    signpost: str
    expected_direction: str | None = None
    signals_cited: list[str] = Field(default_factory=list)
    evidence_event_ids: list[int] = Field(default_factory=list)
    evidence_thread_ids: list[int] = Field(default_factory=list)
    cross_topic_signal_ids: list[int] = Field(default_factory=list)
    status: ForecastQuestionStatus = "open"
    external_ref: str | None = None


class ForecastScenario(BaseModel):
    """Optional grouped scenario for future expansion."""

    scenario_id: int | None = None
    scenario_key: str
    label: str
    probability: float = Field(default=0.5, ge=0.0, le=1.0)
    description: str = ""
    signposts: list[str] = Field(default_factory=list)
    status: str = "open"


class ForecastRun(BaseModel):
    """Topic-level structured forecast run."""

    run_id: int | None = None
    topic_slug: str
    topic_name: str
    engine: str = "native"
    generated_for: date
    summary: str = ""
    questions: list[ForecastQuestion] = Field(default_factory=list)
    scenarios: list[ForecastScenario] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)


class ForecastResolution(BaseModel):
    """Stored outcome for a forecast question."""

    forecast_question_id: int
    outcome_status: ForecastResolutionStatus = "pending"
    resolved_bool: bool | None = None
    realized_direction: str | None = None
    actual_value: float | None = None
    brier_score: float | None = None
    log_loss: float | None = None
    notes: str = ""
    resolved_at: date | None = None
    external_ref: str | None = None


class GraphSnapshot(BaseModel):
    """Materialized graph snapshot for replay-safe retrieval experiments."""

    topic_slug: str
    generated_for: date
    nodes: list[dict] = Field(default_factory=list)
    edges: list[dict] = Field(default_factory=list)
    metrics: dict = Field(default_factory=dict)


class GraphExportBundle(BaseModel):
    """Versioned sidecar export bundle built from strict replay state."""

    schema_version: int = 1
    topic_slug: str
    topic_name: str
    cutoff: date
    threads: list[dict] = Field(default_factory=list)
    events: list[dict] = Field(default_factory=list)
    causal_links: list[dict] = Field(default_factory=list)
    cross_topic_signals: list[dict] = Field(default_factory=list)
    nodes: list[dict] = Field(default_factory=list)
    edges: list[dict] = Field(default_factory=list)
    evidence_catalog: dict = Field(default_factory=dict)
    metadata: dict = Field(default_factory=dict)


class GraphEvidenceResult(BaseModel):
    """Ranked evidence result returned by a graph sidecar adapter."""

    adapter: str
    status: Literal["ready", "skipped", "error"] = "ready"
    event_ids: list[int] = Field(default_factory=list)
    thread_ids: list[int] = Field(default_factory=list)
    signal_ids: list[int] = Field(default_factory=list)
    paths: list[dict] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)


def confidence_from_probability(probability: float) -> ConfidenceLevel:
    """Map numeric probabilities into user-facing confidence buckets."""
    if probability >= 0.75:
        return "high"
    if probability >= 0.6:
        return "medium"
    return "low"


def build_forecast_key(
    *,
    topic_slug: str,
    generated_for: date,
    target_variable: str,
    target_metadata: dict,
    resolution_date: date,
    horizon_days: int,
    expected_direction: str | None = None,
) -> str:
    """Build a stable forecast key for cross-system benchmark mapping."""
    payload = {
        "topic_slug": topic_slug,
        "generated_for": generated_for.isoformat(),
        "target_variable": target_variable,
        "target_metadata": target_metadata,
        "resolution_date": resolution_date.isoformat(),
        "horizon_days": horizon_days,
        "expected_direction": expected_direction or "",
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha1(serialized.encode("utf-8")).hexdigest()[:16]
    return f"{topic_slug}:{generated_for.isoformat()}:{digest}"
