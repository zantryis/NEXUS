"""Pydantic models for casefile configuration and bundles."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field


ScopeKind = Literal["topic", "case"]
ReadingLevel = Literal["short", "standard", "deep"]
SourceKind = Literal["article", "report", "archive", "feed"]
SourceRole = Literal["primary", "secondary"]
SourceClass = Literal["official", "investigation", "media", "analysis"]
EvidenceStance = Literal["supports", "refutes", "context"]
QualityLabel = Literal["high", "medium", "low"]
AssessmentMode = Literal["posterior", "forecast"]


class ScopeRef(BaseModel):
    kind: ScopeKind
    slug: str


class CaseTimeBounds(BaseModel):
    start: str | None = None
    end: str | None = None


class CaseBuildDefaults(BaseModel):
    max_queries: int = Field(default=10, ge=1, le=30)
    max_search_results_per_query: int = Field(default=4, ge=1, le=10)
    max_documents: int = Field(default=12, ge=3, le=40)
    max_text_chars: int = Field(default=12000, ge=1000, le=50000)
    max_evidence: int = Field(default=36, ge=6, le=100)


class CaseConfig(BaseModel):
    slug: str
    title: str
    question: str
    time_bounds: CaseTimeBounds = Field(default_factory=CaseTimeBounds)
    hypothesis_seeds: list[str] = Field(default_factory=list)
    reading_levels: list[ReadingLevel] = Field(default_factory=lambda: ["short", "standard", "deep"])
    build: CaseBuildDefaults = Field(default_factory=CaseBuildDefaults)


class SeedSource(BaseModel):
    id: str
    label: str
    url: str
    kind: SourceKind
    role: SourceRole
    source_class: SourceClass
    priority: int = Field(default=5, ge=1, le=10)
    notes: str | None = None


class SeedManifest(BaseModel):
    sources: list[SeedSource] = Field(default_factory=list)


class CandidateDocument(BaseModel):
    id: str
    label: str
    url: str
    kind: SourceKind
    role: SourceRole
    source_class: SourceClass
    priority: int = Field(default=5, ge=1, le=10)
    notes: str | None = None
    discovered_via: str | None = None
    search_snippet: str | None = None
    title_hint: str | None = None


class AcquisitionQuery(BaseModel):
    query: str
    source_class: SourceClass
    rationale: str


class FetchedDocument(BaseModel):
    id: str
    label: str
    url: str
    canonical_url: str
    kind: SourceKind
    role: SourceRole
    source_class: SourceClass
    priority: int = Field(default=5, ge=1, le=10)
    notes: str | None = None
    discovered_via: str | None = None
    title: str
    published_at: str | None = None
    raw_text: str
    excerpt: str | None = None
    ingestion_status: str = "ok"
    ingestion_error: str | None = None


class AcquisitionResult(BaseModel):
    queries: list[AcquisitionQuery] = Field(default_factory=list)
    candidates: list[CandidateDocument] = Field(default_factory=list)
    documents: list[FetchedDocument] = Field(default_factory=list)


class ExtractedEntity(BaseModel):
    name: str
    type: str = "other"
    description: str | None = None


class ExtractedClaim(BaseModel):
    claim: str
    stance: EvidenceStance = "context"
    related_hypotheses: list[str] = Field(default_factory=list)
    importance: QualityLabel = "medium"
    excerpt: str | None = None
    why_it_matters: str | None = None


class ExtractedDocument(BaseModel):
    id: str
    title: str
    url: str
    canonical_url: str
    kind: SourceKind
    role: SourceRole
    source_class: SourceClass
    source_label: str
    priority: int = Field(default=5, ge=1, le=10)
    notes: str | None = None
    discovered_via: str | None = None
    published_at: str | None = None
    quality_label: QualityLabel = "medium"
    summary: str
    time_anchors: list[str] = Field(default_factory=list)
    entities: list[ExtractedEntity] = Field(default_factory=list)
    claims: list[ExtractedClaim] = Field(default_factory=list)
    excerpt: str | None = None
    ingestion_status: str = "ok"
    ingestion_error: str | None = None


class CaseMetadata(BaseModel):
    slug: str
    title: str
    question: str
    generated_at: str
    last_updated: str
    time_bounds: CaseTimeBounds = Field(default_factory=CaseTimeBounds)
    reading_levels: list[ReadingLevel] = Field(default_factory=lambda: ["short", "standard", "deep"])
    source_count: int = 0
    document_count: int = 0
    evidence_count: int = 0
    build_defaults: CaseBuildDefaults = Field(default_factory=CaseBuildDefaults)
    presentable: bool = False


class CaseOverview(BaseModel):
    best_current_account: str
    confidence_label: str
    reading_levels: dict[ReadingLevel, str]
    key_judgments: list[str] = Field(default_factory=list)
    recent_updates: list[str] = Field(default_factory=list)


class CaseHypothesis(BaseModel):
    id: str
    title: str
    summary: str
    confidence_label: str
    evidence_for: list[str] = Field(default_factory=list)
    evidence_against: list[str] = Field(default_factory=list)
    unresolved_gaps: list[str] = Field(default_factory=list)
    what_would_change_my_mind: list[str] = Field(default_factory=list)


class TimelineEntry(BaseModel):
    id: str
    label: str
    date: str
    description: str
    evidence_ids: list[str] = Field(default_factory=list)


class EvidenceItem(BaseModel):
    id: str
    claim: str
    stance: EvidenceStance
    quality_label: QualityLabel
    summary: str
    document_id: str
    document_title: str
    document_url: str
    source_label: str
    source_class: SourceClass
    related_hypotheses: list[str] = Field(default_factory=list)
    excerpt: str | None = None
    time_anchors: list[str] = Field(default_factory=list)


class CaseEntity(BaseModel):
    id: str
    name: str
    type: str = "other"
    description: str | None = None


class CaseRelation(BaseModel):
    source_entity_id: str
    target_entity_id: str
    relationship: str
    evidence_ids: list[str] = Field(default_factory=list)


class CaseThread(BaseModel):
    thread_id: int
    slug: str
    headline: str
    status: str = "emerging"
    significance: int = 5
    key_entities: list[str] = Field(default_factory=list)
    event_count: int = 0
    convergence_count: int = 0
    divergence_count: int = 0
    trajectory_label: str | None = None
    momentum_score: float | None = None
    velocity_7d: float | None = None
    acceleration_7d: float | None = None
    snapshot_count: int | None = None
    created_at: str | None = None
    updated_at: str | None = None


class CaseDivergence(BaseModel):
    thread_id: int
    thread_slug: str
    thread_headline: str
    shared_event: str
    source_a: str
    framing_a: str
    source_b: str
    framing_b: str


class CaseAssessment(BaseModel):
    id: str
    target_hypothesis_id: str
    mode: AssessmentMode = "posterior"
    question: str
    probability: float = Field(default=0.5, ge=0.0, le=1.0)
    confidence: str = "medium"
    rationale: str
    counterarguments: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    evidence_thread_ids: list[int] = Field(default_factory=list)
    signposts: list[str] = Field(default_factory=list)


class CaseGraphSummary(BaseModel):
    node_count: int = 0
    edge_count: int = 0
    top_entities: list[str] = Field(default_factory=list)


class ReviewScores(BaseModel):
    coherence: int = Field(default=1, ge=1, le=5)
    evidentiary_grounding: int = Field(default=1, ge=1, le=5)
    source_diversity: int = Field(default=1, ge=1, le=5)
    separation_of_hypotheses: int = Field(default=1, ge=1, le=5)
    uncertainty_honesty: int = Field(default=1, ge=1, le=5)
    non_obvious_synthesis: int = Field(default=1, ge=1, le=5)


class CaseReview(BaseModel):
    scores: ReviewScores = Field(default_factory=ReviewScores)
    flags: list[str] = Field(default_factory=list)
    blocker_reasons: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    verdict: str = "draft"
    presentable: bool = False


class CasefileBundle(BaseModel):
    metadata: CaseMetadata
    overview: CaseOverview
    hypotheses: list[CaseHypothesis] = Field(default_factory=list)
    timeline: list[TimelineEntry] = Field(default_factory=list)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    documents: list[ExtractedDocument] = Field(default_factory=list)
    entities: list[CaseEntity] = Field(default_factory=list)
    relations: list[CaseRelation] = Field(default_factory=list)
    threads: list[CaseThread] = Field(default_factory=list)
    divergence: list[CaseDivergence] = Field(default_factory=list)
    assessments: list[CaseAssessment] = Field(default_factory=list)
    graph: CaseGraphSummary = Field(default_factory=CaseGraphSummary)
    recent_changes: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    review: CaseReview = Field(default_factory=CaseReview)


class LoadedCaseDefinition(BaseModel):
    path: str
    case: CaseConfig
    seeds: SeedManifest | None = None
    bundle: CasefileBundle | None = None


class BuildStatus(BaseModel):
    slug: str
    status: Literal["idle", "running", "completed", "failed"] = "idle"
    label: str = "Idle"
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).replace(microsecond=0).isoformat())
    error: str | None = None
    presentable: bool | None = None
