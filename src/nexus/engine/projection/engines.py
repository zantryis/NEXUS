"""Projection engine contract plus native and sidecar adapters."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Protocol

from nexus.engine.knowledge.events import Event
from nexus.engine.projection.analytics import detect_converging_threads
from nexus.engine.projection.models import CrossTopicSignal, ProjectionItem, TopicProjection
from nexus.engine.synthesis.knowledge import NarrativeThread
from nexus.llm.client import LLMClient

logger = logging.getLogger(__name__)

ANALYST_SYSTEM_PROMPT = (
    "You are a forward-looking analyst. Use only the supplied signals to produce 2-3 "
    "specific short-horizon projections. Every projection must cite the exact signals "
    "that support it, include a signpost that would confirm or disconfirm it, and keep "
    "confidence categorical: low, medium, or high.\n\n"
    "Return JSON only in the form "
    "{\"summary\": \"...\", \"items\": [{\"claim\": \"...\", \"confidence\": \"medium\", "
    "\"horizon_days\": 7, \"signpost\": \"...\", \"signals_cited\": [\"...\"]}]}"
)

CRITIC_SYSTEM_PROMPT = (
    "You are a projection critic. Tighten speculative language, demote unsupported "
    "confidence, and ensure every claim is grounded in the evidence provided. "
    "Return the same JSON schema."
)


@dataclass
class ProjectionEngineInput:
    """Normalized topic state passed to any projection engine."""

    topic_slug: str
    topic_name: str
    run_date: date
    threads: list[NarrativeThread] = field(default_factory=list)
    recent_events: list[Event] = field(default_factory=list)
    cross_topic_signals: list[CrossTopicSignal] = field(default_factory=list)
    trajectory_threads: list[NarrativeThread] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class ProjectionEngine(Protocol):
    """Common adapter contract for projection engines."""

    engine_name: str

    async def project(
        self,
        llm: LLMClient | None,
        payload: ProjectionEngineInput,
        *,
        critic_pass: bool = True,
        max_items: int = 3,
    ) -> TopicProjection:
        """Produce a topic-level projection artifact."""


def _get_convergence_signals(payload: ProjectionEngineInput) -> list[dict]:
    """Compute convergence signals from payload threads."""
    threads = payload.trajectory_threads or payload.threads
    if not threads:
        return []
    return detect_converging_threads(threads)


def _fallback_projection(payload: ProjectionEngineInput, engine_name: str, max_items: int) -> TopicProjection:
    """Deterministic fallback when LLM passes are disabled or fail."""
    items: list[ProjectionItem] = []
    sorted_threads = sorted(
        payload.trajectory_threads or payload.threads,
        key=lambda thread: (thread.momentum_score or 0.0, thread.significance),
        reverse=True,
    )
    for thread in sorted_threads[:max_items]:
        label = thread.trajectory_label or "steady"
        horizon_days = 3 if label == "about_to_break" else 7
        confidence = "high" if label in {"about_to_break", "accelerating"} else "medium"
        claim = f"{thread.headline} is likely to stay {label.replace('_', ' ')} over the next {horizon_days} days."
        signpost = (
            thread.events[-1].summary if thread.events else f"New reporting tied to {thread.headline}"
        )
        signals = [
            f"trajectory:{label}",
            f"momentum:{round(thread.momentum_score or 0.0, 2)}",
        ]
        if thread.events:
            signals.append(f"latest-event:{thread.events[-1].summary[:100]}")
        items.append(ProjectionItem(
            claim=claim,
            confidence=confidence,
            horizon_days=horizon_days,
            signpost=signpost,
            signals_cited=signals,
            evidence_event_ids=[event.event_id for event in thread.events if event.event_id],
            evidence_thread_ids=[thread.thread_id] if thread.thread_id else [],
            review_after=payload.run_date + timedelta(days=horizon_days),
        ))

    if payload.cross_topic_signals and len(items) < max_items:
        signal = payload.cross_topic_signals[0]
        items.append(ProjectionItem(
            claim=(
                f"Cross-topic activity around {signal.shared_entity} may pull "
                f"{payload.topic_name} toward developments in {signal.related_topic_slug.replace('-', ' ')}."
            ),
            confidence="medium",
            horizon_days=7,
            signpost=signal.note,
            signals_cited=[f"cross-topic:{signal.shared_entity}", signal.note],
            evidence_event_ids=signal.event_ids + signal.related_event_ids,
            evidence_thread_ids=[],
            review_after=payload.run_date + timedelta(days=7),
        ))

    convergence_signals = _get_convergence_signals(payload)

    status = "ready" if items else "insufficient_history"
    return TopicProjection(
        topic_slug=payload.topic_slug,
        topic_name=payload.topic_name,
        engine=engine_name,
        generated_for=payload.run_date,
        status=status,
        summary=f"{payload.topic_name} forward look generated by {engine_name}.",
        items=items[:max_items],
        cross_topic_signals=payload.cross_topic_signals[:5],
        convergence_signals=convergence_signals,
        metadata={"fallback": True},
    )


def _build_prompt_context(payload: ProjectionEngineInput, max_items: int) -> str:
    lines = [
        f"Topic: {payload.topic_name}",
        f"Run date: {payload.run_date.isoformat()}",
        "",
        "Threads:",
    ]
    threads = payload.trajectory_threads or payload.threads
    for thread in threads[:6]:
        lines.append(
            f"- {thread.headline} | status={thread.status or 'unknown'} | "
            f"trajectory={thread.trajectory_label or 'steady'} | "
            f"momentum={round(thread.momentum_score or 0.0, 2)} | "
            f"significance={thread.significance}"
        )
        if thread.events:
            lines.append(f"  latest_event: {thread.events[-1].summary}")

    if payload.cross_topic_signals:
        lines.append("")
        lines.append("Cross-topic signals:")
        for signal in payload.cross_topic_signals[:5]:
            lines.append(
                f"- {signal.shared_entity}: {payload.topic_slug} <-> {signal.related_topic_slug} "
                f"on {signal.observed_at.isoformat()} ({signal.note})"
            )

    convergence_signals = _get_convergence_signals(payload)
    if convergence_signals:
        lines.append("")
        lines.append("Converging threads (shared entities + hot trajectories):")
        for cs in convergence_signals[:5]:
            entities_str = ", ".join(cs["shared_entities"])
            lines.append(
                f"- Threads {cs['threads']} share [{entities_str}] "
                f"(pattern={cs['pattern']}, confidence={cs['confidence']})"
            )

    lines.append("")
    lines.append(f"Return at most {max_items} items.")
    return "\n".join(lines)


class NativeProjectionEngine:
    """Default runtime engine using Nexus-native context plus optional LLM passes."""

    engine_name = "native"

    async def project(
        self,
        llm: LLMClient | None,
        payload: ProjectionEngineInput,
        *,
        critic_pass: bool = True,
        max_items: int = 3,
    ) -> TopicProjection:
        if llm is None:
            return _fallback_projection(payload, self.engine_name, max_items)

        user_prompt = _build_prompt_context(payload, max_items)
        try:
            analyst_raw = await llm.complete(
                config_key="knowledge_summary",
                system_prompt=ANALYST_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                json_response=True,
            )
            candidate = json.loads(analyst_raw)
            if critic_pass:
                critic_raw = await llm.complete(
                    config_key="knowledge_summary",
                    system_prompt=CRITIC_SYSTEM_PROMPT,
                    user_prompt=json.dumps(candidate, indent=2),
                    json_response=True,
                )
                candidate = json.loads(critic_raw)

            items = []
            for item in candidate.get("items", [])[:max_items]:
                horizon_days = int(item.get("horizon_days", 7))
                if horizon_days not in (3, 7, 14):
                    horizon_days = 7
                confidence = str(item.get("confidence", "medium")).lower()
                if confidence not in {"low", "medium", "high"}:
                    confidence = "medium"
                items.append(ProjectionItem(
                    claim=item["claim"],
                    confidence=confidence,
                    horizon_days=horizon_days,
                    signpost=item["signpost"],
                    signals_cited=item.get("signals_cited", []),
                    evidence_event_ids=_collect_evidence_event_ids(payload),
                    evidence_thread_ids=[
                        thread.thread_id for thread in (payload.trajectory_threads or payload.threads)
                        if thread.thread_id
                    ][:4],
                    review_after=payload.run_date + timedelta(days=horizon_days),
                ))

            if not items:
                return _fallback_projection(payload, self.engine_name, max_items)

            return TopicProjection(
                topic_slug=payload.topic_slug,
                topic_name=payload.topic_name,
                engine=self.engine_name,
                generated_for=payload.run_date,
                status="ready",
                summary=candidate.get("summary", ""),
                items=items,
                cross_topic_signals=payload.cross_topic_signals[:5],
                convergence_signals=_get_convergence_signals(payload),
                metadata={"fallback": False},
            )
        except Exception as exc:
            logger.warning("Native projection generation failed for %s: %s", payload.topic_slug, exc)
            return _fallback_projection(payload, self.engine_name, max_items)


class GraphitiProjectionEngine:
    """Optional sidecar adapter. Falls back to deterministic output in-core."""

    engine_name = "graphiti"

    async def project(
        self,
        llm: LLMClient | None,
        payload: ProjectionEngineInput,
        *,
        critic_pass: bool = True,
        max_items: int = 3,
    ) -> TopicProjection:
        projection = _fallback_projection(payload, self.engine_name, max_items)
        projection.metadata["sidecar_mode"] = "graphiti_kuzu_stub"
        return projection


class MirofishSpikeProjectionEngine:
    """Optional AGPL sidecar adapter. Kept isolated from the runtime path."""

    engine_name = "mirofish-spike"

    async def project(
        self,
        llm: LLMClient | None,
        payload: ProjectionEngineInput,
        *,
        critic_pass: bool = True,
        max_items: int = 3,
    ) -> TopicProjection:
        projection = _fallback_projection(payload, self.engine_name, max_items)
        projection.metadata["sidecar_mode"] = "mirofish_stub"
        return projection


def _collect_evidence_event_ids(payload: ProjectionEngineInput) -> list[int]:
    evidence_ids: list[int] = []
    for thread in payload.trajectory_threads or payload.threads:
        for event in thread.events:
            if event.event_id and event.event_id not in evidence_ids:
                evidence_ids.append(event.event_id)
    return evidence_ids[:8]


def get_projection_engine(engine_name: str) -> ProjectionEngine:
    """Resolve a projection engine by name."""
    normalized = engine_name.strip().lower()
    if normalized == "native":
        return NativeProjectionEngine()
    if normalized == "graphiti":
        return GraphitiProjectionEngine()
    if normalized == "mirofish-spike":
        return MirofishSpikeProjectionEngine()
    raise ValueError(f"Unknown projection engine: {engine_name}")


def export_engine_payload(base_dir: Path, engine_name: str, payload: ProjectionEngineInput) -> Path:
    """Persist a read-only payload snapshot for sidecar experiments."""
    out_dir = base_dir / "frameworks" / engine_name.replace("-", "_")
    out_dir.mkdir(parents=True, exist_ok=True)
    export_path = out_dir / f"{payload.topic_slug}-{payload.run_date.isoformat()}.json"
    export_path.write_text(json.dumps({
        "topic_slug": payload.topic_slug,
        "topic_name": payload.topic_name,
        "run_date": payload.run_date.isoformat(),
        "threads": [thread.model_dump(mode="json") for thread in payload.threads],
        "recent_events": [event.model_dump(mode="json") for event in payload.recent_events],
        "cross_topic_signals": [signal.model_dump(mode="json") for signal in payload.cross_topic_signals],
        "metadata": payload.metadata,
    }, indent=2))
    return export_path
