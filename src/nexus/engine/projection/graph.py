"""Materialized graph snapshots and evidence ranking helpers."""

from __future__ import annotations

import importlib.util
import json
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Protocol

from nexus.engine.knowledge.events import Event
from nexus.engine.projection.models import (
    CrossTopicSignal,
    GraphEvidenceResult,
    GraphExportBundle,
    GraphSnapshot,
)
from nexus.engine.synthesis.knowledge import NarrativeThread


def build_graph_snapshot(
    *,
    topic_slug: str,
    run_date: date,
    threads: list[NarrativeThread],
    cross_topic_signals: list[CrossTopicSignal],
) -> GraphSnapshot:
    """Build a lightweight graph snapshot from current thread/topic state."""
    nodes: list[dict] = [{"id": f"topic:{topic_slug}", "type": "topic", "label": topic_slug}]
    edges: list[dict] = []
    degrees: dict[str, int] = defaultdict(int)

    def add_node(node_id: str, node_type: str, label: str, **attrs) -> None:
        if any(node["id"] == node_id for node in nodes):
            return
        payload = {"id": node_id, "type": node_type, "label": label}
        payload.update(attrs)
        nodes.append(payload)

    def add_edge(source: str, target: str, edge_type: str, **attrs) -> None:
        edges.append({"source": source, "target": target, "type": edge_type, **attrs})
        degrees[source] += 1
        degrees[target] += 1

    for thread in threads:
        if not thread.thread_id:
            continue
        thread_node = f"thread:{thread.thread_id}"
        add_node(
            thread_node,
            "thread",
            thread.headline,
            trajectory=thread.trajectory_label or "steady",
            momentum=round(thread.momentum_score or 0.0, 3),
        )
        add_edge(f"topic:{topic_slug}", thread_node, "topic_thread")

        previous_event_node = None
        for event in thread.events:
            if not event.event_id:
                continue
            event_node = f"event:{event.event_id}"
            add_node(
                event_node,
                "event",
                event.summary[:120],
                event_date=event.date.isoformat(),
                significance=event.significance,
            )
            add_edge(thread_node, event_node, "thread_event")
            if previous_event_node:
                add_edge(previous_event_node, event_node, "causal_follow_on")
            previous_event_node = event_node
            for entity in event.entities:
                entity_key = entity.strip()
                if not entity_key:
                    continue
                entity_node = f"entity:{entity_key.lower()}"
                add_node(entity_node, "entity", entity_key)
                add_edge(event_node, entity_node, "event_entity")

    for signal in cross_topic_signals:
        signal_id = signal.signal_id or abs(hash(
            (signal.topic_slug, signal.related_topic_slug, signal.shared_entity, signal.observed_at.isoformat())
        ))
        signal_node = f"signal:{signal_id}"
        add_node(
            signal_node,
            "cross_topic_signal",
            signal.shared_entity,
            related_topic=signal.related_topic_slug,
            observed_at=signal.observed_at.isoformat(),
        )
        add_edge(f"topic:{topic_slug}", signal_node, "topic_signal")
        entity_node = f"entity:{signal.shared_entity.lower()}"
        add_node(entity_node, "entity", signal.shared_entity)
        add_edge(signal_node, entity_node, "signal_entity")

    top_nodes = sorted(
        [{"node_id": node_id, "degree": degree} for node_id, degree in degrees.items()],
        key=lambda item: item["degree"],
        reverse=True,
    )[:10]
    return GraphSnapshot(
        topic_slug=topic_slug,
        generated_for=run_date,
        nodes=nodes,
        edges=edges,
        metrics={
            "node_count": len(nodes),
            "edge_count": len(edges),
            "top_degree_nodes": top_nodes,
        },
    )


def rank_graph_evidence(
    snapshot: GraphSnapshot,
    *,
    max_event_ids: int = 8,
    max_thread_ids: int = 4,
    max_signal_ids: int = 4,
) -> dict:
    """Rank evidence IDs using simple graph degree heuristics."""
    degree_map = {
        item["node_id"]: item["degree"]
        for item in snapshot.metrics.get("top_degree_nodes", [])
    }
    for edge in snapshot.edges:
        degree_map.setdefault(edge["source"], 0)
        degree_map.setdefault(edge["target"], 0)

    event_nodes = sorted(
        [node for node in snapshot.nodes if node["type"] == "event"],
        key=lambda node: degree_map.get(node["id"], 0),
        reverse=True,
    )
    thread_nodes = sorted(
        [node for node in snapshot.nodes if node["type"] == "thread"],
        key=lambda node: degree_map.get(node["id"], 0),
        reverse=True,
    )
    signal_nodes = sorted(
        [node for node in snapshot.nodes if node["type"] == "cross_topic_signal"],
        key=lambda node: degree_map.get(node["id"], 0),
        reverse=True,
    )

    def _node_suffix(node_id: str) -> int | None:
        try:
            return int(node_id.split(":", 1)[1])
        except (IndexError, ValueError):
            return None

    event_ids = [event_id for node in event_nodes if (event_id := _node_suffix(node["id"])) is not None][:max_event_ids]
    thread_ids = [thread_id for node in thread_nodes if (thread_id := _node_suffix(node["id"])) is not None][:max_thread_ids]
    signal_ids = [signal_id for node in signal_nodes if (signal_id := _node_suffix(node["id"])) is not None][:max_signal_ids]

    return {
        "event_ids": event_ids,
        "thread_ids": thread_ids,
        "signal_ids": signal_ids,
        "signals_cited": [node["label"] for node in signal_nodes[:max_signal_ids]],
    }


def export_graph_snapshot(base_dir: Path, engine_name: str, snapshot: GraphSnapshot) -> Path:
    """Persist a read-only graph snapshot for sidecar experiments."""
    out_dir = base_dir / "frameworks" / engine_name.replace("-", "_") / "graph_snapshots"
    out_dir.mkdir(parents=True, exist_ok=True)
    export_path = out_dir / f"{snapshot.topic_slug}-{snapshot.generated_for.isoformat()}.json"
    export_path.write_text(json.dumps(snapshot.model_dump(mode="json"), indent=2))
    return export_path


def build_graph_export_bundle(
    *,
    topic_slug: str,
    topic_name: str,
    cutoff: date,
    threads: list[NarrativeThread],
    recent_events: list[Event],
    causal_links: list[dict],
    cross_topic_signals: list[CrossTopicSignal],
    schema_version: int = 1,
) -> GraphExportBundle:
    """Build the canonical graph sidecar input bundle from strict replay state."""
    snapshot = build_graph_snapshot(
        topic_slug=topic_slug,
        run_date=cutoff,
        threads=threads,
        cross_topic_signals=cross_topic_signals,
    )

    event_ids_seen: set[int] = set()
    event_rows: list[dict] = []
    for event in recent_events:
        if event.event_id is not None and event.event_id in event_ids_seen:
            continue
        if event.event_id is not None:
            event_ids_seen.add(event.event_id)
        event_rows.append({
            "event_id": event.event_id,
            "date": event.date.isoformat(),
            "summary": event.summary,
            "significance": event.significance,
            "relation_to_prior": event.relation_to_prior,
            "entities": list(event.entities),
            "raw_entities": list(event.raw_entities),
        })

    thread_rows: list[dict] = []
    for thread in threads:
        thread_rows.append({
            "thread_id": thread.thread_id,
            "slug": thread.slug,
            "headline": thread.headline,
            "status": thread.status,
            "significance": thread.significance,
            "snapshot_count": thread.snapshot_count,
            "trajectory_label": thread.trajectory_label,
            "momentum_score": thread.momentum_score,
            "velocity_7d": thread.velocity_7d,
            "acceleration_7d": thread.acceleration_7d,
            "significance_trend_7d": thread.significance_trend_7d,
            "event_ids": [event.event_id for event in thread.events if event.event_id],
        })

    evidence_catalog = {
        "event_ids": [row["event_id"] for row in event_rows if row["event_id"] is not None],
        "thread_ids": [row["thread_id"] for row in thread_rows if row["thread_id"] is not None],
        "signal_ids": [signal.signal_id for signal in cross_topic_signals if signal.signal_id is not None],
    }

    return GraphExportBundle(
        schema_version=schema_version,
        topic_slug=topic_slug,
        topic_name=topic_name,
        cutoff=cutoff,
        threads=thread_rows,
        events=event_rows,
        causal_links=causal_links,
        cross_topic_signals=[signal.model_dump(mode="json") for signal in cross_topic_signals],
        nodes=snapshot.nodes,
        edges=snapshot.edges,
        evidence_catalog=evidence_catalog,
        metadata={"metrics": snapshot.metrics},
    )


def export_graph_bundle(base_dir: Path, adapter_name: str, bundle: GraphExportBundle) -> Path:
    """Persist a canonical graph export bundle for a sidecar adapter."""
    out_dir = base_dir / adapter_name.replace("-", "_")
    out_dir.mkdir(parents=True, exist_ok=True)
    export_path = out_dir / f"{bundle.topic_slug}-{bundle.cutoff.isoformat()}.json"
    export_path.write_text(json.dumps(bundle.model_dump(mode="json"), indent=2))
    return export_path


def bundle_to_snapshot(bundle: GraphExportBundle) -> GraphSnapshot:
    """Convert a graph export bundle back into a lightweight snapshot."""
    return GraphSnapshot(
        topic_slug=bundle.topic_slug,
        generated_for=bundle.cutoff,
        nodes=bundle.nodes,
        edges=bundle.edges,
        metrics=bundle.metadata.get("metrics", {}),
    )


class GraphEvidenceAdapter(Protocol):
    """Read-only sidecar contract for graph evidence ranking."""

    adapter_name: str

    async def query(self, bundle: GraphExportBundle, *, max_evidence_ids: int = 8) -> GraphEvidenceResult:
        """Return ranked evidence IDs from a graph bundle."""


class KuzuGraphEvidenceAdapter:
    """Optional Kuzu sidecar adapter used only for readiness and benchmark probing."""

    adapter_name = "kuzu"

    async def query(self, bundle: GraphExportBundle, *, max_evidence_ids: int = 8) -> GraphEvidenceResult:
        if importlib.util.find_spec("kuzu") is None:
            return GraphEvidenceResult(
                adapter=self.adapter_name,
                status="skipped",
                metadata={"reason": "kuzu dependency not installed"},
            )
        ranked = rank_graph_evidence(
            bundle_to_snapshot(bundle),
            max_event_ids=max_evidence_ids,
            max_thread_ids=max(1, max_evidence_ids // 2),
            max_signal_ids=max(1, max_evidence_ids // 2),
        )
        return GraphEvidenceResult(
            adapter=self.adapter_name,
            status="ready",
            event_ids=ranked["event_ids"],
            thread_ids=ranked["thread_ids"],
            signal_ids=ranked["signal_ids"],
            paths=[],
            metadata={"backend": "kuzu", "mode": "adapter_probe"},
        )


class GraphitiGraphEvidenceAdapter:
    """Optional Graphiti adapter kept off the runtime path."""

    adapter_name = "graphiti"

    async def query(self, bundle: GraphExportBundle, *, max_evidence_ids: int = 8) -> GraphEvidenceResult:
        if importlib.util.find_spec("graphiti_core") is None and importlib.util.find_spec("graphiti") is None:
            return GraphEvidenceResult(
                adapter=self.adapter_name,
                status="skipped",
                metadata={"reason": "graphiti dependency not installed"},
            )
        ranked = rank_graph_evidence(
            bundle_to_snapshot(bundle),
            max_event_ids=max_evidence_ids,
            max_thread_ids=max(1, max_evidence_ids // 2),
            max_signal_ids=max(1, max_evidence_ids // 2),
        )
        return GraphEvidenceResult(
            adapter=self.adapter_name,
            status="ready",
            event_ids=ranked["event_ids"],
            thread_ids=ranked["thread_ids"],
            signal_ids=ranked["signal_ids"],
            paths=[],
            metadata={"backend": "graphiti", "mode": "adapter_probe"},
        )


def get_graph_evidence_adapter(adapter_name: str) -> GraphEvidenceAdapter:
    """Resolve a graph evidence adapter by name."""
    normalized = adapter_name.strip().lower()
    if normalized == "kuzu":
        return KuzuGraphEvidenceAdapter()
    if normalized == "graphiti":
        return GraphitiGraphEvidenceAdapter()
    raise ValueError(f"Unknown graph evidence adapter: {adapter_name}")
