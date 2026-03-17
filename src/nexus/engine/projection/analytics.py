"""Deterministic analytics for thread trajectories and cross-topic bridges."""

from __future__ import annotations

from collections import defaultdict
from datetime import date, timedelta
from typing import TYPE_CHECKING

from nexus.engine.projection.models import CrossTopicSignal, ThreadSnapshot

if TYPE_CHECKING:
    from nexus.engine.synthesis.knowledge import NarrativeThread


# Trajectory classification thresholds.
# Tuned empirically against historical thread progressions.
BREAK_MIN_EVENTS = 3
BREAK_MIN_SIGNIFICANCE = 7
BREAK_MIN_VELOCITY = 2.0
ACCEL_VELOCITY_THRESHOLD = 1.0
ACCEL_ACCEL_THRESHOLD = 0.5
ACCEL_SIG_TREND_THRESHOLD = 1.0
DECEL_VELOCITY_THRESHOLD = -0.5
DECEL_ACCEL_THRESHOLD = -0.5
DECEL_SIG_TREND_THRESHOLD = -1.0


def classify_trajectory(
    velocity_7d: float,
    acceleration_7d: float,
    significance_trend_7d: float,
    event_count: int,
    significance: int,
) -> str:
    """Assign a simple, interpretable trajectory label."""
    if (
        event_count >= BREAK_MIN_EVENTS
        and significance >= BREAK_MIN_SIGNIFICANCE
        and velocity_7d >= BREAK_MIN_VELOCITY
        and acceleration_7d > 0
    ):
        return "about_to_break"
    if (
        velocity_7d > ACCEL_VELOCITY_THRESHOLD
        or acceleration_7d > ACCEL_ACCEL_THRESHOLD
        or significance_trend_7d > ACCEL_SIG_TREND_THRESHOLD
    ):
        return "accelerating"
    if (
        velocity_7d < DECEL_VELOCITY_THRESHOLD
        or acceleration_7d < DECEL_ACCEL_THRESHOLD
        or significance_trend_7d < DECEL_SIG_TREND_THRESHOLD
    ):
        return "decelerating"
    return "steady"


def compute_snapshot_metrics(history: list[ThreadSnapshot], current: ThreadSnapshot) -> ThreadSnapshot:
    """Populate derived metrics for a snapshot using prior snapshot history."""
    previous = sorted(
        [snap for snap in history if snap.snapshot_date < current.snapshot_date],
        key=lambda snap: snap.snapshot_date,
    )

    def _find_baseline(days: int) -> ThreadSnapshot | None:
        cutoff = current.snapshot_date - timedelta(days=days)
        candidates = [snap for snap in previous if snap.snapshot_date <= cutoff]
        return candidates[-1] if candidates else (previous[0] if previous else None)

    baseline_7 = _find_baseline(7)
    baseline_14 = _find_baseline(14)

    current_velocity = float(current.event_count - (baseline_7.event_count if baseline_7 else 0))
    previous_velocity = 0.0
    if baseline_7 and baseline_14:
        previous_velocity = float(baseline_7.event_count - baseline_14.event_count)

    significance_trend = float(
        current.significance - (baseline_7.significance if baseline_7 else current.significance)
    )
    acceleration = current_velocity - previous_velocity
    momentum = round((current_velocity * 1.5) + acceleration + significance_trend, 3)

    current.velocity_7d = round(current_velocity, 3)
    current.acceleration_7d = round(acceleration, 3)
    current.significance_trend_7d = round(significance_trend, 3)
    current.momentum_score = momentum
    current.trajectory_label = classify_trajectory(
        current.velocity_7d,
        current.acceleration_7d,
        current.significance_trend_7d,
        current.event_count,
        current.significance,
    )
    return current


def detect_cross_topic_signals(rows: list[dict], lookback_days: int = 14) -> list[CrossTopicSignal]:
    """Detect shared-entity bridges across topics in a recent time window."""
    if not rows:
        return []

    grouped: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    newest = max(row["event_date"] for row in rows)
    cutoff = newest - timedelta(days=lookback_days)

    for row in rows:
        event_date = row["event_date"]
        if event_date < cutoff:
            continue
        grouped[row["canonical_name"]][row["topic_slug"]].append(row)

    signals: list[CrossTopicSignal] = []
    seen_keys: set[tuple[str, str, str, date]] = set()

    for entity_name, topic_map in grouped.items():
        topic_slugs = sorted(topic_map)
        if len(topic_slugs) < 2:
            continue

        for idx, topic_slug in enumerate(topic_slugs):
            for related_slug in topic_slugs[idx + 1:]:
                for row_a in topic_map[topic_slug]:
                    for row_b in topic_map[related_slug]:
                        delta_hours = abs((row_a["event_date"] - row_b["event_date"]).days) * 24
                        if delta_hours > 48:
                            continue
                        observed_at = max(row_a["event_date"], row_b["event_date"])
                        dedup_key = (topic_slug, related_slug, entity_name, observed_at)
                        if dedup_key in seen_keys:
                            continue
                        seen_keys.add(dedup_key)
                        note = (
                            f"{entity_name} was active in {topic_slug} and {related_slug} "
                            f"within 48h."
                        )
                        signals.append(CrossTopicSignal(
                            topic_slug=topic_slug,
                            related_topic_slug=related_slug,
                            shared_entity=entity_name,
                            observed_at=observed_at,
                            event_ids=[row_a["event_id"]],
                            related_event_ids=[row_b["event_id"]],
                            note=note,
                        ))
                        signals.append(CrossTopicSignal(
                            topic_slug=related_slug,
                            related_topic_slug=topic_slug,
                            shared_entity=entity_name,
                            observed_at=observed_at,
                            event_ids=[row_b["event_id"]],
                            related_event_ids=[row_a["event_id"]],
                            note=note,
                        ))

    signals.sort(key=lambda signal: (signal.topic_slug, signal.observed_at, signal.shared_entity))
    return signals


def detect_converging_threads(
    threads: list[NarrativeThread],
    *,
    min_shared_entities: int = 2,
) -> list[dict]:
    """Find thread pairs sharing entities where both are accelerating or about to break.

    Returns a list of convergence signal dicts:
        {threads: [id_a, id_b], shared_entities: [...], pattern: str, confidence: float}
    """
    hot_labels = {"accelerating", "about_to_break"}
    # Filter to threads with an id and a hot trajectory
    eligible = [
        t for t in threads
        if t.thread_id is not None and (t.trajectory_label or "steady") in hot_labels
    ]

    signals: list[dict] = []
    for i in range(len(eligible)):
        entities_i = {e.lower() for e in eligible[i].key_entities}
        for j in range(i + 1, len(eligible)):
            entities_j = {e.lower() for e in eligible[j].key_entities}
            shared = entities_i & entities_j
            if len(shared) < min_shared_entities:
                continue

            # Confidence: ratio of shared entities to smaller entity set, capped at 1.0
            min_size = min(len(entities_i), len(entities_j))
            confidence = round(len(shared) / max(min_size, 1), 3)

            # Preserve original casing for shared entities
            all_entities = {e.lower(): e for e in eligible[i].key_entities}
            all_entities.update({e.lower(): e for e in eligible[j].key_entities})
            shared_original = sorted(all_entities[s] for s in shared)

            label_i = eligible[i].trajectory_label or "accelerating"
            label_j = eligible[j].trajectory_label or "accelerating"
            pattern = f"both_{label_i}" if label_i == label_j else f"{label_i}+{label_j}"

            signals.append({
                "threads": [eligible[i].thread_id, eligible[j].thread_id],
                "shared_entities": shared_original,
                "pattern": pattern,
                "confidence": confidence,
            })

    signals.sort(key=lambda s: s["confidence"], reverse=True)
    return signals
