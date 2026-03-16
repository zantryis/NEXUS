"""Budget-safe deterministic evaluation for stored projections."""

from __future__ import annotations

import json
import re
from math import log
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path
from statistics import mean

from nexus.config.models import NexusConfig
from nexus.engine.knowledge.store import KnowledgeStore
from nexus.engine.projection.engines import ProjectionEngineInput, get_projection_engine
from nexus.engine.projection.forecasting import FAMILY_TARGET_SPECS
from nexus.engine.projection.forecasting import get_forecast_engine
from nexus.engine.projection.graph import (
    build_graph_export_bundle,
    export_graph_bundle,
    get_graph_evidence_adapter,
)
from nexus.engine.projection.historical import HistoricalTopicState
from nexus.engine.projection.kalshi import (
    KalshiClient,
    KalshiLedger,
    compare_forecasts_to_kalshi,
    load_kalshi_mappings,
)
from nexus.engine.projection.models import ForecastResolution, ProjectionOutcome
from nexus.engine.projection.service import (
    _build_forecast_input,
    hydrate_synthesis_threads,
    load_historical_topic_state,
    projection_eligibility,
    topic_slug_from_name,
)
from nexus.engine.synthesis.knowledge import TopicSynthesis


_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "for", "from", "in", "into", "is", "it",
    "likely", "next", "of", "on", "or", "over", "that", "the", "this", "to", "will", "with",
}


def _keywords(text: str) -> set[str]:
    return {
        token.lower() for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9'-]+", text)
        if len(token) > 2 and token.lower() not in _STOPWORDS
    }


async def auto_evaluate_projections(
    store: KnowledgeStore,
    *,
    start: date,
    end: date,
    engine: str | None = None,
) -> dict:
    """Evaluate stored projections against subsequent events without LLM judge calls."""
    due_items = await store.get_pending_projection_items(until=end, engine=engine)
    summary: dict[str, dict[str, float]] = defaultdict(lambda: {"total": 0, "hit": 0, "miss": 0, "mixed": 0})

    for item in due_items:
        generated_for = date.fromisoformat(item["generated_for"])
        review_after = date.fromisoformat(item["review_after"])
        if generated_for < start or generated_for > end:
            continue
        future_events = await store.get_events(
            item["topic_slug"],
            since=generated_for + timedelta(days=1),
            until=review_after,
        )
        score = evaluate_projection_item(item["claim"], item["signpost"], future_events)
        if score >= 0.6:
            outcome_status = "hit"
        elif score >= 0.3:
            outcome_status = "mixed"
        else:
            outcome_status = "miss"

        summary[item["engine"]]["total"] += 1
        summary[item["engine"]][outcome_status] += 1
        await store.set_projection_outcome(ProjectionOutcome(
            projection_item_id=item["projection_item_id"],
            outcome_status=outcome_status,
            score=score,
            notes=f"Auto-evaluated against {len(future_events)} future events.",
            reviewed_at=end,
        ))

    for engine_name, metrics in summary.items():
        total = metrics["total"] or 1
        metrics["hit_rate"] = round(metrics["hit"] / total, 3)
    return dict(summary)


def evaluate_projection_item(claim: str, signpost: str, future_events) -> float:
    """Score a projection item against later event summaries."""
    target_terms = _keywords(claim) | _keywords(signpost)
    if not target_terms:
        return 0.0

    matched = 0
    for event in future_events:
        haystack = " ".join([event.summary, " ".join(event.entities)]).lower()
        event_terms = _keywords(haystack)
        if target_terms & event_terms:
            matched += 1
    return round(min(1.0, matched / max(1, len(future_events) or 1)), 3)


async def trajectory_lift_report(store: KnowledgeStore, *, start: date, end: date) -> dict:
    """Compare future event growth for accelerating versus non-accelerating threads."""
    cursor = await store.db.execute(
        "SELECT thread_id, snapshot_date, event_count, trajectory_label FROM thread_snapshots "
        "WHERE snapshot_date >= ? AND snapshot_date <= ? ORDER BY snapshot_date ASC",
        (start.isoformat(), end.isoformat()),
    )
    rows = await cursor.fetchall()
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        thread_id = row[0]
        snapshot_date = date.fromisoformat(row[1])
        current_count = row[2]
        future_stats = await store.get_thread_event_stats(thread_id, until=snapshot_date + timedelta(days=7))
        grouped[row[3]].append(float(future_stats["event_count"] - current_count))

    return {
        label: round(sum(deltas) / len(deltas), 3)
        for label, deltas in grouped.items() if deltas
    }


async def cross_topic_bridge_report(store: KnowledgeStore, *, start: date, end: date) -> dict:
    """Measure whether saved bridge signals are followed by more activity."""
    cursor = await store.db.execute(
        "SELECT topic_slug, related_topic_slug, observed_at FROM cross_topic_signals "
        "WHERE observed_at >= ? AND observed_at <= ?",
        (start.isoformat(), end.isoformat()),
    )
    rows = await cursor.fetchall()
    follow_on = 0
    for row in rows:
        observed_at = date.fromisoformat(row[2])
        related_events = await store.get_events(
            row[1],
            since=observed_at + timedelta(days=1),
            until=observed_at + timedelta(days=7),
        )
        if related_events:
            follow_on += 1
    total = len(rows)
    return {
        "signals": total,
        "follow_on_signals": follow_on,
        "follow_on_rate": round(follow_on / total, 3) if total else 0.0,
    }


async def compare_projection_engines(
    store: KnowledgeStore,
    *,
    start: date,
    end: date,
    engines: list[str],
) -> dict:
    """Run deterministic engine comparisons on historical synthesis cutoffs."""
    topic_stats = await store.get_topic_stats()
    results: dict[str, dict[str, float]] = {}
    for engine_name in engines:
        engine = get_projection_engine(engine_name)
        scores: list[float] = []
        item_count = 0
        for topic in topic_stats:
            topic_slug = topic["topic_slug"]
            synthesis_dates = [
                date.fromisoformat(raw)
                for raw in await store.get_synthesis_dates(topic_slug)
                if start <= date.fromisoformat(raw) <= end
            ]
            for cutoff in synthesis_dates:
                synthesis_data = await store.get_synthesis(topic_slug, cutoff)
                if not synthesis_data:
                    continue
                synthesis = TopicSynthesis(**synthesis_data)
                recent_events = await store.get_recent_events(topic_slug, days=14, limit=40, reference_date=cutoff)
                cross_topic_rows = await store.get_recent_entity_activity(reference_date=cutoff, lookback_days=14)
                cross_topic_signals = await store.get_cross_topic_signals_as_of(topic_slug, cutoff, limit=5)
                payload = ProjectionEngineInput(
                    topic_slug=topic_slug,
                    topic_name=synthesis.topic_name,
                    run_date=cutoff,
                    threads=synthesis.threads,
                    recent_events=recent_events,
                    cross_topic_signals=cross_topic_signals,
                    trajectory_threads=synthesis.threads,
                    metadata={"cross_topic_rows": len(cross_topic_rows)},
                )
                projection = await engine.project(None, payload, critic_pass=False, max_items=3)
                future_events = await store.get_events(
                    topic_slug,
                    since=cutoff + timedelta(days=1),
                    until=cutoff + timedelta(days=14),
                )
                for item in projection.items:
                    scores.append(evaluate_projection_item(item.claim, item.signpost, future_events))
                    item_count += 1
        results[engine_name] = {
            "avg_score": round(sum(scores) / len(scores), 3) if scores else 0.0,
            "items_evaluated": item_count,
        }
    return results


def _bounded_probability(value: float) -> float:
    return max(0.05, min(0.95, float(value)))


def _brier_score(probability: float, outcome: bool) -> float:
    target = 1.0 if outcome else 0.0
    return round((_bounded_probability(probability) - target) ** 2, 4)


def _log_loss(probability: float, outcome: bool) -> float:
    p = _bounded_probability(probability)
    return round(-log(p if outcome else (1.0 - p)), 4)


def _bucket_label(probability: float) -> str:
    p = _bounded_probability(probability)
    if p < 0.2:
        return "0.05-0.20"
    if p < 0.4:
        return "0.20-0.40"
    if p < 0.6:
        return "0.40-0.60"
    if p < 0.8:
        return "0.60-0.80"
    return "0.80-0.95"


def _event_mentions_entity(event, entity: str) -> bool:
    entity_lower = entity.lower()
    haystack = [part.lower() for part in event.entities + event.raw_entities]
    return entity_lower in haystack


def _event_matches_keywords(event, keywords: list[str]) -> bool:
    haystack = " ".join([
        event.summary.lower(),
        " ".join(entity.lower() for entity in event.entities),
        " ".join(entity.lower() for entity in event.raw_entities),
    ])
    return any(keyword.lower() in haystack for keyword in keywords)


def _event_matches_anchor_entities(event, anchor_entities: list[str]) -> bool:
    if not anchor_entities:
        return True
    return any(_event_mentions_entity(event, entity) for entity in anchor_entities)


async def _has_sufficient_future_data(store: KnowledgeStore, topic_slug: str, resolution_date: date) -> bool:
    topic_range = await store.get_topic_event_range(topic_slug)
    return bool(topic_range["last_date"] and topic_range["last_date"] >= resolution_date)


async def resolve_forecast_question(store: KnowledgeStore, item: dict) -> ForecastResolution:
    """Resolve a structured forecast question against subsequent events."""
    run_date = date.fromisoformat(item["generated_for"])
    resolution_date = date.fromisoformat(item["resolution_date"])
    target_variable = item["target_variable"]
    target_metadata = item.get("target_metadata", {})
    probability = float(item["probability"])
    notes = ""

    relevant_topic = item["topic_slug"]
    if target_variable == "cross_topic_follow_on":
        relevant_topic = target_metadata["related_topic_slug"]

    if not await _has_sufficient_future_data(store, relevant_topic, resolution_date):
        return ForecastResolution(
            forecast_question_id=item["forecast_question_id"],
            outcome_status="unresolved",
            notes="Not enough future data collected yet.",
        )

    resolved_bool: bool | None = None
    actual_value: float | None = None
    realized_direction: str | None = None

    if target_variable == "thread_new_event_count":
        thread_id = int(target_metadata["thread_id"])
        threshold = int(target_metadata.get("threshold", 1))
        events = await store.get_events_for_thread(thread_id)
        new_events = [event for event in events if run_date < event.date <= resolution_date]
        actual_value = float(len(new_events))
        realized_direction = "up" if new_events else "flat"
        resolved_bool = len(new_events) >= threshold
        notes = f"Observed {len(new_events)} new thread-linked events."

    elif target_variable == "topic_event_growth":
        horizon_days = int(item["horizon_days"])
        future_events = await store.get_events(
            item["topic_slug"],
            since=run_date + timedelta(days=1),
            until=resolution_date,
        )
        prior_events = await store.get_events(
            item["topic_slug"],
            since=run_date - timedelta(days=horizon_days - 1),
            until=run_date,
        )
        delta = len(future_events) - len(prior_events)
        actual_value = float(delta)
        realized_direction = "up" if delta > 0 else ("down" if delta < 0 else "flat")
        expected_direction = item.get("expected_direction") or "up"
        resolved_bool = realized_direction == expected_direction
        notes = f"Future window delta vs prior window: {delta}."

    elif target_variable == "entity_recurrence":
        entity = target_metadata["entity"]
        events = await store.get_events(
            item["topic_slug"],
            since=run_date + timedelta(days=1),
            until=resolution_date,
        )
        matches = [event for event in events if _event_mentions_entity(event, entity)]
        actual_value = float(len(matches))
        realized_direction = "up" if matches else "flat"
        resolved_bool = bool(matches)
        notes = f"Observed {len(matches)} future events mentioning {entity}."

    elif target_variable == "cross_topic_follow_on":
        entity = target_metadata["entity"]
        related_topic_slug = target_metadata["related_topic_slug"]
        events = await store.get_events(
            related_topic_slug,
            since=run_date + timedelta(days=1),
            until=resolution_date,
        )
        matches = [event for event in events if _event_mentions_entity(event, entity)]
        actual_value = float(len(matches))
        realized_direction = "up" if matches else "flat"
        resolved_bool = bool(matches)
        notes = f"Observed {len(matches)} future {related_topic_slug} events mentioning {entity}."

    elif target_variable in FAMILY_TARGET_SPECS:
        keywords = target_metadata.get("keywords", [])
        anchor_entities = target_metadata.get("anchor_entities", [])
        events = await store.get_events(
            item["topic_slug"],
            since=run_date + timedelta(days=1),
            until=resolution_date,
        )
        matches = [
            event for event in events
            if _event_matches_keywords(event, keywords) and _event_matches_anchor_entities(event, anchor_entities)
        ]
        actual_value = float(len(matches))
        realized_direction = "up" if matches else "flat"
        resolved_bool = bool(matches)
        notes = (
            f"Observed {len(matches)} future {item['topic_slug']} events matching "
            f"{target_variable} keywords."
        )

    else:
        return ForecastResolution(
            forecast_question_id=item["forecast_question_id"],
            outcome_status="invalid",
            notes=f"Unsupported target variable: {target_variable}",
        )

    return ForecastResolution(
        forecast_question_id=item["forecast_question_id"],
        outcome_status="resolved",
        resolved_bool=resolved_bool,
        realized_direction=realized_direction,
        actual_value=actual_value,
        brier_score=_brier_score(probability, bool(resolved_bool)),
        log_loss=_log_loss(probability, bool(resolved_bool)),
        notes=notes,
        resolved_at=resolution_date,
    )


def _summarize_calibration(rows: list[dict]) -> dict:
    buckets: dict[str, dict[str, float]] = defaultdict(lambda: {"total": 0, "resolved_true": 0})
    for row in rows:
        bucket = _bucket_label(float(row["probability"]))
        buckets[bucket]["total"] += 1
        buckets[bucket]["resolved_true"] += 1 if row["resolved_bool"] else 0
    return {
        bucket: {
            "total": metrics["total"],
            "resolved_true": metrics["resolved_true"],
            "hit_rate": round(metrics["resolved_true"] / metrics["total"], 3) if metrics["total"] else 0.0,
        }
        for bucket, metrics in buckets.items()
    }


def statistical_significance_test(
    native_briers: list[float],
    baseline_briers: list[float],
) -> dict:
    """Paired t-test on Brier scores between native and baseline engines.

    Returns {t_statistic, p_value, n, significant_at_005, native_mean, baseline_mean, note}.
    """
    n = min(len(native_briers), len(baseline_briers))
    result: dict = {
        "n": n,
        "native_mean": round(mean(native_briers), 4) if native_briers else 0.0,
        "baseline_mean": round(mean(baseline_briers), 4) if baseline_briers else 0.0,
    }
    if n < 5:
        result.update({"t_statistic": None, "p_value": None, "significant_at_005": False,
                        "note": "insufficient samples for reliable test"})
        return result
    try:
        from scipy.stats import ttest_rel
        stat, p_value = ttest_rel(native_briers[:n], baseline_briers[:n])
        result.update({
            "t_statistic": round(float(stat), 4),
            "p_value": round(float(p_value), 6),
            "significant_at_005": bool(p_value < 0.05),
            "note": "paired t-test on Brier scores",
        })
    except ImportError:
        # Fallback: manual paired t-test without scipy
        diffs = [a - b for a, b in zip(native_briers[:n], baseline_briers[:n])]
        mean_diff = sum(diffs) / n
        var_diff = sum((d - mean_diff) ** 2 for d in diffs) / (n - 1) if n > 1 else 0.0
        if var_diff == 0:
            result.update({"t_statistic": 0.0, "p_value": 1.0, "significant_at_005": False,
                            "note": "zero variance — scores identical"})
            return result
        se = (var_diff / n) ** 0.5
        t_stat = mean_diff / se
        # Approximate two-tailed p-value using normal approximation for large n
        from math import erfc
        p_approx = erfc(abs(t_stat) / (2 ** 0.5))
        result.update({
            "t_statistic": round(t_stat, 4),
            "p_value": round(p_approx, 6),
            "significant_at_005": bool(p_approx < 0.05),
            "note": "paired t-test (scipy unavailable, normal approx)",
        })
    return result


def _finalize_engine_rows(rows: list[dict]) -> dict:
    total = len(rows)
    return {
        "total": total,
        "accuracy": round(sum(1 for row in rows if row["resolved_bool"]) / total, 3) if total else 0.0,
        "mean_brier": round(sum(row["brier_score"] for row in rows) / total, 4) if total else 0.0,
        "mean_log_loss": round(sum(row["log_loss"] for row in rows) / total, 4) if total else 0.0,
        "calibration": _summarize_calibration(rows),
    }


def _validity_label(cutoff_count: int, resolved_forecast_count: int, domains: set[str]) -> str:
    if cutoff_count >= 30 and resolved_forecast_count >= 100 and len(domains) >= 2:
        return "statistically-valid benchmark"
    return "infrastructure-valid, statistically-insufficient"


def _float_changed(before: float | None, after: float | None) -> bool:
    before_norm = round(float(before or 0.0), 4)
    after_norm = round(float(after or 0.0), 4)
    return before_norm != after_norm


async def audit_forecast_leakage(
    store: KnowledgeStore,
    config: NexusConfig,
    *,
    start: date,
    end: date,
    profile: str = "signal-rich",
) -> dict:
    """Audit known leakage vectors in the legacy replay path."""
    topics_audited: set[str] = set()
    thread_state_leak_cutoffs = 0
    future_signal_leak_cutoffs = 0
    examples: list[dict] = []
    cutoffs_audited = 0

    for topic in config.topics:
        topic_slug = topic_slug_from_name(topic.name)
        synthesis_dates = sorted(
            date.fromisoformat(raw)
            for raw in await store.get_synthesis_dates(topic_slug)
            if start <= date.fromisoformat(raw) <= end
        )
        for cutoff in synthesis_dates:
            state = await load_historical_topic_state(
                store,
                topic_slug=topic_slug,
                topic_name=topic.name,
                cutoff=cutoff,
                config=config.future_projection,
                profile=profile,
            )
            if not state:
                continue
            cutoffs_audited += 1
            topics_audited.add(topic_slug)

            raw = await store.get_synthesis(topic_slug, cutoff)
            if not raw:
                continue

            unsafe = await hydrate_synthesis_threads(
                store,
                TopicSynthesis(**raw),
                topic_slug=topic_slug,
            )
            safe = await hydrate_synthesis_threads(
                store,
                TopicSynthesis(**raw),
                topic_slug=topic_slug,
                as_of=cutoff,
            )

            by_headline = {thread.headline: thread for thread in unsafe.threads}
            leak_detected = False
            for historical_thread in safe.threads:
                current_thread = by_headline.get(historical_thread.headline)
                if not current_thread:
                    continue
                if (
                    current_thread.snapshot_count != historical_thread.snapshot_count
                    or current_thread.trajectory_label != historical_thread.trajectory_label
                    or _float_changed(current_thread.momentum_score, historical_thread.momentum_score)
                ):
                    leak_detected = True
                    if len(examples) < 10:
                        examples.append({
                            "type": "thread_state",
                            "topic_slug": topic_slug,
                            "cutoff": cutoff.isoformat(),
                            "thread": historical_thread.headline,
                            "latest_snapshot_count": current_thread.snapshot_count,
                            "as_of_snapshot_count": historical_thread.snapshot_count,
                            "latest_trajectory": current_thread.trajectory_label,
                            "as_of_trajectory": historical_thread.trajectory_label,
                        })
                    break
            if leak_detected:
                thread_state_leak_cutoffs += 1

            latest_signals = await store.get_cross_topic_signals(topic_slug, limit=5)
            future_signals = [signal for signal in latest_signals if signal.observed_at > cutoff]
            if future_signals:
                future_signal_leak_cutoffs += 1
                if len(examples) < 10:
                    examples.append({
                        "type": "cross_topic_signal",
                        "topic_slug": topic_slug,
                        "cutoff": cutoff.isoformat(),
                        "future_signal_dates": [signal.observed_at.isoformat() for signal in future_signals],
                        "entities": [signal.shared_entity for signal in future_signals],
                    })

    return {
        "cutoffs_audited": cutoffs_audited,
        "topics_audited": sorted(topics_audited),
        "thread_state_leak_cutoffs": thread_state_leak_cutoffs,
        "future_signal_leak_cutoffs": future_signal_leak_cutoffs,
        "passes_strict_gate": thread_state_leak_cutoffs == 0 and future_signal_leak_cutoffs == 0,
        "examples": examples,
    }


async def auto_resolve_forecasts(
    store: KnowledgeStore,
    *,
    start: date,
    end: date,
    engine: str | None = None,
) -> dict:
    """Resolve stored forecast questions and score calibration-aware metrics."""
    due_items = await store.get_pending_forecast_questions(until=end, engine=engine)
    by_engine: dict[str, list[dict]] = defaultdict(list)

    for item in due_items:
        generated_for = date.fromisoformat(item["generated_for"])
        if generated_for < start or generated_for > end:
            continue
        resolution = await resolve_forecast_question(store, item)
        await store.set_forecast_resolution(resolution)
        if resolution.outcome_status != "resolved":
            continue
        by_engine[item["engine"]].append({
            "probability": item["probability"],
            "resolved_bool": bool(resolution.resolved_bool),
            "brier_score": resolution.brier_score or 0.0,
            "log_loss": resolution.log_loss or 0.0,
        })

    summary = {}
    for engine_name, rows in by_engine.items():
        summary[engine_name] = _finalize_engine_rows(rows)
    return summary


async def benchmark_forecast_engines(
    store: KnowledgeStore,
    config: NexusConfig,
    *,
    start: date,
    end: date,
    engines: list[str],
    llm=None,
    mode: str = "audit",
    profile: str = "signal-rich",
    strict: bool = True,
    max_questions: int = 4,
    min_thread_snapshots_override: int | None = None,
) -> dict:
    """Replay quantified forecast engines across historical syntheses and score them."""
    if mode not in {"audit", "replay"}:
        raise ValueError(f"Unsupported benchmark mode: {mode}")

    results: dict[str, dict] = {}
    raw_briers: dict[str, list[float]] = {}
    cutoffs_used: set[tuple[str, str]] = set()
    domains: set[str] = set()

    for engine_name in engines:
        engine = get_forecast_engine(engine_name)
        scored_rows: list[dict] = []
        for topic in config.topics:
            topic_slug = topic_slug_from_name(topic.name)
            synthesis_dates = sorted([
                date.fromisoformat(raw)
                for raw in await store.get_synthesis_dates(topic_slug)
                if start <= date.fromisoformat(raw) <= end
            ])
            for cutoff in synthesis_dates:
                state = await load_historical_topic_state(
                    store,
                    topic_slug=topic_slug,
                    topic_name=topic.name,
                    cutoff=cutoff,
                    config=config.future_projection,
                    profile=profile,
                    min_thread_snapshots_override=min_thread_snapshots_override,
                )
                if not state:
                    continue
                cutoffs_used.add((topic_slug, cutoff.isoformat()))
                domains.add(topic_slug)
                payload = await _build_forecast_input(
                    store,
                    state.synthesis,
                    topic_slug=topic_slug,
                    run_date=cutoff,
                    cross_topic_signals=state.cross_topic_signals,
                    metadata=state.metadata,
                )
                calibration_data = await store.get_historical_calibration(as_of=cutoff)
                # Swarm/graph engines always need LLM;
                # native gets LLM only in non-strict mode for reproducibility.
                needs_llm = engine_name in {"swarm", "graph"} or (not strict and engine_name == "native")
                extra_kwargs = {}
                if engine_name == "graph":
                    extra_kwargs["store"] = store
                forecast_run = await engine.generate(
                    llm if needs_llm else None,
                    payload,
                    critic_pass=(engine_name == "native" and not strict),
                    max_questions=max_questions,
                    calibration_data=calibration_data or None,
                    **extra_kwargs,
                )
                for question in forecast_run.questions:
                    resolution = await resolve_forecast_question(
                        store,
                        {
                            "forecast_question_id": -1,
                            "topic_slug": topic_slug,
                            "engine": engine_name,
                            "generated_for": cutoff.isoformat(),
                            "question": question.question,
                            "forecast_type": question.forecast_type,
                            "target_variable": question.target_variable,
                            "target_metadata": question.target_metadata,
                            "probability": question.probability,
                            "base_rate": question.base_rate,
                            "resolution_criteria": question.resolution_criteria,
                            "resolution_date": question.resolution_date.isoformat(),
                            "horizon_days": question.horizon_days,
                            "expected_direction": question.expected_direction,
                        },
                    )
                    if resolution.outcome_status != "resolved":
                        continue
                    scored_rows.append({
                        "probability": question.probability,
                        "resolved_bool": bool(resolution.resolved_bool),
                        "brier_score": resolution.brier_score or 0.0,
                        "log_loss": resolution.log_loss or 0.0,
                    })
        raw_briers[engine_name] = [row["brier_score"] for row in scored_rows]
        results[engine_name] = _finalize_engine_rows(scored_rows)

    # Add significance test if both native and baseline are present
    significance = None
    if "native" in raw_briers and "baseline" in raw_briers:
        significance = statistical_significance_test(raw_briers["native"], raw_briers["baseline"])

    resolved_forecast_count = max((result["total"] for result in results.values()), default=0)
    meta = {
        "mode": mode,
        "profile": profile,
        "strict": strict,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "cutoff_count": len(cutoffs_used),
        "domains": sorted(domains),
        "resolved_forecast_count": resolved_forecast_count,
        "validity_label": _validity_label(len(cutoffs_used), resolved_forecast_count, domains),
    }
    if significance:
        meta["significance_test"] = significance
    if strict:
        meta["leakage_audit"] = await audit_forecast_leakage(
            store,
            config,
            start=start,
            end=end,
            profile=profile,
        )
    return {"meta": meta, "engines": results}


async def export_graph_bundles(
    store: KnowledgeStore,
    config: NexusConfig,
    *,
    start: date,
    end: date,
    profile: str = "signal-rich",
    target_dir: Path,
) -> dict:
    """Export canonical graph bundles from strict replay state."""
    target_dir.mkdir(parents=True, exist_ok=True)
    exported_paths: list[str] = []
    for topic in config.topics:
        topic_slug = topic_slug_from_name(topic.name)
        synthesis_dates = sorted([
            date.fromisoformat(raw)
            for raw in await store.get_synthesis_dates(topic_slug)
            if start <= date.fromisoformat(raw) <= end
        ])
        for cutoff in synthesis_dates:
            state = await load_historical_topic_state(
                store,
                topic_slug=topic_slug,
                topic_name=topic.name,
                cutoff=cutoff,
                config=config.future_projection,
                profile=profile,
            )
            if not state:
                continue
            event_ids = [event.event_id for event in state.recent_events if event.event_id is not None]
            causal_links = await store.get_causal_links_for_events(event_ids)
            bundle = build_graph_export_bundle(
                topic_slug=topic_slug,
                topic_name=topic.name,
                cutoff=cutoff,
                threads=state.synthesis.threads,
                recent_events=state.recent_events,
                causal_links=causal_links,
                cross_topic_signals=state.cross_topic_signals,
                schema_version=config.future_projection.graph_sidecars.export_schema_version,
            )
            export_path = export_graph_bundle(target_dir, "canonical", bundle)
            exported_paths.append(str(export_path))
    return {
        "schema_version": config.future_projection.graph_sidecars.export_schema_version,
        "exports": len(exported_paths),
        "paths": exported_paths,
    }


async def forecast_readiness_report(
    store: KnowledgeStore,
    config: NexusConfig,
    *,
    start: date,
    end: date,
    profile: str = "signal-rich",
    base_dir: Path,
) -> dict:
    """Report whether graph and Kalshi benchmark seams are ready to enable later."""
    benchmark_first = await benchmark_forecast_engines(
        store,
        config,
        start=start,
        end=end,
        engines=["baseline", "trajectory", "native"],
        llm=None,
        mode="replay",
        profile=profile,
        strict=True,
    )
    benchmark_second = await benchmark_forecast_engines(
        store,
        config,
        start=start,
        end=end,
        engines=["baseline", "trajectory", "native"],
        llm=None,
        mode="replay",
        profile=profile,
        strict=True,
    )
    graph_dir = base_dir / "graph"
    graph_export = await export_graph_bundles(
        store,
        config,
        start=start,
        end=end,
        profile=profile,
        target_dir=graph_dir,
    )

    graph_probe_results: dict[str, dict] = {}
    exported_paths = [Path(path) for path in graph_export["paths"]]
    if exported_paths:
        sample_bundle = json.loads(exported_paths[0].read_text())
        from nexus.engine.projection.models import GraphExportBundle  # local import to avoid circular loading during tests

        bundle = GraphExportBundle(**sample_bundle)
        for adapter_name in config.future_projection.graph_sidecars.adapters:
            result = await get_graph_evidence_adapter(adapter_name).query(
                bundle,
                max_evidence_ids=config.future_projection.graph_sidecars.max_evidence_ids,
            )
            graph_probe_results[adapter_name] = result.model_dump(mode="json")
    else:
        for adapter_name in config.future_projection.graph_sidecars.adapters:
            graph_probe_results[adapter_name] = {
                "adapter": adapter_name,
                "status": "skipped",
                "metadata": {"reason": "no graph bundles exported"},
            }

    kalshi_cfg = config.future_projection.kalshi
    kalshi_client = KalshiClient(kalshi_cfg)
    mapping_path = Path(kalshi_cfg.mapping_file)
    kalshi_mappings = load_kalshi_mappings(mapping_path)
    kalshi_auth = await kalshi_client.auth_check()
    kalshi_ledger = KalshiLedger(Path(kalshi_cfg.ledger_path))
    await kalshi_ledger.initialize()
    try:
        kalshi_counts = await kalshi_ledger.counts()
        if kalshi_mappings and kalshi_counts.get("snapshots", 0) > 0:
            kalshi_compare = await compare_forecasts_to_kalshi(
                store,
                kalshi_ledger,
                start=start,
                end=end,
                mapping_path=mapping_path,
            )
        else:
            kalshi_compare = {
                "meta": {
                    "start": start.isoformat(),
                    "end": end.isoformat(),
                    "mapped_forecasts": 0,
                    "resolved_mapped_forecasts": 0,
                    "mapping_file": str(mapping_path),
                    "forecast_time_convention": "end_of_day",
                },
                "rows": [],
                "summary": {
                    "mean_probability_gap": 0.0,
                    "our_mean_brier": 0.0,
                    "market_mean_brier": 0.0,
                },
            }
    finally:
        await kalshi_ledger.close()

    readiness_flags = {
        "benchmark_trusted": benchmark_first == benchmark_second,
        "graph_export_ready": graph_export["exports"] > 0,
        "graph_sidecar_ready": any(
            result.get("status") == "ready" for result in graph_probe_results.values()
        ),
        "kalshi_adapter_ready": kalshi_client.auth_capable(),
        "kalshi_mapping_ready": bool(kalshi_mappings),
        "kalshi_auth_ready": bool(kalshi_auth["auth_success"]),
        "kalshi_compare_ready": kalshi_compare["meta"]["mapped_forecasts"] > 0,
    }
    readiness_flags["ready_to_enable"] = all((
        readiness_flags["benchmark_trusted"],
        readiness_flags["graph_export_ready"],
        readiness_flags["graph_sidecar_ready"],
        readiness_flags["kalshi_adapter_ready"],
        readiness_flags["kalshi_mapping_ready"],
        readiness_flags["kalshi_auth_ready"],
        readiness_flags["kalshi_compare_ready"],
    ))

    return {
        "meta": {
            "start": start.isoformat(),
            "end": end.isoformat(),
            "profile": profile,
        },
        "readiness": readiness_flags,
        "benchmark": benchmark_first,
        "graph_export": graph_export,
        "graph_sidecars": graph_probe_results,
        "kalshi": {
            "configured": kalshi_client.credentials_available(),
            "auth_capable": kalshi_client.auth_capable(),
            "auth_check": kalshi_auth,
            "mapping_file": str(mapping_path),
            "mapping_count": len(kalshi_mappings),
            "ledger_path": kalshi_cfg.ledger_path,
            "ledger_counts": kalshi_counts,
            "compare": kalshi_compare,
        },
    }


async def generate_prediction_audit(
    store: KnowledgeStore,
    config: NexusConfig,
    *,
    start: date,
    end: date,
    engines: list[str],
    profile: str = "signal-rich",
) -> dict:
    """Produce a detailed per-question prediction audit with full context.

    Returns per-question detail, per-topic breakdown, per-target-variable breakdown,
    calibration curve, and best/worst predictions.
    """
    report = await benchmark_forecast_engines(
        store, config,
        start=start, end=end,
        engines=engines, llm=None,
        mode="replay", profile=profile,
        strict=True,
    )

    # Re-run to collect per-question detail (benchmark only returns aggregates)
    per_question: list[dict] = []
    for engine_name in engines:
        engine = get_forecast_engine(engine_name)
        for topic in config.topics:
            topic_slug = topic_slug_from_name(topic.name)
            synthesis_dates = sorted([
                date.fromisoformat(raw)
                for raw in await store.get_synthesis_dates(topic_slug)
                if start <= date.fromisoformat(raw) <= end
            ])
            for cutoff in synthesis_dates:
                state = await load_historical_topic_state(
                    store, topic_slug=topic_slug, topic_name=topic.name,
                    cutoff=cutoff, config=config.future_projection, profile=profile,
                )
                if not state:
                    continue
                payload = await _build_forecast_input(
                    store, state.synthesis,
                    topic_slug=topic_slug, run_date=cutoff,
                    cross_topic_signals=state.cross_topic_signals,
                    metadata=state.metadata,
                )
                calibration_data = await store.get_historical_calibration(as_of=cutoff)
                forecast_run = await engine.generate(
                    None, payload, critic_pass=False,
                    calibration_data=calibration_data or None,
                )
                for question in forecast_run.questions:
                    resolution = await resolve_forecast_question(
                        store,
                        {
                            "forecast_question_id": -1,
                            "topic_slug": topic_slug,
                            "generated_for": cutoff.isoformat(),
                            "resolution_date": question.resolution_date.isoformat(),
                            "horizon_days": question.horizon_days,
                            "probability": question.probability,
                            "target_variable": question.target_variable,
                            "target_metadata": question.target_metadata,
                            "expected_direction": question.expected_direction,
                        },
                    )
                    if resolution.outcome_status != "resolved":
                        continue
                    per_question.append({
                        "engine": engine_name,
                        "topic_slug": topic_slug,
                        "cutoff": cutoff.isoformat(),
                        "question": question.question,
                        "target_variable": question.target_variable,
                        "probability": question.probability,
                        "base_rate": question.base_rate,
                        "resolved_bool": bool(resolution.resolved_bool),
                        "brier_score": resolution.brier_score or 0.0,
                        "resolution_date": question.resolution_date.isoformat(),
                        "horizon_days": question.horizon_days,
                        "signpost": question.signpost,
                        "notes": resolution.notes,
                    })

    # Per-topic breakdown
    by_topic: dict[str, list[dict]] = defaultdict(list)
    for q in per_question:
        by_topic[q["topic_slug"]].append(q)
    topic_breakdown = {
        slug: {
            "total": len(qs),
            "accuracy": round(sum(1 for q in qs if q["resolved_bool"]) / len(qs), 3) if qs else 0.0,
            "mean_brier": round(sum(q["brier_score"] for q in qs) / len(qs), 4) if qs else 0.0,
        }
        for slug, qs in by_topic.items()
    }

    # Per-target-variable breakdown
    by_target: dict[str, list[dict]] = defaultdict(list)
    for q in per_question:
        by_target[q["target_variable"]].append(q)
    target_breakdown = {
        tv: {
            "total": len(qs),
            "accuracy": round(sum(1 for q in qs if q["resolved_bool"]) / len(qs), 3) if qs else 0.0,
            "mean_brier": round(sum(q["brier_score"] for q in qs) / len(qs), 4) if qs else 0.0,
        }
        for tv, qs in by_target.items()
    }

    # Best/worst predictions (by Brier score)
    sorted_by_brier = sorted(per_question, key=lambda q: q["brier_score"])
    best = sorted_by_brier[:5]
    worst = sorted_by_brier[-5:] if len(sorted_by_brier) >= 5 else sorted_by_brier

    return {
        "meta": {
            "start": start.isoformat(),
            "end": end.isoformat(),
            "engines": engines,
            "total_questions": len(per_question),
        },
        "benchmark_summary": report,
        "per_question": per_question,
        "topic_breakdown": topic_breakdown,
        "target_variable_breakdown": target_breakdown,
        "best_predictions": best,
        "worst_predictions": worst,
    }


def render_prediction_audit_markdown(audit: dict) -> str:
    """Render a human-readable prediction audit report from audit data."""
    lines = [
        "# Prediction Audit Report",
        "",
        f"**Period**: {audit['meta']['start']} to {audit['meta']['end']}",
        f"**Engines**: {', '.join(audit['meta']['engines'])}",
        f"**Total resolved questions**: {audit['meta']['total_questions']}",
        "",
    ]

    # Benchmark summary table
    summary = audit.get("benchmark_summary", {})
    engines = summary.get("engines", {})
    if engines:
        lines.extend([
            "## Engine Performance Summary",
            "",
            "| Engine | Total | Accuracy | Mean Brier | Mean Log Loss |",
            "| --- | ---: | ---: | ---: | ---: |",
        ])
        for eng_name, eng_data in engines.items():
            lines.append(
                f"| `{eng_name}` | {eng_data['total']} | {eng_data['accuracy']} | "
                f"{eng_data['mean_brier']} | {eng_data['mean_log_loss']} |"
            )
        sig = summary.get("meta", {}).get("significance_test")
        if sig:
            lines.extend([
                "",
                f"**Significance test**: p={sig.get('p_value', 'N/A')}, "
                f"n={sig.get('n', 0)}, "
                f"{'significant' if sig.get('significant_at_005') else 'not significant'} at p<0.05",
            ])
        lines.append("")

    # Per-topic breakdown
    topic_bd = audit.get("topic_breakdown", {})
    if topic_bd:
        lines.extend([
            "## Per-Topic Breakdown",
            "",
            "| Topic | Total | Accuracy | Mean Brier |",
            "| --- | ---: | ---: | ---: |",
        ])
        for slug, data in sorted(topic_bd.items()):
            lines.append(f"| `{slug}` | {data['total']} | {data['accuracy']} | {data['mean_brier']} |")
        lines.append("")

    # Per-target breakdown
    target_bd = audit.get("target_variable_breakdown", {})
    if target_bd:
        lines.extend([
            "## Per-Target Variable Breakdown",
            "",
            "| Target Variable | Total | Accuracy | Mean Brier |",
            "| --- | ---: | ---: | ---: |",
        ])
        for tv, data in sorted(target_bd.items()):
            lines.append(f"| `{tv}` | {data['total']} | {data['accuracy']} | {data['mean_brier']} |")
        lines.append("")

    # Best predictions
    best = audit.get("best_predictions", [])
    if best:
        lines.extend(["## Best Predictions (lowest Brier)", ""])
        for i, q in enumerate(best, 1):
            lines.extend([
                f"**{i}. [{q['topic_slug']}] {q['question'][:120]}**",
                f"  - Predicted: {q['probability']:.0%} | Outcome: {'Yes' if q['resolved_bool'] else 'No'} | Brier: {q['brier_score']:.4f}",
                f"  - Engine: {q['engine']} | Cutoff: {q['cutoff']} | Horizon: {q['horizon_days']}d",
                "",
            ])

    # Worst predictions
    worst = audit.get("worst_predictions", [])
    if worst:
        lines.extend(["## Worst Predictions (highest Brier)", ""])
        for i, q in enumerate(worst, 1):
            lines.extend([
                f"**{i}. [{q['topic_slug']}] {q['question'][:120]}**",
                f"  - Predicted: {q['probability']:.0%} | Outcome: {'Yes' if q['resolved_bool'] else 'No'} | Brier: {q['brier_score']:.4f}",
                f"  - Engine: {q['engine']} | Cutoff: {q['cutoff']} | Horizon: {q['horizon_days']}d",
                "",
            ])

    return "\n".join(lines)
