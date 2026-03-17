"""Budget-safe deterministic evaluation for stored projections.

Includes both a cheap LLM-based judge (preferred) and a deterministic keyword
fallback for when no LLM client is available or the LLM call fails.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from math import log
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path
from statistics import mean
from typing import Literal

from nexus.config.models import NexusConfig
from nexus.engine.knowledge.store import KnowledgeStore
from nexus.engine.projection.engines import ProjectionEngineInput, get_projection_engine
from nexus.engine.projection.graph import (
    build_graph_export_bundle,
    export_graph_bundle,
)
from nexus.engine.projection.models import ProjectionOutcome
from nexus.engine.projection.service import (
    hydrate_synthesis_threads,
    load_historical_topic_state,
    topic_slug_from_name,
)
from nexus.engine.synthesis.knowledge import TopicSynthesis

logger = logging.getLogger(__name__)


# ── Keyword fallback (deterministic, no LLM) ────────────────────────

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "for", "from", "in", "into", "is", "it",
    "likely", "next", "of", "on", "or", "over", "that", "the", "this", "to", "will", "with",
}


def _keywords(text: str) -> set[str]:
    return {
        token.lower() for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9'-]+", text)
        if len(token) > 2 and token.lower() not in _STOPWORDS
    }


# ── LLM-based projection judge ──────────────────────────────────────

VerdictLabel = Literal["confirms", "partially_confirms", "contradicts", "irrelevant"]

_VERDICT_SCORES: dict[VerdictLabel, float | None] = {
    "confirms": 1.0,
    "partially_confirms": 0.5,
    "contradicts": 0.0,
    "irrelevant": None,  # excluded from scoring
}

_JUDGE_SYSTEM_PROMPT = (
    "You are a precise evaluator of forward-looking intelligence claims.\n"
    "Given a CLAIM (a prediction about the future) and its SIGNPOST (the observable\n"
    "indicator that would confirm it), judge whether each subsequent real-world EVENT\n"
    "confirms, partially confirms, contradicts, or is irrelevant to the claim.\n\n"
    "## Verdict definitions\n"
    "- **confirms**: The event provides direct evidence that the claim is happening or has happened.\n"
    "- **partially_confirms**: The event provides indirect or partial evidence. The claim may be\n"
    "  partially true but the evidence is incomplete or the scope differs.\n"
    "- **contradicts**: The event provides evidence AGAINST the claim. The opposite of what was\n"
    "  predicted is occurring. Pay close attention to NEGATION: 'withdraws from talks' is the\n"
    "  opposite of 'enters talks' even though they share the same keywords.\n"
    "- **irrelevant**: The event has no meaningful bearing on the claim.\n\n"
    "## Critical instructions\n"
    "- Focus on MEANING, not keyword overlap. Two events can share every keyword but have\n"
    "  opposite meanings (e.g., 'bans exports' vs 'lifts export ban').\n"
    "- Consider the DIRECTION of actions: escalation vs de-escalation, increase vs decrease,\n"
    "  joining vs leaving, starting vs stopping.\n"
    "- Be conservative with 'confirms' — only use it when the event directly supports the\n"
    "  specific claim, not just a vaguely related topic.\n\n"
    "Respond with JSON:\n"
    '{"verdicts": [\n'
    '  {"event_index": 0, "verdict": "confirms|partially_confirms|contradicts|irrelevant",\n'
    '   "rationale": "One sentence explaining why"}\n'
    "]}\n"
)


@dataclass
class EventVerdict:
    """LLM judgment for a single event against a projection claim."""

    event_index: int
    verdict: VerdictLabel
    rationale: str = ""


@dataclass
class ProjectionJudgment:
    """Aggregate judgment for a projection item against future events."""

    score: float
    outcome_status: Literal["hit", "mixed", "miss"]
    verdicts: list[EventVerdict] = field(default_factory=list)
    relevant_count: int = 0
    used_fallback: bool = False


def _format_judge_prompt(claim: str, signpost: str, events: list) -> str:
    """Build the user prompt for the projection judge."""
    lines = [
        f"## CLAIM\n{claim}\n",
        f"## SIGNPOST\n{signpost}\n",
        "## EVENTS\n",
    ]
    for i, event in enumerate(events):
        event_date = getattr(event, "date", "?")
        summary = getattr(event, "summary", str(event))
        entities = getattr(event, "entities", [])
        entity_str = f" [{', '.join(entities)}]" if entities else ""
        lines.append(f"[{i}] ({event_date}){entity_str} {summary}")
    return "\n".join(lines)


def _score_to_status(score: float) -> Literal["hit", "mixed", "miss"]:
    if score >= 0.6:
        return "hit"
    if score >= 0.3:
        return "mixed"
    return "miss"


def _parse_verdicts(raw: str, event_count: int) -> list[EventVerdict]:
    """Parse LLM JSON response into EventVerdict list. Raises on failure."""
    parsed = json.loads(raw)
    if "verdicts" not in parsed:
        raise KeyError("Missing 'verdicts' key in LLM response")

    verdicts = []
    for item in parsed["verdicts"]:
        idx = item.get("event_index", -1)
        verdict_str = item.get("verdict", "irrelevant")
        if verdict_str not in _VERDICT_SCORES:
            verdict_str = "irrelevant"
        if idx < 0 or idx >= event_count:
            continue
        verdicts.append(EventVerdict(
            event_index=idx,
            verdict=verdict_str,
            rationale=item.get("rationale", ""),
        ))
    return verdicts


def _compute_score(verdicts: list[EventVerdict]) -> tuple[float, int]:
    """Compute aggregate score from verdicts. Returns (score, relevant_count)."""
    relevant_scores = []
    for v in verdicts:
        s = _VERDICT_SCORES.get(v.verdict)
        if s is not None:
            relevant_scores.append(s)
    if not relevant_scores:
        return 0.0, 0
    return round(sum(relevant_scores) / len(relevant_scores), 3), len(relevant_scores)


async def judge_projection_item(
    claim: str,
    signpost: str,
    future_events: list,
    llm,
) -> ProjectionJudgment:
    """Evaluate a projection claim against subsequent events using an LLM judge.

    Uses a single cheap LLM call (filtering-tier model) to classify each event as
    confirms/partially_confirms/contradicts/irrelevant. Falls back to keyword
    matching if the LLM call fails.
    """
    if not future_events:
        return ProjectionJudgment(score=0.0, outcome_status="miss")

    try:
        user_prompt = _format_judge_prompt(claim, signpost, future_events)
        response = await llm.complete(
            config_key="filtering",
            system_prompt=_JUDGE_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            json_response=True,
        )
        verdicts = _parse_verdicts(response, len(future_events))
        score, relevant_count = _compute_score(verdicts)
        return ProjectionJudgment(
            score=score,
            outcome_status=_score_to_status(score),
            verdicts=verdicts,
            relevant_count=relevant_count,
        )
    except Exception as exc:
        logger.warning("LLM judge failed, falling back to keyword matching: %s", exc)
        score = evaluate_projection_item(claim, signpost, future_events)
        return ProjectionJudgment(
            score=score,
            outcome_status=_score_to_status(score),
            used_fallback=True,
        )


async def auto_evaluate_projections(
    store: KnowledgeStore,
    *,
    start: date,
    end: date,
    engine: str | None = None,
    llm=None,
) -> dict:
    """Evaluate stored projections against subsequent events.

    If an LLM client is provided, uses the semantic LLM judge for accurate
    evaluation (handles negation, paraphrases, conceptual matches). Falls back
    to deterministic keyword matching when no LLM is available.
    """
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

        if llm is not None:
            judgment = await judge_projection_item(
                item["claim"], item["signpost"], future_events, llm,
            )
            score = judgment.score
            outcome_status = judgment.outcome_status
            method = "llm_judge" if not judgment.used_fallback else "keyword_fallback"
        else:
            score = evaluate_projection_item(item["claim"], item["signpost"], future_events)
            outcome_status = _score_to_status(score)
            method = "keyword"

        summary[item["engine"]]["total"] += 1
        summary[item["engine"]][outcome_status] += 1
        await store.set_projection_outcome(ProjectionOutcome(
            projection_item_id=item["projection_item_id"],
            outcome_status=outcome_status,
            score=score,
            notes=f"Auto-evaluated ({method}) against {len(future_events)} future events.",
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
    llm=None,
) -> dict:
    """Run engine comparisons on historical synthesis cutoffs.

    Uses LLM judge when available, keyword matching otherwise.
    """
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
                    if llm is not None:
                        judgment = await judge_projection_item(
                            item.claim, item.signpost, future_events, llm,
                        )
                        scores.append(judgment.score)
                    else:
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


def statistical_significance_test(
    native_briers: list[float],
    baseline_briers: list[float],
) -> dict:
    """Paired t-test on Brier scores between two engines.

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
        diffs = [a - b for a, b in zip(native_briers[:n], baseline_briers[:n])]
        mean_diff = sum(diffs) / n
        var_diff = sum((d - mean_diff) ** 2 for d in diffs) / (n - 1) if n > 1 else 0.0
        if var_diff == 0:
            result.update({"t_statistic": 0.0, "p_value": 1.0, "significant_at_005": False,
                            "note": "zero variance — scores identical"})
            return result
        se = (var_diff / n) ** 0.5
        t_stat = mean_diff / se
        from math import erfc
        p_approx = erfc(abs(t_stat) / (2 ** 0.5))
        result.update({
            "t_statistic": round(t_stat, 4),
            "p_value": round(p_approx, 6),
            "significant_at_005": bool(p_approx < 0.05),
            "note": "paired t-test (scipy unavailable, normal approx)",
        })
    return result


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
    """Audit known leakage vectors in the replay path."""
    topics_audited: set[str] = set()
    thread_state_leak_cutoffs = 0
    future_signal_leak_cutoffs = 0
    recent_events_leak_cutoffs = 0
    examples: list[dict] = []
    cutoffs_audited = 0

    for topic in config.topics:
        if not getattr(topic, "projection_eligible", True):
            continue
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

            # Check recent_events cutoff — events fed to evidence assembly
            # must not include anything after the cutoff date
            recent_events = await store.get_recent_events(
                topic_slug, days=14, limit=40, reference_date=cutoff,
            )
            leaked_events = [e for e in recent_events if e.date > cutoff]
            if leaked_events:
                recent_events_leak_cutoffs += 1
                if len(examples) < 10:
                    examples.append({
                        "type": "recent_events",
                        "topic_slug": topic_slug,
                        "cutoff": cutoff.isoformat(),
                        "leaked_event_dates": [
                            e.date.isoformat() if hasattr(e.date, "isoformat") else str(e.date)
                            for e in leaked_events[:5]
                        ],
                    })

    return {
        "cutoffs_audited": cutoffs_audited,
        "topics_audited": sorted(topics_audited),
        "thread_state_leak_cutoffs": thread_state_leak_cutoffs,
        "future_signal_leak_cutoffs": future_signal_leak_cutoffs,
        "recent_events_leak_cutoffs": recent_events_leak_cutoffs,
        "passes_strict_gate": (
            thread_state_leak_cutoffs == 0
            and future_signal_leak_cutoffs == 0
            and recent_events_leak_cutoffs == 0
        ),
        "examples": examples,
    }


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
        if not getattr(topic, "projection_eligible", True):
            continue
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
