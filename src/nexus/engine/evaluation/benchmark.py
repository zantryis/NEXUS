"""Multi-axis benchmark — pipeline vs naive, style variants, topic diversity.

Usage:
    python -m nexus benchmark
    python -m nexus benchmark --topics space,cyber
    python -m nexus benchmark --styles analytical,editorial
"""

import json
import logging
import random
import time
from dataclasses import dataclass, field, asdict
from datetime import date, datetime, timedelta
from pathlib import Path

from nexus.config.models import NexusConfig, TopicConfig
from nexus.engine.evaluation.judge import judge_synthesis
from nexus.engine.knowledge.events import Event
from nexus.engine.sources.polling import ContentItem
from nexus.engine.synthesis.knowledge import (
    TopicSynthesis, NarrativeThread, Summary, synthesize_topic,
)
from nexus.engine.synthesis.renderers import render_text_briefing
from nexus.llm.client import LLMClient

logger = logging.getLogger(__name__)


# ── Text quality rubric (separate from synthesis quality) ────────────────────

BRIEFING_JUDGE_PROMPT = (
    "You are an expert editor evaluating a news briefing.\n"
    "Score the TEXT using the anchor definitions below.\n\n"
    "## 1. Clarity (2-10)\n"
    "- **2**: Dense, jargon-heavy, requires re-reading to parse\n"
    "- **4**: Understandable but poorly organized, hard to scan\n"
    "- **6**: Clear writing, reasonable structure, some dense sections\n"
    "- **8**: Easy to scan, good headers, key info immediately visible\n"
    "- **10**: Exemplary — scannable in 30s, every sentence earns its place\n\n"
    "## 2. Insight Density (2-10)\n"
    "- **2**: Mostly background/filler, <1 new fact per paragraph\n"
    "- **4**: Some new info but padded with obvious context\n"
    "- **6**: Solid info density, 1-2 insights per paragraph\n"
    "- **8**: High density — nearly every sentence adds new info\n"
    "- **10**: Exceptional — every paragraph delivers non-obvious insights\n\n"
    "## 3. Source Attribution (2-10)\n"
    "- **2**: No sources cited, claims float without attribution\n"
    "- **4**: Occasional attribution but most claims unattributed\n"
    "- **6**: Key claims attributed, but attribution inconsistent\n"
    "- **8**: Claims consistently attributed without cluttering the text\n"
    "- **10**: Clean attribution — sources named where needed, not over-attributed\n\n"
    "## 4. Narrative Coherence (2-10)\n"
    "- **2**: Disjointed fact list, no narrative arc\n"
    "- **4**: Loosely connected paragraphs, weak transitions\n"
    "- **6**: Clear topic grouping, stories have basic structure\n"
    "- **8**: Strong narrative — stories flow logically, connections drawn\n"
    "- **10**: Compelling — reads like expert analysis, not a dump\n\n"
    "## 5. Actionability (2-10)\n"
    "- **2**: Pure background, a decision-maker learns nothing new\n"
    "- **4**: Some news but implications not drawn\n"
    "- **6**: Developments reported with basic context for action\n"
    "- **8**: Clear implications — reader knows what to watch/do next\n"
    "- **10**: Fully actionable — risks, opportunities, and timeline clear\n\n"
    "Respond with JSON:\n"
    '{"clarity": <int>, "insight_density": <int>, "source_attribution": <int>,\n'
    ' "narrative_coherence": <int>, "actionability": <int>, "overall": <float>,\n'
    ' "notes": "..."}\n\n'
    "IMPORTANT: Use the FULL range. A raw article list should score 2-4. "
    "A good newsletter scores 6-8. Only an exceptional briefing deserves 9-10."
)


# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class StyleResult:
    style: str
    briefing_text: str
    scores: dict = field(default_factory=dict)


@dataclass
class TopicBenchmark:
    topic_name: str
    topic_slug: str
    nexus_scores: dict = field(default_factory=dict)
    naive_scores: dict = field(default_factory=dict)
    style_results: list[StyleResult] = field(default_factory=list)
    improvement_pct: float = 0.0
    funnel: dict = field(default_factory=dict)
    source_count: int = 0
    article_count: int = 0
    event_count: int = 0


@dataclass
class BenchmarkReport:
    topics: list[TopicBenchmark] = field(default_factory=list)
    aggregate: dict = field(default_factory=dict)
    style_ranking: list[tuple] = field(default_factory=list)
    cost: dict = field(default_factory=dict)
    timestamp: str = ""
    judge_model: str = ""
    duration_s: float = 0.0

    def to_json(self) -> dict:
        return asdict(self)

    def to_markdown(self) -> str:
        lines = [
            f"# Nexus Benchmark Report",
            f"**Date**: {self.timestamp}",
            f"**Judge model**: {self.judge_model}",
            f"**Duration**: {self.duration_s:.0f}s",
            "",
        ]

        # Aggregate
        if self.aggregate:
            lines.append("## Aggregate Scores")
            lines.append("")
            lines.append("| Metric | Nexus | Naive | Improvement |")
            lines.append("|--------|-------|-------|-------------|")
            nexus_agg = self.aggregate.get("nexus", {})
            naive_agg = self.aggregate.get("naive", {})
            for dim in ["completeness", "source_balance", "convergence_accuracy",
                        "divergence_detection", "entity_coverage", "overall"]:
                n = nexus_agg.get(dim, 0)
                b = naive_agg.get(dim, 0)
                pct = ((n - b) / b * 100) if b else 0
                lines.append(f"| {dim} | {n:.1f} | {b:.1f} | {pct:+.0f}% |")
            lines.append("")

        # Style ranking
        if self.style_ranking:
            lines.append("## Style Ranking (by text quality)")
            lines.append("")
            lines.append("| Rank | Style | Overall |")
            lines.append("|------|-------|---------|")
            for i, (style, score) in enumerate(self.style_ranking, 1):
                lines.append(f"| {i} | {style} | {score:.1f} |")
            lines.append("")

        # Per-topic details
        for tb in self.topics:
            lines.append(f"## {tb.topic_name}")
            lines.append(f"- Articles: {tb.article_count}, Events: {tb.event_count}, Sources: {tb.source_count}")
            if tb.funnel:
                funnel_str = " → ".join(f"{k}: {v}" for k, v in tb.funnel.items())
                lines.append(f"- Funnel: {funnel_str}")
            lines.append(f"- Improvement over naive: **{tb.improvement_pct:+.0f}%**")
            lines.append("")

            if tb.nexus_scores and tb.naive_scores:
                lines.append("| Dimension | Nexus | Naive |")
                lines.append("|-----------|-------|-------|")
                for dim in ["completeness", "source_balance", "convergence_accuracy",
                            "divergence_detection", "entity_coverage", "overall"]:
                    n = tb.nexus_scores.get(dim, 0)
                    b = tb.naive_scores.get(dim, 0)
                    lines.append(f"| {dim} | {n} | {b} |")
                lines.append("")

            if tb.nexus_scores.get("strengths"):
                lines.append("**Strengths**: " + ", ".join(tb.nexus_scores["strengths"]))
            if tb.nexus_scores.get("weaknesses"):
                lines.append("**Weaknesses**: " + ", ".join(tb.nexus_scores["weaknesses"]))

            if tb.style_results:
                lines.append("")
                lines.append("### Style Scores")
                lines.append("| Style | Clarity | Insight | Attribution | Coherence | Action | Overall |")
                lines.append("|-------|---------|---------|-------------|-----------|--------|---------|")
                for sr in tb.style_results:
                    s = sr.scores
                    lines.append(
                        f"| {sr.style} | {s.get('clarity', '-')} | {s.get('insight_density', '-')} | "
                        f"{s.get('source_attribution', '-')} | {s.get('narrative_coherence', '-')} | "
                        f"{s.get('actionability', '-')} | {s.get('overall', '-')} |"
                    )
            lines.append("")

        # Cost
        if self.cost:
            lines.append("## Cost")
            tokens_in = self.cost.get("total_input_tokens", 0)
            tokens_out = self.cost.get("total_output_tokens", 0)
            cost_usd = self.cost.get("total_cost_usd", 0)
            lines.append(f"- Input tokens: {tokens_in:,}")
            lines.append(f"- Output tokens: {tokens_out:,}")
            lines.append(f"- Estimated cost: ${cost_usd:.4f}")

        return "\n".join(lines)


# ── Naive baseline builder ───────────────────────────────────────────────────

def build_naive_synthesis(
    topic_name: str,
    events: list[Event],
    max_events: int = 30,
) -> TopicSynthesis:
    """Build a no-synthesis baseline: real events, no thread grouping or analysis.

    Uses real extracted events (with dates, entities, summaries) but skips
    synthesis: no thread grouping, no convergence, no divergence.
    This isolates the value added by the synthesis LLM call.
    """
    sorted_events = sorted(events, key=lambda e: e.date, reverse=True)[:max_events]

    # One thread per event — no grouping
    threads = [
        NarrativeThread(
            headline=e.summary[:80],
            events=[e],
            convergence=[],
            divergence=[],
            key_entities=e.entities,
            significance=e.significance,
        )
        for e in sorted_events
    ]

    source_balance: dict[str, int] = {}
    languages: set[str] = set()
    for e in sorted_events:
        for s in e.sources:
            affil = s.get("affiliation", "unknown")
            source_balance[affil] = source_balance.get(affil, 0) + 1
            lang = s.get("language")
            if lang:
                languages.add(lang)

    return TopicSynthesis(
        topic_name=topic_name,
        threads=threads,
        background=[],
        source_balance=source_balance,
        languages_represented=sorted(languages),
        metadata={"naive": True, "event_count": len(sorted_events)},
    )


# ── Text quality judge ───────────────────────────────────────────────────────

async def judge_briefing_text(
    llm: LLMClient, briefing_text: str, model_override: str | None = None,
) -> dict:
    """Judge a rendered briefing on text quality.

    If model_override is set, temporarily swaps the agent model for this call.
    """
    original_model = None
    if model_override and hasattr(llm, "_config"):
        original_model = llm._config.agent
        llm._config.agent = model_override
    try:
        response = await llm.complete(
            config_key="agent",
            system_prompt=BRIEFING_JUDGE_PROMPT,
            user_prompt=briefing_text,
            json_response=True,
        )
        scores = json.loads(response)
        if "overall" not in scores:
            dims = ["clarity", "insight_density", "source_attribution",
                    "narrative_coherence", "actionability"]
            vals = [scores.get(d, 5) for d in dims]
            scores["overall"] = round(sum(vals) / len(vals), 1)
        return scores
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        logger.warning(f"Briefing judge failed: {e}")
        return {"error": str(e)}
    finally:
        if original_model is not None:
            llm._config.agent = original_model


# ── Funnel stats ─────────────────────────────────────────────────────────────

async def get_funnel_stats(store, topic_slug: str, run_date: str) -> dict:
    """Query filter_log for funnel statistics."""
    try:
        async with store._db_connection() as db:
            cursor = await db.execute(
                "SELECT outcome, COUNT(*) FROM filter_log "
                "WHERE run_date = ? AND topic_slug = ? GROUP BY outcome",
                (run_date, topic_slug),
            )
            rows = await cursor.fetchall()
            return {row[0]: row[1] for row in rows}
    except Exception:
        return {}


# ── Main benchmark runner ────────────────────────────────────────────────────

async def run_benchmark(
    config: NexusConfig,
    llm: LLMClient,
    data_dir: Path,
    topics: list[str] | None = None,
    styles: list[str] | None = None,
    judge_model: str | None = None,
) -> BenchmarkReport:
    """Run a multi-axis benchmark across topics and styles.

    Steps per topic:
    1. Poll sources → get raw articles
    2. Run full pipeline path → filter → extract → synthesize → TopicSynthesis
    3. Build naive baseline from same articles
    4. Judge both syntheses (synthesis quality)
    5. Render briefings in each style, judge text quality
    6. Collect funnel stats
    """
    from nexus.engine.knowledge.store import KnowledgeStore
    from nexus.engine.sources.polling import poll_all_feeds, filter_recent
    from nexus.engine.ingestion.ingest import async_ingest_items
    from nexus.engine.ingestion.dedup import dedup_items
    from nexus.engine.filtering.filter import filter_items

    if styles is None:
        styles = ["analytical", "conversational", "editorial"]

    start_time = time.monotonic()
    report = BenchmarkReport(
        timestamp=datetime.now().isoformat(),
        judge_model=judge_model or "default (agent config)",
    )

    # Filter topics if specified
    topic_configs = config.topics
    if topics:
        topic_configs = [
            t for t in config.topics
            if t.name.lower().replace(" ", "-").replace("/", "-") in topics
            or t.name.lower() in [s.lower() for s in topics]
        ]

    if not topic_configs:
        logger.error("No matching topics found")
        return report

    # Initialize store for funnel stats
    store = KnowledgeStore(data_dir / "knowledge.db")
    await store.initialize()
    await llm.set_store(store)

    try:
        for topic_cfg in topic_configs:
            slug = topic_cfg.name.lower().replace(" ", "-").replace("/", "-")
            logger.info(f"Benchmarking topic: {topic_cfg.name}")

            tb = TopicBenchmark(
                topic_name=topic_cfg.name,
                topic_slug=slug,
            )

            # Step 1: Poll sources
            registry_path = data_dir / "sources" / slug / "registry.yaml"
            if not registry_path.exists():
                logger.warning(f"No source registry for {slug}, skipping")
                report.topics.append(tb)
                continue

            import yaml
            reg_data = yaml.safe_load(registry_path.read_text()) or {}
            sources_list = reg_data.get("sources", [])
            if not sources_list:
                logger.warning(f"Empty registry for {slug}")
                report.topics.append(tb)
                continue

            raw_items = poll_all_feeds(sources_list)
            recent = filter_recent(raw_items, max_age_hours=168)  # 7 days
            tb.source_count = len(set(a.source_id for a in recent))

            if not recent:
                logger.warning(f"No recent articles for {slug}")
                report.topics.append(tb)
                continue

            # Step 2: Full pipeline path
            # Ingest (extract full text) — cap at 50 articles for speed
            to_ingest = recent[:50]
            ingested = await async_ingest_items(to_ingest)
            deduped = dedup_items(ingested)
            tb.article_count = len(deduped)

            # Filter (two-pass)
            filter_result = await filter_items(llm, deduped, topic_cfg)
            passed = filter_result.accepted

            # Extract events from passed articles
            from nexus.engine.knowledge.events import extract_event, is_duplicate_event
            events = []
            for article in passed:
                ev = await extract_event(llm, article, topic_cfg, events)
                if ev and not any(is_duplicate_event(ev, e) for e in events):
                    events.append(ev)

            tb.event_count = len(events)

            # Synthesize
            nexus_synthesis = await synthesize_topic(
                llm, topic_cfg, events, deduped,
                weekly_summaries=[], monthly_summaries=[],
                store=store, topic_slug=slug,
            )

            # Step 3: Naive baseline (no-synthesis: real events, no thread grouping)
            naive_synthesis = build_naive_synthesis(topic_cfg.name, events)

            # Step 4: Judge both syntheses
            logger.info(f"  Judging Nexus synthesis for {topic_cfg.name}")
            tb.nexus_scores = await judge_synthesis(llm, nexus_synthesis)

            logger.info(f"  Judging naive baseline for {topic_cfg.name}")
            tb.naive_scores = await judge_synthesis(llm, naive_synthesis)

            # Compute improvement
            nexus_overall = tb.nexus_scores.get("overall", 0)
            naive_overall = tb.naive_scores.get("overall", 0)
            if naive_overall > 0:
                tb.improvement_pct = (nexus_overall - naive_overall) / naive_overall * 100
            else:
                tb.improvement_pct = 0

            # Step 5: Render briefings in each style and judge text quality
            for style in styles:
                logger.info(f"  Rendering + judging style: {style}")
                # Temporarily override style
                original_style = config.briefing.style
                config.briefing.style = style
                try:
                    briefing_text = await render_text_briefing(
                        llm, config, [nexus_synthesis],
                    )
                    text_scores = await judge_briefing_text(llm, briefing_text)
                    tb.style_results.append(StyleResult(
                        style=style,
                        briefing_text=briefing_text,
                        scores=text_scores,
                    ))
                except Exception as e:
                    logger.warning(f"  Style {style} failed: {e}")
                    tb.style_results.append(StyleResult(style=style, briefing_text="", scores={"error": str(e)}))
                finally:
                    config.briefing.style = original_style

            # Step 6: Funnel stats
            tb.funnel = await get_funnel_stats(store, slug, date.today().isoformat())

            report.topics.append(tb)

        # Aggregate
        report.aggregate = _compute_aggregate(report.topics)
        report.style_ranking = _compute_style_ranking(report.topics)
        report.duration_s = time.monotonic() - start_time

        # Cost info
        if hasattr(llm, "usage"):
            usage_summary = llm.usage.summary()
            cost_summary = llm.usage.cost_summary()
            report.cost = {
                "total_input_tokens": usage_summary.get("total_input_tokens", 0),
                "total_output_tokens": usage_summary.get("total_output_tokens", 0),
                "total_cost_usd": cost_summary.get("total_cost_usd", 0.0),
            }

    finally:
        await store.close()

    return report


def _compute_aggregate(topics: list[TopicBenchmark]) -> dict:
    """Average scores across all topics."""
    dims = ["completeness", "source_balance", "convergence_accuracy",
            "divergence_detection", "entity_coverage", "overall"]
    nexus_avg = {}
    naive_avg = {}

    valid = [t for t in topics if t.nexus_scores and "error" not in t.nexus_scores]
    if not valid:
        return {}

    for dim in dims:
        nexus_vals = [t.nexus_scores.get(dim, 0) for t in valid]
        naive_vals = [t.naive_scores.get(dim, 0) for t in valid]
        nexus_avg[dim] = round(sum(nexus_vals) / len(nexus_vals), 1)
        naive_avg[dim] = round(sum(naive_vals) / len(naive_vals), 1) if naive_vals else 0

    return {"nexus": nexus_avg, "naive": naive_avg}


def _compute_style_ranking(topics: list[TopicBenchmark]) -> list[tuple]:
    """Rank styles by average text quality across topics."""
    style_scores: dict[str, list[float]] = {}
    for t in topics:
        for sr in t.style_results:
            if sr.scores and "error" not in sr.scores:
                overall = sr.scores.get("overall", 0)
                style_scores.setdefault(sr.style, []).append(overall)

    ranking = []
    for style, scores in style_scores.items():
        avg = round(sum(scores) / len(scores), 1) if scores else 0
        ranking.append((style, avg))

    ranking.sort(key=lambda x: x[1], reverse=True)
    return ranking
