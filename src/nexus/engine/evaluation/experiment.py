"""Experiment suite — controlled benchmarks for README claims.

Runs multiple experiment suites (A-G) across topics with article caching,
parameter sweeps, multi-judge validation, and statistical reporting.

Usage:
    python -m nexus experiment
    python -m nexus experiment --suite A,B,G --topics iran,ai --budget 15
"""

import copy
import json
import logging
import math
import random
import statistics
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

from nexus.config.models import ModelsConfig, NexusConfig, TopicConfig
from nexus.engine.evaluation.benchmark import (
    StyleResult, build_naive_synthesis, judge_briefing_text,
)
from nexus.engine.evaluation.judge import judge_synthesis
from nexus.engine.knowledge.events import Event
from nexus.engine.sources.polling import ContentItem
from nexus.engine.synthesis.knowledge import TopicSynthesis
from nexus.llm.client import LLMClient

logger = logging.getLogger(__name__)


# ── Data Classes ─────────────────────────────────────────────────────────────

@dataclass
class CachedTopicData:
    """Articles polled and ingested once, reused across all experiment variants."""
    topic_cfg: TopicConfig
    slug: str
    raw_articles: list[ContentItem] = field(default_factory=list)
    ingested_articles: list[ContentItem] = field(default_factory=list)
    recent_events: list[Event] = field(default_factory=list)


@dataclass
class VariantResult:
    """Result of running one pipeline variant on one topic."""
    variant_name: str
    topic_slug: str
    synthesis: TopicSynthesis | None = None
    scores: dict = field(default_factory=dict)  # {judge_label: {dim: score}}
    style_results: list = field(default_factory=list)
    filter_stats: dict = field(default_factory=dict)
    params: dict = field(default_factory=dict)
    cost_usd: float = 0.0
    duration_s: float = 0.0


@dataclass
class SuiteReport:
    """Report for a single experiment suite."""
    suite_id: str
    description: str
    claim: str
    results: list[VariantResult] = field(default_factory=list)
    stats: dict = field(default_factory=dict)


@dataclass
class ExperimentReport:
    """Full experiment report with statistical analysis."""
    suites: dict = field(default_factory=dict)  # {id: SuiteReport}
    total_cost: dict = field(default_factory=dict)
    timestamp: str = ""
    duration_s: float = 0.0
    limitations: list = field(default_factory=list)

    def to_json(self) -> dict:
        result = {
            "timestamp": self.timestamp,
            "duration_s": self.duration_s,
            "total_cost": self.total_cost,
            "limitations": self.limitations,
            "suites": {},
        }
        for sid, sr in self.suites.items():
            result["suites"][sid] = {
                "suite_id": sr.suite_id,
                "description": sr.description,
                "claim": sr.claim,
                "stats": sr.stats,
                "results": [asdict(r) for r in sr.results
                            if not hasattr(r, "synthesis") or True],
            }
            # Strip synthesis objects from JSON (too large)
            for r in result["suites"][sid]["results"]:
                r.pop("synthesis", None)
        return result

    def to_markdown(self) -> str:
        """Full detailed report for docs/benchmark-results.md."""
        lines = [
            "# Nexus Experiment Report",
            f"**Date**: {self.timestamp}",
            f"**Duration**: {self.duration_s:.0f}s",
            f"**Cost**: Gemini ${self.total_cost.get('gemini', 0):.2f}, "
            f"DeepSeek ${self.total_cost.get('deepseek', 0):.2f}",
            "",
        ]

        for sid, sr in self.suites.items():
            lines.append(f"## Suite {sid}: {sr.description}")
            lines.append(f"**Claim**: {sr.claim}")
            lines.append("")

            if sr.stats:
                # Build a table from stats
                variants = list(sr.stats.keys())
                if variants:
                    dims = list(sr.stats[variants[0]].keys())
                    lines.append("| Variant | " + " | ".join(dims) + " |")
                    lines.append("|---------|" + "|".join(["------"] * len(dims)) + "|")
                    for v in variants:
                        cells = []
                        for d in dims:
                            s = sr.stats[v].get(d, {})
                            if isinstance(s, dict) and "mean" in s:
                                cells.append(f"{s['mean']:.1f} +/- {s.get('std', 0):.1f}")
                            else:
                                cells.append(str(s))
                        lines.append(f"| {v} | " + " | ".join(cells) + " |")
                    lines.append("")

        if self.limitations:
            lines.append("## Limitations")
            for lim in self.limitations:
                lines.append(f"- {lim}")
            lines.append("")

        return "\n".join(lines)

    def to_readme_snippet(self) -> str:
        """Compact summary for README.md."""
        lines = [
            "### Pipeline Quality",
            "",
        ]

        # Main comparison from Suite A
        suite_a = self.suites.get("A")
        if suite_a and suite_a.stats:
            fp = suite_a.stats.get("full_pipeline", {})
            nb = suite_a.stats.get("naive_baseline", {})
            nf = suite_a.stats.get("no_filter", {})

            if fp and nb:
                dims = ["overall", "completeness", "source_balance",
                        "convergence_accuracy", "divergence_detection", "entity_coverage"]
                lines.append("| Metric | Nexus Pipeline | Naive Baseline | Improvement |")
                lines.append("|--------|---------------|----------------|-------------|")
                for d in dims:
                    fp_s = fp.get(d, {})
                    nb_s = nb.get(d, {})
                    if isinstance(fp_s, dict) and isinstance(nb_s, dict):
                        fp_m = fp_s.get("mean", 0)
                        fp_std = fp_s.get("std", 0)
                        nb_m = nb_s.get("mean", 0)
                        nb_std = nb_s.get("std", 0)
                        imp = ((fp_m - nb_m) / nb_m * 100) if nb_m > 0 else 0
                        lines.append(
                            f"| {d} | {fp_m:.1f} +/- {fp_std:.1f} | "
                            f"{nb_m:.1f} +/- {nb_std:.1f} | {imp:+.0f}% |"
                        )
                lines.append("")

            if nf:
                nf_overall = nf.get("overall", {})
                fp_overall = fp.get("overall", {})
                if isinstance(nf_overall, dict) and isinstance(fp_overall, dict):
                    nf_m = nf_overall.get("mean", 0)
                    fp_m = fp_overall.get("mean", 0)
                    if fp_m > 0:
                        filter_contrib = ((fp_m - nf_m) / fp_m * 100)
                        lines.append(
                            f"**Filtering contributes ~{filter_contrib:.0f}%** "
                            f"of the quality improvement (ablation study, N={fp_overall.get('n', '?')} topics)."
                        )
                        lines.append("")

        # Limitations
        lines.append(f"*Limitations: {', '.join(self.limitations)}*")
        lines.append("")

        return "\n".join(lines)


# ── Statistics ───────────────────────────────────────────────────────────────

def compute_stats(values: list[float]) -> dict:
    """Compute mean, std, min, max, n for a list of values."""
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "n": 0}
    n = len(values)
    mean = sum(values) / n
    if n < 2:
        std = 0.0
    else:
        # Population std (not sample) for consistency
        variance = sum((x - mean) ** 2 for x in values) / n
        std = math.sqrt(variance)
    return {
        "mean": round(mean, 2),
        "std": round(std, 3),
        "min": round(min(values), 2),
        "max": round(max(values), 2),
        "n": n,
    }


def compute_improvement(a_values: list[float], b_values: list[float]) -> dict:
    """Compute paired % improvement of a over b. Skips zero-baseline pairs."""
    improvements = []
    for a, b in zip(a_values, b_values):
        if b > 0:
            improvements.append((a - b) / b * 100)
    if not improvements:
        return {"mean_pct": 0.0, "std_pct": 0.0, "n": 0}
    return {
        "mean_pct": round(sum(improvements) / len(improvements), 1),
        "std_pct": round(
            math.sqrt(sum((x - sum(improvements) / len(improvements)) ** 2
                         for x in improvements) / len(improvements)),
            1,
        ) if len(improvements) > 1 else 0.0,
        "n": len(improvements),
    }


def pearson_r(x: list[float], y: list[float]) -> float:
    """Pearson correlation coefficient. Returns 0.0 if insufficient data."""
    n = min(len(x), len(y))
    if n < 2:
        return 0.0
    x, y = x[:n], y[:n]
    mx = sum(x) / n
    my = sum(y) / n
    cov = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    sx = math.sqrt(sum((xi - mx) ** 2 for xi in x))
    sy = math.sqrt(sum((yi - my) ** 2 for yi in y))
    if sx == 0 or sy == 0:
        return 0.0
    return round(cov / (sx * sy), 4)


# ── Model Override Context Manager ───────────────────────────────────────────

@contextmanager
def model_overrides(config: ModelsConfig, overrides: dict[str, str]):
    """Temporarily override config_key -> model mappings.

    Usage:
        with model_overrides(llm._config, {"filtering": "deepseek-chat"}):
            await filter_items(llm, ...)
    """
    originals = {}
    for key, model in overrides.items():
        if hasattr(config, key):
            originals[key] = getattr(config, key)
            setattr(config, key, model)
    try:
        yield
    finally:
        for key, original in originals.items():
            setattr(config, key, original)


# ── Multi-Judge ──────────────────────────────────────────────────────────────

async def multi_judge(
    llm: LLMClient,
    synthesis: TopicSynthesis,
    judges: dict[str, str],
) -> dict[str, dict]:
    """Judge a synthesis with multiple models. Returns {label: scores_dict}."""
    results = {}
    for label, model_name in judges.items():
        logger.info(f"    Judging with {label} ({model_name})")
        scores = await judge_synthesis(llm, synthesis, model_override=model_name)
        results[label] = scores
    return results


# ── Article Caching ──────────────────────────────────────────────────────────

async def cache_topic_articles(
    llm: LLMClient,
    data_dir: Path,
    store,
    topic_cfg: TopicConfig,
) -> CachedTopicData:
    """Poll, ingest, dedup articles for a topic. Returns reusable cache."""
    import yaml
    from nexus.engine.ingestion.dedup import dedup_items
    from nexus.engine.ingestion.ingest import async_ingest_items
    from nexus.engine.sources.polling import filter_recent, poll_all_feeds

    slug = topic_cfg.name.lower().replace(" ", "-").replace("/", "-")
    cache = CachedTopicData(topic_cfg=topic_cfg, slug=slug)

    registry_path = data_dir / "sources" / slug / "registry.yaml"
    if not registry_path.exists():
        logger.warning(f"No source registry for {slug}")
        return cache

    reg_data = yaml.safe_load(registry_path.read_text()) or {}
    sources_list = reg_data.get("sources", [])
    if not sources_list:
        logger.warning(f"Empty registry for {slug}")
        return cache

    logger.info(f"  Polling {len(sources_list)} sources for {slug}")
    raw_items = poll_all_feeds(sources_list)
    recent = filter_recent(raw_items, max_age_hours=168)  # 7 days
    cache.raw_articles = recent

    if not recent:
        logger.warning(f"  No recent articles for {slug}")
        return cache

    # Ingest — cap at 50 for speed
    to_ingest = recent[:50]
    logger.info(f"  Ingesting {len(to_ingest)} articles for {slug}")
    ingested = await async_ingest_items(to_ingest)
    cache.ingested_articles = dedup_items(ingested)

    # Load recent events for pass-2 context
    try:
        events = await store.get_recent_events(slug, days=7, limit=30)
        cache.recent_events = events
    except Exception:
        cache.recent_events = []

    logger.info(
        f"  Cached {len(cache.ingested_articles)} articles, "
        f"{len(cache.recent_events)} existing events for {slug}"
    )
    return cache


# ── Pipeline Variant Runners ─────────────────────────────────────────────────

async def run_full_pipeline_variant(
    llm: LLMClient,
    cached: CachedTopicData,
    store,
    threshold: float | None = None,
    relevance_weight: float | None = None,
    significance_weight: float | None = None,
    diversity_max_items: int | None = None,
    divergence_instructions: str | None = None,
    divergence_output_qualifier: str | None = None,
) -> tuple[TopicSynthesis | None, dict]:
    """Run full pipeline (filter -> extract -> synthesize) on cached articles."""
    from nexus.engine.filtering.filter import filter_items
    from nexus.engine.knowledge.events import extract_event, is_duplicate_event
    from nexus.engine.synthesis.knowledge import synthesize_topic

    if not cached.ingested_articles:
        return None, {"articles_in": 0}

    # Deep copy articles since filter_items mutates relevance_score
    articles = [item.model_copy() for item in cached.ingested_articles]

    # Filter
    filter_result = await filter_items(
        llm, articles, cached.topic_cfg,
        threshold=threshold,
        recent_events=cached.recent_events or None,
        relevance_weight=relevance_weight,
        significance_weight=significance_weight,
        diversity_max_items=diversity_max_items,
    )
    passed = filter_result.accepted

    stats = {
        "articles_in": len(articles),
        "pass_out": len(passed),
        "pass_rate": round(len(passed) / len(articles), 2) if articles else 0,
    }

    if not passed:
        return None, stats

    # Extract events
    events = []
    for article in passed:
        ev = await extract_event(llm, article, cached.topic_cfg, events)
        if ev and not any(is_duplicate_event(ev, e) for e in events):
            events.append(ev)

    stats["events"] = len(events)

    if not events:
        return None, stats

    # Synthesize
    synthesis = await synthesize_topic(
        llm, cached.topic_cfg, events, articles,
        weekly_summaries=[], monthly_summaries=[],
        store=store, topic_slug=cached.slug,
        divergence_instructions=divergence_instructions,
        divergence_output_qualifier=divergence_output_qualifier,
    )

    return synthesis, stats


async def run_no_filter_variant(
    llm: LLMClient,
    cached: CachedTopicData,
    store,
    max_articles: int = 50,
) -> tuple[TopicSynthesis | None, dict]:
    """Skip filtering — pass articles directly to extract + synthesize."""
    from nexus.engine.knowledge.events import extract_event, is_duplicate_event
    from nexus.engine.synthesis.knowledge import synthesize_topic

    if not cached.ingested_articles:
        return None, {"articles_in": 0}

    # Take top N by recency (articles are already recency-filtered)
    articles = cached.ingested_articles[:max_articles]
    stats = {"articles_in": len(articles), "pass_out": len(articles), "pass_rate": 1.0}

    events = []
    for article in articles:
        ev = await extract_event(llm, article, cached.topic_cfg, events)
        if ev and not any(is_duplicate_event(ev, e) for e in events):
            events.append(ev)

    stats["events"] = len(events)

    if not events:
        return None, stats

    synthesis = await synthesize_topic(
        llm, cached.topic_cfg, events, articles,
        weekly_summaries=[], monthly_summaries=[],
        store=store, topic_slug=cached.slug,
    )

    return synthesis, stats


# ── Suite Runners ────────────────────────────────────────────────────────────

SYNTH_DIMS = ["completeness", "source_balance", "convergence_accuracy",
              "divergence_detection", "entity_coverage", "overall"]

TEXT_DIMS = ["clarity", "insight_density", "source_attribution",
             "narrative_coherence", "actionability", "overall"]


def _aggregate_variant_stats(
    results: list[VariantResult],
    variant_name: str,
    judge_label: str = "gemini_pro",
    dims: list[str] = SYNTH_DIMS,
) -> dict:
    """Compute mean +/- std for each dimension across topics for one variant."""
    variant_results = [r for r in results if r.variant_name == variant_name]
    stats = {}
    for dim in dims:
        values = []
        for r in variant_results:
            scores = r.scores.get(judge_label, {})
            if dim in scores and not isinstance(scores.get(dim), str):
                val = scores[dim]
                if isinstance(val, (int, float)):
                    values.append(float(val))
        stats[dim] = compute_stats(values)
    return stats


async def run_suite_a(
    llm: LLMClient,
    cached_topics: list[CachedTopicData],
    store,
    config: NexusConfig,
    judge_model: str = "gemini-3.1-pro-preview",
) -> SuiteReport:
    """Suite A: Pipeline vs Baselines — full pipeline, naive, no-filter ablation."""
    from nexus.engine.synthesis.renderers import render_text_briefing

    report = SuiteReport(
        suite_id="A",
        description="Pipeline vs Baselines",
        claim="Nexus pipeline produces X% higher quality than naive summarization",
    )

    for cached in cached_topics:
        if not cached.ingested_articles:
            continue
        slug = cached.slug
        logger.info(f"  Suite A: {slug}")

        try:
            # Variant 1: Full pipeline
            logger.info(f"    Running full pipeline for {slug}")
            t0 = time.monotonic()
            fp_synth, fp_stats = await run_full_pipeline_variant(llm, cached, store)
            fp_duration = time.monotonic() - t0

            fp_result = VariantResult(
                variant_name="full_pipeline", topic_slug=slug,
                synthesis=fp_synth, filter_stats=fp_stats,
                params={"variant": "full_pipeline"}, duration_s=fp_duration,
            )
            if fp_synth:
                fp_result.scores["gemini_pro"] = await judge_synthesis(
                    llm, fp_synth, model_override=judge_model,
                )
                # Render styles for Suite D reuse
                for style in ["analytical", "conversational", "editorial"]:
                    try:
                        original_style = config.briefing.style
                        config.briefing.style = style
                        text = await render_text_briefing(llm, config, [fp_synth])
                        text_scores = await judge_briefing_text(
                            llm, text, model_override=judge_model,
                        )
                        fp_result.style_results.append(
                            StyleResult(style=style, briefing_text=text, scores=text_scores)
                        )
                    except Exception as e:
                        logger.warning(f"    Style {style} failed: {e}")
                    finally:
                        config.briefing.style = original_style
            report.results.append(fp_result)

            # Variant 2: Naive baseline (no-synthesis: real events, no thread grouping)
            logger.info(f"    Building naive baseline for {slug}")
            # Collect extracted events from the full pipeline synthesis
            fp_events = [e for t in fp_synth.threads for e in t.events] if fp_synth else []
            naive_synth = build_naive_synthesis(cached.topic_cfg.name, fp_events)
            naive_result = VariantResult(
                variant_name="naive_baseline", topic_slug=slug,
                synthesis=naive_synth, params={"variant": "naive_baseline"},
            )
            naive_result.scores["gemini_pro"] = await judge_synthesis(
                llm, naive_synth, model_override=judge_model,
            )
            report.results.append(naive_result)

            # Variant 3: No-filter ablation
            logger.info(f"    Running no-filter ablation for {slug}")
            t0 = time.monotonic()
            nf_synth, nf_stats = await run_no_filter_variant(llm, cached, store)
            nf_duration = time.monotonic() - t0

            nf_result = VariantResult(
                variant_name="no_filter", topic_slug=slug,
                synthesis=nf_synth, filter_stats=nf_stats,
                params={"variant": "no_filter"}, duration_s=nf_duration,
            )
            if nf_synth:
                nf_result.scores["gemini_pro"] = await judge_synthesis(
                    llm, nf_synth, model_override=judge_model,
                )
            report.results.append(nf_result)
        except Exception as e:
            logger.error(f"  Suite A topic {slug} failed: {e} — continuing")

    # Aggregate stats per variant
    for variant in ["full_pipeline", "naive_baseline", "no_filter"]:
        report.stats[variant] = _aggregate_variant_stats(report.results, variant)

    return report


async def run_suite_b(
    llm: LLMClient,
    cached_topics: list[CachedTopicData],
    store,
    judge_model: str = "gemini-3.1-pro-preview",
) -> SuiteReport:
    """Suite B: Filter Threshold Sensitivity — [4.0, 5.0, 6.0, 7.0, 8.0]."""
    thresholds = [4.0, 5.0, 6.0, 7.0, 8.0]
    report = SuiteReport(
        suite_id="B",
        description="Filter Threshold Sensitivity",
        claim="Optimal filter threshold is X.0 (stable across topic types)",
    )

    for cached in cached_topics:
        if not cached.ingested_articles:
            continue
        for threshold in thresholds:
            logger.info(f"    Suite B: {cached.slug} threshold={threshold}")
            try:
                t0 = time.monotonic()
                synth, stats = await run_full_pipeline_variant(
                    llm, cached, store, threshold=threshold,
                )
                duration = time.monotonic() - t0
                result = VariantResult(
                    variant_name=f"threshold_{threshold}",
                    topic_slug=cached.slug,
                    synthesis=synth, filter_stats=stats,
                    params={"threshold": threshold}, duration_s=duration,
                )
                if synth:
                    result.scores["gemini_pro"] = await judge_synthesis(
                        llm, synth, model_override=judge_model,
                    )
                report.results.append(result)
            except Exception as e:
                logger.error(f"    Suite B variant threshold={threshold}/{cached.slug} failed: {e}")

    for t in thresholds:
        report.stats[f"threshold_{t}"] = _aggregate_variant_stats(
            report.results, f"threshold_{t}",
        )

    return report


async def run_suite_c(
    llm: LLMClient,
    cached_topics: list[CachedTopicData],
    store,
    judge_model: str = "gemini-3.1-pro-preview",
) -> SuiteReport:
    """Suite C: Source Diversity Impact — low/medium/high."""
    diversity_levels = ["low", "medium", "high"]
    report = SuiteReport(
        suite_id="C",
        description="Source Diversity Impact",
        claim="High diversity improves perspective balance by X%",
    )

    for cached in cached_topics:
        if not cached.ingested_articles:
            continue
        for div in diversity_levels:
            logger.info(f"    Suite C: {cached.slug} diversity={div}")
            try:
                # Create modified topic config
                modified_topic = cached.topic_cfg.model_copy(
                    update={"perspective_diversity": div},
                )
                modified_cached = CachedTopicData(
                    topic_cfg=modified_topic,
                    slug=cached.slug,
                    raw_articles=cached.raw_articles,
                    ingested_articles=cached.ingested_articles,
                    recent_events=cached.recent_events,
                )
                t0 = time.monotonic()
                synth, stats = await run_full_pipeline_variant(
                    llm, modified_cached, store,
                )
                duration = time.monotonic() - t0
                result = VariantResult(
                    variant_name=f"diversity_{div}",
                    topic_slug=cached.slug,
                    synthesis=synth, filter_stats=stats,
                    params={"perspective_diversity": div}, duration_s=duration,
                )
                if synth:
                    result.scores["gemini_pro"] = await judge_synthesis(
                        llm, synth, model_override=judge_model,
                    )
                report.results.append(result)
            except Exception as e:
                logger.error(f"    Suite C variant diversity={div}/{cached.slug} failed: {e}")

    for div in diversity_levels:
        report.stats[f"diversity_{div}"] = _aggregate_variant_stats(
            report.results, f"diversity_{div}",
        )

    return report


async def run_suite_d(
    suite_a: SuiteReport,
) -> SuiteReport:
    """Suite D: Style Comparison — reuses Suite A style_results."""
    report = SuiteReport(
        suite_id="D",
        description="Style Comparison",
        claim="Editorial style scores highest on actionability",
    )

    # Extract style scores from Suite A full_pipeline results
    style_scores: dict[str, dict[str, list[float]]] = {}
    for r in suite_a.results:
        if r.variant_name != "full_pipeline":
            continue
        for sr in r.style_results:
            if sr.scores and "error" not in sr.scores:
                style_scores.setdefault(sr.style, {})
                for dim in TEXT_DIMS:
                    val = sr.scores.get(dim)
                    if isinstance(val, (int, float)):
                        style_scores[sr.style].setdefault(dim, []).append(float(val))

    for style, dim_vals in style_scores.items():
        stats = {}
        for dim, vals in dim_vals.items():
            stats[dim] = compute_stats(vals)
        report.stats[style] = stats

    return report


async def run_suite_e(
    llm: LLMClient,
    suite_a: SuiteReport,
    ds_model: str = "deepseek-reasoner",
    gemini_model: str = "gemini-3.1-pro-preview",
) -> SuiteReport:
    """Suite E: Cross-Judge Validation — re-judge Suite A with DeepSeek Reasoner."""
    report = SuiteReport(
        suite_id="E",
        description="Cross-Judge Validation",
        claim="Inter-judge agreement: Pearson r = X",
    )

    gemini_scores_all = []
    ds_scores_all = []

    for r in suite_a.results:
        if not r.synthesis:
            continue
        logger.info(f"    Suite E: re-judging {r.variant_name}/{r.topic_slug}")

        ds_scores = await judge_synthesis(llm, r.synthesis, model_override=ds_model)
        # Copy result with DS scores added
        new_result = VariantResult(
            variant_name=r.variant_name,
            topic_slug=r.topic_slug,
            synthesis=r.synthesis,
            scores={**r.scores, "ds_reasoner": ds_scores},
            params=r.params,
        )
        report.results.append(new_result)

        # Collect paired scores for correlation
        gemini = r.scores.get("gemini_pro", {})
        for dim in SYNTH_DIMS:
            g_val = gemini.get(dim)
            d_val = ds_scores.get(dim)
            if isinstance(g_val, (int, float)) and isinstance(d_val, (int, float)):
                gemini_scores_all.append(float(g_val))
                ds_scores_all.append(float(d_val))

    # Also re-judge text quality for full_pipeline styles
    for r in suite_a.results:
        if r.variant_name != "full_pipeline" or not r.style_results:
            continue
        for sr in r.style_results:
            if sr.briefing_text and "error" not in sr.scores:
                logger.info(f"    Suite E: re-judging text {sr.style}/{r.topic_slug}")
                ds_text_scores = await judge_briefing_text(
                    llm, sr.briefing_text, model_override=ds_model,
                )
                # Collect for correlation
                for dim in TEXT_DIMS:
                    g_val = sr.scores.get(dim)
                    d_val = ds_text_scores.get(dim)
                    if isinstance(g_val, (int, float)) and isinstance(d_val, (int, float)):
                        gemini_scores_all.append(float(g_val))
                        ds_scores_all.append(float(d_val))

    r_val = pearson_r(gemini_scores_all, ds_scores_all)
    report.stats["correlation"] = {
        "pearson_r": r_val,
        "n_pairs": len(gemini_scores_all),
    }
    report.claim = f"Inter-judge agreement: Pearson r = {r_val:.2f}"

    return report


async def run_suite_f(
    llm: LLMClient,
    cached_topics: list[CachedTopicData],
    store,
    judge_model: str = "gemini-3.1-pro-preview",
) -> SuiteReport:
    """Suite F: Scoring Weight Sensitivity."""
    weight_combos = [
        (0.3, 0.7), (0.4, 0.6), (0.5, 0.5), (0.6, 0.4),
    ]
    report = SuiteReport(
        suite_id="F",
        description="Scoring Weight Sensitivity",
        claim="Default 0.4/0.6 weighting is optimal",
    )

    # Use first 3 topics to save budget
    topics_to_use = cached_topics[:3]

    for cached in topics_to_use:
        if not cached.ingested_articles:
            continue
        for rel_w, sig_w in weight_combos:
            label = f"w_{rel_w}_{sig_w}"
            logger.info(f"    Suite F: {cached.slug} weights={rel_w}/{sig_w}")
            try:
                t0 = time.monotonic()
                synth, stats = await run_full_pipeline_variant(
                    llm, cached, store,
                    relevance_weight=rel_w, significance_weight=sig_w,
                )
                duration = time.monotonic() - t0
                result = VariantResult(
                    variant_name=label,
                    topic_slug=cached.slug,
                    synthesis=synth, filter_stats=stats,
                    params={"relevance_weight": rel_w, "significance_weight": sig_w},
                    duration_s=duration,
                )
                if synth:
                    result.scores["gemini_pro"] = await judge_synthesis(
                        llm, synth, model_override=judge_model,
                    )
                report.results.append(result)
            except Exception as e:
                logger.error(f"    Suite F variant {label}/{cached.slug} failed: {e}")

    for rel_w, sig_w in weight_combos:
        label = f"w_{rel_w}_{sig_w}"
        report.stats[label] = _aggregate_variant_stats(report.results, label)

    return report


async def run_suite_g(
    llm: LLMClient,
    cached_topics: list[CachedTopicData],
    store,
    judge_model: str = "gemini-3.1-pro-preview",
) -> SuiteReport:
    """Suite G: Model Combination Matrix — test different models per stage."""
    combos = [
        {"label": "all_flash", "filtering": "gemini-3-flash-preview", "knowledge_summary": "gemini-3-flash-preview"},
        {"label": "upgrade_filter", "filtering": "gemini-3.1-pro-preview", "knowledge_summary": "gemini-3-flash-preview"},
        {"label": "upgrade_extract_synth", "filtering": "gemini-3-flash-preview", "knowledge_summary": "gemini-3.1-pro-preview"},
        {"label": "all_pro", "filtering": "gemini-3.1-pro-preview", "knowledge_summary": "gemini-3.1-pro-preview"},
        {"label": "all_ds_chat", "filtering": "deepseek-chat", "knowledge_summary": "deepseek-chat"},
        {"label": "ds_smart_synth", "filtering": "deepseek-chat", "knowledge_summary": "deepseek-reasoner"},
        {"label": "all_ds_reasoner", "filtering": "deepseek-reasoner", "knowledge_summary": "deepseek-reasoner"},
    ]
    report = SuiteReport(
        suite_id="G",
        description="Model Combination Matrix",
        claim="Stronger models at [stage] yield X% improvement",
    )

    # Use first 3 topics to control cost
    topics_to_use = cached_topics[:3]

    for cached in topics_to_use:
        if not cached.ingested_articles:
            continue
        for combo in combos:
            label = combo["label"]
            overrides = {k: v for k, v in combo.items() if k != "label"}
            logger.info(f"    Suite G: {cached.slug} combo={label}")

            try:
                t0 = time.monotonic()
                with model_overrides(llm._config, overrides):
                    synth, stats = await run_full_pipeline_variant(llm, cached, store)
                duration = time.monotonic() - t0

                result = VariantResult(
                    variant_name=label,
                    topic_slug=cached.slug,
                    synthesis=synth, filter_stats=stats,
                    params=combo, duration_s=duration,
                )
                if synth:
                    for judge_name, judge_m in [
                        ("gemini_pro", "gemini-3.1-pro-preview"),
                        ("deepseek_chat", "deepseek-chat"),
                    ]:
                        try:
                            result.scores[judge_name] = await judge_synthesis(
                                llm, synth, model_override=judge_m,
                            )
                        except Exception as je:
                            logger.warning(f"Judge {judge_name} failed for {label}/{cached.slug}: {je}")
                report.results.append(result)
            except Exception as e:
                logger.error(f"    Suite G variant {label}/{cached.slug} failed: {e}")

    for combo in combos:
        for judge_name in ("gemini_pro", "deepseek_chat"):
            key = f"{combo['label']}_{judge_name}"
            report.stats[key] = _aggregate_variant_stats(
                report.results, combo["label"], judge_label=judge_name,
            )

    return report


async def run_suite_h(
    llm: LLMClient,
    cached_topics: list[CachedTopicData],
    store,
    judge_model: str = "gemini-3.1-pro-preview",
) -> SuiteReport:
    """Suite H: Divergence Prompt Variants — test broadened divergence detection."""
    from nexus.engine.synthesis.knowledge import DIVERGENCE_VARIANTS

    report = SuiteReport(
        suite_id="H",
        description="Divergence Prompt Variants",
        claim="Broadened divergence instructions improve divergence_detection by X%",
    )

    variant_names = list(DIVERGENCE_VARIANTS.keys())

    for cached in cached_topics:
        if not cached.ingested_articles:
            continue
        for variant_name in variant_names:
            label = f"div_{variant_name}"
            logger.info(f"    Suite H: {cached.slug} variant={variant_name}")
            try:
                t0 = time.monotonic()
                variant = DIVERGENCE_VARIANTS[variant_name]
                synth, stats = await run_full_pipeline_variant(
                    llm, cached, store,
                    divergence_instructions=variant["instructions"],
                    divergence_output_qualifier=variant["output_qualifier"],
                )
                duration = time.monotonic() - t0

                result = VariantResult(
                    variant_name=label,
                    topic_slug=cached.slug,
                    synthesis=synth,
                    filter_stats=stats,
                    params={"divergence_variant": variant_name},
                    duration_s=duration,
                )
                if synth:
                    result.scores["gemini_pro"] = await judge_synthesis(
                        llm, synth, model_override=judge_model,
                    )
                report.results.append(result)
            except Exception as e:
                logger.error(
                    f"    Suite H variant {variant_name}/{cached.slug} failed: {e}"
                )

    for variant_name in variant_names:
        label = f"div_{variant_name}"
        report.stats[label] = _aggregate_variant_stats(
            report.results, label,
        )

    return report


# ── Main Experiment Runner ───────────────────────────────────────────────────

SUITE_RUNNERS = {
    "A": "run_suite_a",
    "B": "run_suite_b",
    "C": "run_suite_c",
    "D": "run_suite_d",
    "E": "run_suite_e",
    "F": "run_suite_f",
    "G": "run_suite_g",
    "H": "run_suite_h",
}

ALL_SUITES = list(SUITE_RUNNERS.keys())


async def run_experiments(
    config: NexusConfig,
    llm: LLMClient,
    data_dir: Path,
    suites: list[str] | None = None,
    topics: list[str] | None = None,
    budget_usd: float = 15.0,
) -> ExperimentReport:
    """Main entry point — cache articles, run suites, report results."""
    from nexus.engine.knowledge.store import KnowledgeStore

    if suites is None:
        suites = ALL_SUITES

    start_time = time.monotonic()
    report = ExperimentReport(
        timestamp=datetime.now().isoformat(),
        limitations=[
            "LLM-as-judge evaluation (not human annotation)",
            f"Single time snapshot ({date.today().isoformat()})",
            f"N={len(config.topics)} topics",
            "Article availability depends on RSS feed state at time of polling",
        ],
    )

    # Filter topics
    topic_configs = config.topics
    if topics:
        topic_configs = [
            t for t in config.topics
            if t.name.lower().replace(" ", "-").replace("/", "-") in
            [s.lower().replace(" ", "-") for s in topics]
            or t.name.lower() in [s.lower() for s in topics]
        ]

    if not topic_configs:
        logger.error("No matching topics found")
        return report

    # Raise daily budget limit for experiment duration
    # Must account for already-spent amounts so the guard doesn't block
    original_daily_limit = None
    if hasattr(llm, "_budget_guard") and llm._budget_guard:
        original_daily_limit = llm._budget_guard._config.daily_limit_usd
        current_spend = llm._budget_guard.today_spend
        new_limit = max(current_spend + budget_usd, original_daily_limit)
        llm._budget_guard._config.daily_limit_usd = new_limit
        logger.info(
            f"Raised daily budget limit to ${new_limit:.2f} for experiment "
            f"(already spent ${current_spend:.2f}, headroom ${budget_usd:.2f})"
        )

    # Initialize store
    exp_dir = data_dir / "experiments"
    exp_dir.mkdir(parents=True, exist_ok=True)
    store = KnowledgeStore(exp_dir / "experiment.db")
    await store.initialize()
    await llm.set_store(store)

    try:
        # Phase 1: Cache articles for all topics
        logger.info("Phase 1: Caching articles for all topics")
        cached_topics: list[CachedTopicData] = []
        for tc in topic_configs:
            cache = await cache_topic_articles(llm, data_dir, store, tc)
            if cache.ingested_articles:
                cached_topics.append(cache)

        if not cached_topics:
            logger.error("No articles cached for any topic")
            return report

        logger.info(f"Cached articles for {len(cached_topics)} topics")

        # Phase 2+: Run suites
        suite_a_report = None

        for suite_id in suites:
            # Budget check
            cost = llm.usage.cost_summary()
            current_cost = cost.get("total_usd", 0)
            if current_cost >= budget_usd:
                logger.warning(
                    f"Budget cap reached (${current_cost:.2f} >= ${budget_usd:.2f}), "
                    f"stopping after suite {suite_id}"
                )
                break

            logger.info(f"Running Suite {suite_id} (spent ${current_cost:.2f}/{budget_usd:.2f})")

            try:
                if suite_id == "A":
                    suite_a_report = await run_suite_a(
                        llm, cached_topics, store, config,
                    )
                    report.suites["A"] = suite_a_report

                elif suite_id == "B":
                    report.suites["B"] = await run_suite_b(
                        llm, cached_topics, store,
                    )

                elif suite_id == "C":
                    report.suites["C"] = await run_suite_c(
                        llm, cached_topics, store,
                    )

                elif suite_id == "D":
                    if suite_a_report is None:
                        logger.warning("Suite D requires Suite A — skipping")
                        continue
                    report.suites["D"] = await run_suite_d(suite_a_report)

                elif suite_id == "E":
                    if suite_a_report is None:
                        logger.warning("Suite E requires Suite A — skipping")
                        continue
                    report.suites["E"] = await run_suite_e(llm, suite_a_report)

                elif suite_id == "F":
                    report.suites["F"] = await run_suite_f(
                        llm, cached_topics, store,
                    )

                elif suite_id == "G":
                    report.suites["G"] = await run_suite_g(
                        llm, cached_topics, store,
                    )

                elif suite_id == "H":
                    report.suites["H"] = await run_suite_h(
                        llm, cached_topics, store,
                    )
            except Exception as e:
                logger.error(f"Suite {suite_id} failed: {e} — continuing with remaining suites")
                report.limitations.append(f"Suite {suite_id} failed: {type(e).__name__}")

        # Final cost
        cost = llm.usage.cost_summary()
        report.total_cost = {
            "gemini": cost.get("by_provider", {}).get("gemini", 0),
            "deepseek": cost.get("by_provider", {}).get("deepseek", 0),
            "total_usd": cost.get("total_usd", 0),
        }
        report.duration_s = time.monotonic() - start_time

    finally:
        await llm.flush_usage()
        await store.close()
        # Restore original daily budget limit
        if original_daily_limit is not None and hasattr(llm, "_budget_guard") and llm._budget_guard:
            llm._budget_guard._config.daily_limit_usd = original_daily_limit

    return report
