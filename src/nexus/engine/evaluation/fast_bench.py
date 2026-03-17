"""Fast benchmark — deterministic Suite A on saved fixtures.

Capture mode:  Poll 2 topics, save CachedTopicData JSONs for reuse.
Run mode:      Load fixtures, run Suite A (full pipeline / naive / no-filter),
               single judge, print comparison table.

Usage:
    python -m nexus benchmark --capture           # Poll + save fixtures
    python -m nexus benchmark                     # Run on latest fixtures
    python -m nexus benchmark --threshold 4.0     # Override filter threshold
    python -m nexus benchmark --fixtures DIR      # Specific fixture snapshot
"""

import json
import logging
import re
import time
from datetime import date, datetime
from pathlib import Path

from nexus.config.models import NexusConfig
from nexus.engine.evaluation.benchmark import build_naive_synthesis
from nexus.engine.evaluation.experiment import (
    CachedTopicData,
    ExperimentReport,
    SuiteReport,
    VariantResult,
    _aggregate_variant_stats,
    cache_topic_articles,
    run_full_pipeline_variant,
    run_no_filter_variant,
)
from nexus.engine.evaluation.judge import judge_synthesis
from nexus.llm.client import LLMClient

logger = logging.getLogger(__name__)

BENCH_TOPICS = ["iran-us-relations", "ai-ml-research"]


# ── Fixture I/O ──────────────────────────────────────────────────────────────

def load_fixtures(fixture_dir: Path) -> list[CachedTopicData]:
    """Load CachedTopicData JSONs from a directory."""
    if not fixture_dir.exists():
        raise FileNotFoundError(f"Fixture directory not found: {fixture_dir}")

    json_files = sorted(fixture_dir.glob("*.json"))
    if not json_files:
        raise ValueError(f"No fixture JSON files in {fixture_dir}")

    return [CachedTopicData.from_json(p) for p in json_files]


def find_latest_fixture_dir(benchmarks_dir: Path) -> Path:
    """Find the most recent dated subdirectory (YYYY-MM-DD) in benchmarks_dir."""
    date_pattern = re.compile(r"^\d{4}-\d{2}-\d{2}$")
    candidates = [
        d for d in benchmarks_dir.iterdir()
        if d.is_dir() and date_pattern.match(d.name) and list(d.glob("*.json"))
    ]
    if not candidates:
        raise FileNotFoundError(f"No benchmark fixture directories in {benchmarks_dir}")

    return max(candidates, key=lambda d: d.name)


def apply_threshold_override(fixtures: list[CachedTopicData], threshold: float) -> None:
    """Override filter_threshold on all topic configs in-place."""
    for cached in fixtures:
        cached.topic_cfg.filter_threshold = threshold


# ── Capture ──────────────────────────────────────────────────────────────────

async def capture_benchmark(
    config: NexusConfig,
    llm: LLMClient,
    store,
    data_dir: Path,
) -> Path:
    """Poll topics and save CachedTopicData fixtures for reuse.

    Prefers BENCH_TOPICS if they exist in config; falls back to first 2 topics.
    """
    # Select topics
    topic_slugs = {
        t.name.lower().replace(" ", "-").replace("/", "-"): t
        for t in config.topics
    }

    selected = []
    for slug in BENCH_TOPICS:
        if slug in topic_slugs:
            selected.append(topic_slugs[slug])

    # Fall back: pick first 2 if bench topics not configured
    if len(selected) < 2:
        remaining = [t for t in config.topics if t not in selected]
        selected.extend(remaining[: 2 - len(selected)])

    # Create output dir
    output_dir = data_dir / "benchmarks" / date.today().isoformat()
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Capturing benchmark fixtures for {len(selected)} topics → {output_dir}")

    for topic_cfg in selected:
        slug = topic_cfg.name.lower().replace(" ", "-").replace("/", "-")
        logger.info(f"  Caching {slug}...")
        cached = await cache_topic_articles(llm, data_dir, store, topic_cfg)
        cached.to_json(output_dir / f"{slug}.json")
        logger.info(
            f"  Saved {len(cached.ingested_articles)} articles for {slug}"
        )

    return output_dir


# ── Run ──────────────────────────────────────────────────────────────────────

async def run_fast_benchmark(
    llm: LLMClient,
    store,
    fixture_dir: Path,
    threshold_override: float | None = None,
    judge_model: str = "gemini-3.1-pro-preview",
    results_dir: Path | None = None,
) -> ExperimentReport:
    """Load fixtures, run Suite A, single judge, return report."""
    t0 = time.monotonic()

    fixtures = load_fixtures(fixture_dir)

    if threshold_override is not None:
        apply_threshold_override(fixtures, threshold_override)

    # Run Suite A: full_pipeline, naive_baseline, no_filter
    report = SuiteReport(
        suite_id="A",
        description="Pipeline vs Baselines (fast)",
        claim="Nexus pipeline produces higher quality than naive summarization",
    )

    for cached in fixtures:
        if not cached.ingested_articles:
            continue
        slug = cached.slug
        logger.info(f"  Fast bench: {slug} ({len(cached.ingested_articles)} articles)")

        # Variant 1: Full pipeline
        fp_synth, fp_stats = await run_full_pipeline_variant(
            llm, cached, store, threshold=threshold_override,
        )
        fp_result = VariantResult(
            variant_name="full_pipeline", topic_slug=slug,
            synthesis=fp_synth, filter_stats=fp_stats,
        )
        if fp_synth:
            fp_result.scores["judge"] = await judge_synthesis(
                llm, fp_synth, model_override=judge_model,
            )
        report.results.append(fp_result)

        # Variant 2: Naive baseline
        fp_events = (
            [e for t in fp_synth.threads for e in t.events] if fp_synth else []
        )
        naive_synth = build_naive_synthesis(cached.topic_cfg.name, fp_events)
        naive_result = VariantResult(
            variant_name="naive_baseline", topic_slug=slug,
            synthesis=naive_synth,
        )
        naive_result.scores["judge"] = await judge_synthesis(
            llm, naive_synth, model_override=judge_model,
        )
        report.results.append(naive_result)

        # Variant 3: No-filter ablation
        nf_synth, nf_stats = await run_no_filter_variant(llm, cached, store)
        nf_result = VariantResult(
            variant_name="no_filter", topic_slug=slug,
            synthesis=nf_synth, filter_stats=nf_stats,
        )
        if nf_synth:
            nf_result.scores["judge"] = await judge_synthesis(
                llm, nf_synth, model_override=judge_model,
            )
        report.results.append(nf_result)

    # Aggregate stats
    for variant in ["full_pipeline", "naive_baseline", "no_filter"]:
        report.stats[variant] = _aggregate_variant_stats(
            report.results, variant, judge_label="judge",
        )

    duration = time.monotonic() - t0
    experiment = ExperimentReport(
        suites={"A": report},
        timestamp=datetime.now().isoformat(),
        duration_s=duration,
        environment={
            "env": "fast_bench",
            "fixture_source": str(fixture_dir),
            "threshold_override": threshold_override,
        },
    )

    # Save results
    if results_dir:
        results_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        result_path = results_dir / f"{ts}.json"
        result_path.write_text(
            json.dumps(experiment.to_json(), indent=2, default=str)
        )
        logger.info(f"Results saved: {result_path}")

    return experiment
