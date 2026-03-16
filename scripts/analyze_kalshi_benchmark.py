"""Analyze Kalshi benchmark results — split analysis, significance tests, markdown output."""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nexus.engine.projection.evaluation import statistical_significance_test

BENCH_DIR = Path("data/benchmarks")
DATASET_PATH = BENCH_DIR / "kalshi_benchmark_dataset.json"
REPORT_PATH = BENCH_DIR / "kalshi_engine_comparison.json"
OUTPUT_PATH = Path("docs/benchmark-results.md")

# ── Knowledge coverage mapping ────────────────────────────────────────

COVERAGE_PATTERNS = {
    "iran-us-relations": ["iran", "israel", "abraham"],
    "ai-ml-research": ["ai", "openai", "anthropic", "tech", "datacenter", "ipo"],
    "global-energy-transition": ["energy", "oil", "gas", "nuclear", "coal", "primeeng"],
}


def classify_coverage(ticker: str, question: str) -> str:
    """Map a question to a Nexus topic, or 'uncovered'."""
    combined = f"{ticker} {question}".lower()
    for topic, patterns in COVERAGE_PATTERNS.items():
        if any(p in combined for p in patterns):
            return topic
    return "uncovered"


def analyze():
    dataset = json.loads(DATASET_PATH.read_text())
    report = json.loads(REPORT_PATH.read_text())

    per_question = report["per_question"]
    engine_results = report["engine_results"]
    engines = list(engine_results.keys())

    # ── Overall summary ──
    lines = [
        "# Kalshi Benchmark Results",
        "",
        f"**Dataset**: {len(dataset)} settled markets from Kalshi",
        f"**Engines**: {', '.join(engines)}",
        "",
        "## Overall Results",
        "",
        f"| Engine | Mean Brier | Questions |",
        f"|--------|-----------|-----------|",
    ]

    for name, result in engine_results.items():
        brier = result.get("mean_brier")
        n = result.get("questions_answered", 0)
        brier_str = f"{brier:.4f}" if brier is not None else "N/A"
        lines.append(f"| {name} | {brier_str} | {n} |")

    # ── Probability bracket analysis ──
    brackets = [
        ("Extreme (≤0.05 or ≥0.95)", lambda p: p <= 0.05 or p >= 0.95),
        ("Near-extreme (0.05-0.10 or 0.90-0.95)", lambda p: (0.05 < p <= 0.10) or (0.90 <= p < 0.95)),
        ("Mid-range (0.10-0.90)", lambda p: 0.10 < p < 0.90),
    ]

    lines.extend(["", "## By Probability Bracket", ""])

    for bracket_name, bracket_fn in brackets:
        subset = [r for r in per_question if bracket_fn(r["market_prob"])]
        if not subset:
            continue

        lines.append(f"### {bracket_name} ({len(subset)} questions)")
        lines.append("")
        lines.append(f"| Engine | Mean Brier |")
        lines.append(f"|--------|-----------|")

        for name in engines:
            briers = [r.get(f"{name}_brier", 0) for r in subset]
            if briers:
                lines.append(f"| {name} | {mean(briers):.4f} |")

        lines.append("")

    # ── Knowledge coverage split ──
    lines.extend(["## By Knowledge Coverage", ""])

    coverage_groups: dict[str, list[dict]] = defaultdict(list)
    for r in per_question:
        topic = classify_coverage(r["ticker"], r["question"])
        coverage_groups[topic].append(r)

    for topic in sorted(coverage_groups.keys()):
        subset = coverage_groups[topic]
        lines.append(f"### {topic} ({len(subset)} questions)")
        lines.append("")
        lines.append(f"| Engine | Mean Brier |")
        lines.append(f"|--------|-----------|")

        for name in engines:
            briers = [r.get(f"{name}_brier", 0) for r in subset]
            if briers:
                lines.append(f"| {name} | {mean(briers):.4f} |")

        lines.append("")

    # ── Statistical significance ──
    lines.extend(["## Statistical Significance (vs Market Baseline)", ""])

    market_briers = [r.get("market_brier", 0) for r in per_question]

    for name in engines:
        if name == "market":
            continue

        engine_briers = [r.get(f"{name}_brier", 0) for r in per_question]
        sig = statistical_significance_test(engine_briers, market_briers)

        p_val = sig.get("p_value")
        t_stat = sig.get("t_statistic")
        sig_flag = "YES" if sig.get("significant_at_005") else "no"
        p_str = f"{p_val:.4f}" if p_val is not None else "N/A"
        t_str = f"{t_stat:.2f}" if t_stat is not None else "N/A"

        lines.append(f"**{name} vs market**: t={t_str}, p={p_str}, significant={sig_flag}, "
                      f"n={sig['n']}")

    # Mid-range significance
    mid_range = [r for r in per_question if 0.10 < r["market_prob"] < 0.90]
    if len(mid_range) >= 5:
        lines.extend(["", "### Mid-range subset only (0.10-0.90)", ""])
        mid_market = [r.get("market_brier", 0) for r in mid_range]

        for name in engines:
            if name == "market":
                continue
            mid_engine = [r.get(f"{name}_brier", 0) for r in mid_range]
            sig = statistical_significance_test(mid_engine, mid_market)
            p_val = sig.get("p_value")
            t_stat = sig.get("t_statistic")
            sig_flag = "YES" if sig.get("significant_at_005") else "no"
            p_str = f"{p_val:.4f}" if p_val is not None else "N/A"
            t_str = f"{t_stat:.2f}" if t_stat is not None else "N/A"

            lines.append(f"**{name} vs market**: t={t_str}, p={p_str}, significant={sig_flag}, "
                          f"n={sig['n']}")

    # ── Outcome distribution ──
    lines.extend(["", "## Dataset Characteristics", ""])
    yes = sum(1 for d in dataset if d["outcome"])
    no = len(dataset) - yes
    lines.append(f"- **Outcomes**: YES={yes}, NO={no}")

    prob_extreme = sum(1 for d in dataset if d["market_prob_at_cutoff"] <= 0.05 or d["market_prob_at_cutoff"] >= 0.95)
    prob_mid = sum(1 for d in dataset if 0.10 < d["market_prob_at_cutoff"] < 0.90)
    lines.append(f"- **Probability distribution**: Extreme={prob_extreme}, Mid-range={prob_mid}, Other={len(dataset) - prob_extreme - prob_mid}")

    # Coverage counts
    cov_counts = Counter(classify_coverage(d["ticker"], d["question"]) for d in dataset)
    lines.append("- **Knowledge coverage**:")
    for topic, n in cov_counts.most_common():
        lines.append(f"  - {topic}: {n}")

    # ── Caveats ──
    lines.extend([
        "",
        "## Caveats",
        "",
        "1. **Hindsight bias**: LLM engines may \"remember\" outcomes of older markets from training data. "
        "The engine-vs-engine comparison is still fair (same LLM), but absolute Brier scores may be artificially low.",
        "2. **Single snapshot**: No historical candlestick data available from Kalshi API for settled markets. "
        "Market probability is the last traded price (close to 0/1 for most settled markets).",
        "3. **Extreme skew**: 90%+ of markets have extreme probabilities (≤0.05 or ≥0.95), "
        "making the mid-range subset the most meaningful comparison.",
        "4. **Knowledge coverage**: Most Kalshi categories don't overlap with existing Nexus topics. "
        "Knowledge-augmented engines (GraphRAG, perspective) have limited context to work with.",
    ])

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text("\n".join(lines) + "\n")
    print(f"Report saved to {OUTPUT_PATH}")
    print(f"\n{'='*60}")
    print("\n".join(lines[:30]))
    print("...")


if __name__ == "__main__":
    analyze()
