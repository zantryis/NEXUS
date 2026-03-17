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

from nexus.engine.projection.swarm import extremize

BENCH_DIR = Path("data/benchmarks")
# Support both old and new paths — prefer v2 if it exists
_V2_DATASET = BENCH_DIR / "kalshi_benchmark_full.json"
_V2_REPORT = BENCH_DIR / "kalshi_engine_comparison_v2_independent.json"
DATASET_PATH = _V2_DATASET if _V2_DATASET.exists() else BENCH_DIR / "kalshi_benchmark_dataset.json"
REPORT_PATH = _V2_REPORT if _V2_REPORT.exists() else BENCH_DIR / "kalshi_engine_comparison.json"
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

    # ── Gamma sweep ──
    # For engines that apply extremize(gamma=0.8) internally, undo it and re-apply
    # with a range of gammas to find the optimal calibration.
    # structural engine uses numeric_probability (no gamma), so it's swept raw.
    GAMMA_ENGINES = {"naked", "actor", "graphrag", "perspective", "debate"}  # engines with gamma=0.8
    GAMMA_VALUES = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 1.73, 2.0, 2.5]

    lines.extend(["", "## Gamma Sweep (post-hoc recalibration)", ""])
    lines.append("For each engine, undo the production gamma (0.8) and reapply with candidate gammas.")
    lines.append("")

    for name in engines:
        if name == "market":
            continue
        has_gamma = any(name.startswith(g) for g in GAMMA_ENGINES)
        if not has_gamma:
            # For structural/other engines without gamma: apply gamma directly to output
            lines.append(f"### {name} (no production gamma — sweep applied raw)")
        else:
            lines.append(f"### {name} (production gamma=0.8)")
        lines.append("")
        lines.append(f"| Gamma | Mean Brier | vs Market |")
        lines.append(f"|-------|-----------|-----------|")

        market_mean = mean([r.get("market_brier", 0) for r in per_question]) if per_question else 0

        for gamma in GAMMA_VALUES:
            recal_briers = []
            for r in per_question:
                engine_prob = r.get(f"{name}_prob")
                if engine_prob is None:
                    continue
                outcome = r.get("outcome", False)
                target = 1.0 if outcome else 0.0

                if has_gamma:
                    # Undo gamma=0.8, then apply new gamma
                    raw = extremize(engine_prob, 1.0 / 0.8)  # undo
                    recal = extremize(raw, gamma)
                else:
                    # Apply gamma directly
                    recal = extremize(engine_prob, gamma)

                recal = max(0.05, min(0.95, recal))
                brier = (recal - target) ** 2
                recal_briers.append(brier)

            if recal_briers:
                m = mean(recal_briers)
                delta = m - market_mean
                sign = "+" if delta > 0 else ""
                lines.append(f"| {gamma:.2f} | {m:.4f} | {sign}{delta:.4f} |")

        lines.append("")

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
