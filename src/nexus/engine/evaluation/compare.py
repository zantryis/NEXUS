"""Cross-environment experiment comparison and quality reports."""

SYNTH_DIMS = ["completeness", "source_balance", "convergence_accuracy",
              "divergence_detection", "entity_coverage", "overall"]


def compare_experiments(report_a: dict, report_b: dict) -> dict:
    """Compare two experiment report dicts (from ExperimentReport.to_json()).

    Returns: {shared_suites: {suite_id: comparison}, only_a: [...], only_b: [...]}
    """
    suites_a = set(report_a.get("suites", {}).keys())
    suites_b = set(report_b.get("suites", {}).keys())

    shared = suites_a & suites_b
    result = {
        "shared_suites": {},
        "only_a": sorted(suites_a - suites_b),
        "only_b": sorted(suites_b - suites_a),
        "env_a": report_a.get("environment", {}).get("env", "unknown"),
        "env_b": report_b.get("environment", {}).get("env", "unknown"),
    }

    for sid in sorted(shared):
        stats_a = report_a["suites"][sid].get("stats", {})
        stats_b = report_b["suites"][sid].get("stats", {})
        shared_variants = set(stats_a.keys()) & set(stats_b.keys())

        deltas = {}
        for v in sorted(shared_variants):
            v_deltas = {}
            for dim in SYNTH_DIMS:
                val_a = _extract_mean(stats_a[v], dim)
                val_b = _extract_mean(stats_b[v], dim)
                if val_a is not None and val_b is not None:
                    v_deltas[dim] = {"a": val_a, "b": val_b, "delta": round(val_b - val_a, 2)}
            if v_deltas:
                deltas[v] = v_deltas

        result["shared_suites"][sid] = {
            "variants_compared": len(deltas),
            "deltas": deltas,
        }

    return result


def quality_report(report_data: dict) -> str:
    """Generate human-readable quality report from an experiment report dict.

    Collects all scored variants across suites (especially G and REJUDGE),
    ranks them by overall score, and produces a markdown table.
    """
    lines = [
        "# Nexus Pipeline Quality Report",
        "",
    ]

    # Header metadata
    env = report_data.get("environment", {})
    lines.append(f"**Date**: {report_data.get('timestamp', 'unknown')}")
    lines.append(f"**Environment**: {env.get('env', 'unknown')}")
    if env.get("fixture_source"):
        lines.append(f"**Fixtures**: {env['fixture_source']}")
    if env.get("rejudge_source"):
        lines.append(f"**Re-judge source**: {env['rejudge_source']}")

    cost = report_data.get("total_cost", {})
    lines.append(f"**Cost**: ${cost.get('total_usd', 0):.2f}")
    lines.append("")

    # Collect all scored variants from all suites
    variants = _collect_scored_variants(report_data)

    if not variants:
        lines.append("*No scored variants found in this report.*")
        return "\n".join(lines)

    # Determine which judges are present across all variants
    all_judges = set()
    for v in variants:
        all_judges.update(v["judges"].keys())
    judge_list = sorted(all_judges)

    # Rank by first judge's overall (prefer "opus" as primary)
    primary_judge = "opus" if "opus" in judge_list else judge_list[0]

    # Sort by primary judge overall descending
    variants.sort(
        key=lambda v: v["judges"].get(primary_judge, {}).get("overall", 0),
        reverse=True,
    )

    # Main ranking table
    lines.append(f"## Model Configuration Rankings (judge: {primary_judge})")
    lines.append("")

    dim_cols = ["overall", "completeness", "source_balance",
                "convergence_accuracy", "divergence_detection", "entity_coverage"]
    header = "| Rank | Config | Source | " + " | ".join(d.replace("_", " ").title() for d in dim_cols) + " |"
    sep = "|------|--------|--------|" + "|".join(["--------"] * len(dim_cols)) + "|"
    lines.append(header)
    lines.append(sep)

    for i, v in enumerate(variants, 1):
        scores = v["judges"].get(primary_judge, {})
        cells = []
        for d in dim_cols:
            val = scores.get(d)
            cells.append(f"{val:.1f}" if val is not None else "-")
        lines.append(f"| {i} | {v['variant']} | {v['suite']} | " + " | ".join(cells) + " |")
    lines.append("")

    # Multi-judge comparison (if more than one judge)
    if len(judge_list) > 1:
        lines.append("## Judge Agreement")
        lines.append("")
        lines.append("| Config | " + " | ".join(f"{j} overall" for j in judge_list) + " | Spread |")
        lines.append("|--------|" + "|".join(["--------"] * len(judge_list)) + "|--------|")
        for v in variants:
            overalls = []
            cells = []
            for j in judge_list:
                val = v["judges"].get(j, {}).get("overall")
                if val is not None:
                    overalls.append(val)
                    cells.append(f"{val:.1f}")
                else:
                    cells.append("-")
            spread = f"{max(overalls) - min(overalls):.1f}" if len(overalls) > 1 else "-"
            lines.append(f"| {v['variant']} | " + " | ".join(cells) + f" | {spread} |")
        lines.append("")

    # Key findings
    lines.append("## Key Findings")
    lines.append("")
    if variants:
        best = variants[0]
        best_score = best["judges"].get(primary_judge, {}).get("overall", 0)
        lines.append(f"1. **Best quality**: {best['variant']} at {best_score:.1f}/10")
        if len(variants) > 1:
            worst = variants[-1]
            worst_score = worst["judges"].get(primary_judge, {}).get("overall", 0)
            gap = best_score - worst_score
            lines.append(f"2. **Worst quality**: {worst['variant']} at {worst_score:.1f}/10")
            lines.append(f"3. **Score range**: {gap:.1f} points across {len(variants)} configurations")
    lines.append("")

    return "\n".join(lines)


def _collect_scored_variants(report_data: dict) -> list[dict]:
    """Extract all variants with judge scores from report suites.

    Returns list of: {variant, suite, judges: {judge_label: {dim: score}}}
    """
    variants = []
    for suite_id, suite_data in report_data.get("suites", {}).items():
        stats = suite_data.get("stats", {})
        # Stats keys are "{variant}_{judge}" — group by variant
        variant_judges: dict[str, dict[str, dict]] = {}
        for key, dim_stats in stats.items():
            # Find the last underscore that splits variant from judge
            # variant names may contain underscores, so try known judge suffixes
            variant_name, judge_label = _split_variant_judge(key)
            if variant_name is None:
                continue
            if variant_name not in variant_judges:
                variant_judges[variant_name] = {}
            scores = {}
            for dim, val in dim_stats.items():
                if isinstance(val, dict) and "mean" in val:
                    scores[dim] = val["mean"]
                elif isinstance(val, (int, float)):
                    scores[dim] = float(val)
            if scores:
                variant_judges[variant_name][judge_label] = scores

        for variant_name, judges in variant_judges.items():
            variants.append({
                "variant": variant_name,
                "suite": suite_id,
                "judges": judges,
            })
    return variants


def _split_variant_judge(key: str) -> tuple[str | None, str]:
    """Split a stats key like 'all_flash_gemini_pro' into ('all_flash', 'gemini_pro').

    Known judge labels are tried from the end. Falls back to last underscore split.
    """
    known_judges = [
        "gemini_pro", "deepseek_chat", "opus", "gpt",
        "gemini_pro_2", "deepseek_chat_2",
    ]
    for judge in sorted(known_judges, key=len, reverse=True):
        suffix = f"_{judge}"
        if key.endswith(suffix):
            variant = key[:-len(suffix)]
            if variant:
                return variant, judge
    # Fallback: split on last underscore
    idx = key.rfind("_")
    if idx > 0:
        return key[:idx], key[idx + 1:]
    return None, key


def _extract_mean(stats_entry: dict, dim: str) -> float | None:
    """Get the mean value for a dimension from a stats entry."""
    val = stats_entry.get(dim)
    if val is None:
        return None
    if isinstance(val, dict) and "mean" in val:
        return val["mean"]
    if isinstance(val, (int, float)):
        return float(val)
    return None
