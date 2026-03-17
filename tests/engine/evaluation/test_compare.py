"""Tests for cross-environment experiment comparison and quality reports."""



from nexus.engine.evaluation.compare import compare_experiments, quality_report
from nexus.engine.evaluation.experiment import (
    ExperimentReport, SuiteReport, VariantResult,
)

DIMS = ["completeness", "source_balance", "convergence_accuracy",
        "divergence_detection", "entity_coverage", "overall"]


def _make_scores(overall, completeness=7, source_balance=7,
                 convergence=7, divergence=7, entities=7):
    return {
        "completeness": completeness,
        "source_balance": source_balance,
        "convergence_accuracy": convergence,
        "divergence_detection": divergence,
        "entity_coverage": entities,
        "overall": overall,
    }


def _make_report(env="local", suites=None, timestamp="2026-03-14T10:00:00"):
    report = ExperimentReport(
        timestamp=timestamp,
        duration_s=120.0,
        total_cost={"total_usd": 1.5, "gemini": 1.0, "deepseek": 0.5, "litellm": 0.0},
        environment={"env": env, "fixture_source": "live_polling", "rejudge_source": None},
        suites=suites or {},
    )
    return report


def _make_suite_g_report(variants_scores: dict[str, dict]) -> SuiteReport:
    """Build a Suite G report from {variant_name: {judge: scores_dict}}."""
    sr = SuiteReport(
        suite_id="G", description="Model combos", claim="Test claim",
    )
    for variant_name, judge_scores in variants_scores.items():
        r = VariantResult(
            variant_name=variant_name,
            topic_slug="test-topic",
            scores=judge_scores,
        )
        sr.results.append(r)
    # Populate stats with mean values per variant per judge
    for variant_name, judge_scores in variants_scores.items():
        for judge_label, scores in judge_scores.items():
            stats_key = f"{variant_name}_{judge_label}"
            sr.stats[stats_key] = {
                dim: {"mean": float(scores[dim]), "std": 0.0}
                for dim in DIMS if dim in scores
            }
    return sr


# ── compare_experiments ──────────────────────────────────────────────────────


class TestCompareExperiments:
    def test_shared_suites_produce_deltas(self):
        """Comparing two reports with shared Suite G yields score deltas."""
        local = _make_report(env="local", suites={
            "G": _make_suite_g_report({
                "all_flash": {"gemini_pro": _make_scores(6.0)},
            }),
        })
        cloud = _make_report(env="cloud", suites={
            "G": _make_suite_g_report({
                "all_flash": {"opus": _make_scores(7.5)},
            }),
        })
        result = compare_experiments(local.to_json(), cloud.to_json())
        assert "shared_suites" in result
        assert "G" in result["shared_suites"]

    def test_disjoint_suites_reported(self):
        """Suites only in one report appear in only_a / only_b."""
        local = _make_report(suites={
            "A": SuiteReport(suite_id="A", description="d", claim="c"),
        })
        cloud = _make_report(suites={
            "G": SuiteReport(suite_id="G", description="d", claim="c"),
        })
        result = compare_experiments(local.to_json(), cloud.to_json())
        assert "A" in result["only_a"]
        assert "G" in result["only_b"]

    def test_empty_reports(self):
        """Two empty reports compare without error."""
        a = _make_report().to_json()
        b = _make_report().to_json()
        result = compare_experiments(a, b)
        assert result["shared_suites"] == {}


# ── quality_report ───────────────────────────────────────────────────────────


class TestQualityReport:
    def test_produces_markdown(self):
        """Quality report output is markdown with expected sections."""
        report = _make_report(env="cloud", suites={
            "G": _make_suite_g_report({
                "all_opus": {"opus": _make_scores(8.5, completeness=9, entities=10)},
                "all_gpt": {"opus": _make_scores(7.8)},
            }),
        })
        md = quality_report(report.to_json())
        assert "# Nexus Pipeline Quality Report" in md
        assert "all_opus" in md
        assert "all_gpt" in md

    def test_ranking_order(self):
        """Variants are ranked by overall score descending."""
        report = _make_report(env="cloud", suites={
            "G": _make_suite_g_report({
                "low_config": {"opus": _make_scores(5.0)},
                "high_config": {"opus": _make_scores(9.0)},
                "mid_config": {"opus": _make_scores(7.0)},
            }),
        })
        md = quality_report(report.to_json())
        lines = md.split("\n")
        # Find table rows (skip header/separator)
        table_rows = [l for l in lines if l.startswith("| ") and "Rank" not in l and "---" not in l]
        # First data row should be high_config
        assert "high_config" in table_rows[0]
        # Last should be low_config
        assert "low_config" in table_rows[-1]

    def test_includes_rejudge_variants(self):
        """REJUDGE suite variants appear in the unified ranking."""
        report = _make_report(env="cloud", suites={
            "G": _make_suite_g_report({
                "cloud_variant": {"opus": _make_scores(8.0)},
            }),
            "REJUDGE": _make_suite_g_report({
                "local_variant": {"opus": _make_scores(6.0)},
            }),
        })
        md = quality_report(report.to_json())
        assert "cloud_variant" in md
        assert "local_variant" in md

    def test_empty_suites(self):
        """Quality report handles reports with no suite data."""
        report = _make_report(env="cloud", suites={})
        md = quality_report(report.to_json())
        assert "# Nexus Pipeline Quality Report" in md
        assert "No scored variants" in md

    def test_environment_header(self):
        """Report header includes environment info."""
        report = _make_report(env="cloud", suites={
            "G": _make_suite_g_report({
                "v1": {"opus": _make_scores(7.0)},
            }),
        })
        md = quality_report(report.to_json())
        assert "cloud" in md.lower()

    def test_multi_judge_columns(self):
        """When multiple judges scored, report shows all judge columns."""
        report = _make_report(env="cloud", suites={
            "G": _make_suite_g_report({
                "v1": {
                    "opus": _make_scores(8.0),
                    "gpt": _make_scores(7.5),
                },
            }),
        })
        md = quality_report(report.to_json())
        assert "opus" in md.lower() or "Opus" in md
