"""Tests for experiment infrastructure — statistics, reporting, overrides."""

import math
import pytest
from dataclasses import asdict
from unittest.mock import AsyncMock, MagicMock, patch

from nexus.config.models import ModelsConfig, NexusConfig, TopicConfig, UserConfig
from nexus.engine.evaluation.benchmark import StyleResult


# ── Statistics ───────────────────────────────────────────────────────────────

def test_compute_stats():
    from nexus.engine.evaluation.experiment import compute_stats
    s = compute_stats([2.0, 4.0, 6.0, 8.0, 10.0])
    assert s["mean"] == 6.0
    assert s["n"] == 5
    assert s["min"] == 2.0
    assert s["max"] == 10.0
    assert abs(s["std"] - 2.828) < 0.01  # population std of [2,4,6,8,10]


def test_compute_stats_single():
    from nexus.engine.evaluation.experiment import compute_stats
    s = compute_stats([7.0])
    assert s["mean"] == 7.0
    assert s["std"] == 0.0
    assert s["n"] == 1


def test_compute_stats_empty():
    from nexus.engine.evaluation.experiment import compute_stats
    s = compute_stats([])
    assert s["mean"] == 0.0
    assert s["n"] == 0


def test_compute_improvement():
    from nexus.engine.evaluation.experiment import compute_improvement
    # 50% improvement: [6,8] vs [4,4]
    result = compute_improvement([6.0, 8.0], [4.0, 4.0])
    assert result["mean_pct"] == 75.0  # (50% + 100%) / 2
    assert result["n"] == 2


def test_compute_improvement_zero_baseline():
    from nexus.engine.evaluation.experiment import compute_improvement
    result = compute_improvement([6.0], [0.0])
    assert result["n"] == 0  # Skip zero-baseline pairs


def test_pearson_correlation():
    from nexus.engine.evaluation.experiment import pearson_r
    # Perfect positive correlation
    assert pearson_r([1, 2, 3, 4, 5], [2, 4, 6, 8, 10]) == pytest.approx(1.0, abs=0.001)


def test_pearson_correlation_negative():
    from nexus.engine.evaluation.experiment import pearson_r
    assert pearson_r([1, 2, 3, 4, 5], [10, 8, 6, 4, 2]) == pytest.approx(-1.0, abs=0.001)


def test_pearson_correlation_too_few():
    from nexus.engine.evaluation.experiment import pearson_r
    assert pearson_r([1], [2]) == 0.0  # Not enough data


# ── Data Classes ─────────────────────────────────────────────────────────────

def test_variant_result_structure():
    from nexus.engine.evaluation.experiment import VariantResult
    vr = VariantResult(
        variant_name="full_pipeline",
        topic_slug="iran-us",
        synthesis=None,
        scores={"gemini_pro": {"overall": 7.5, "completeness": 8}},
        style_results=[],
        filter_stats={"articles_in": 50, "pass1_out": 30},
        params={"threshold": 6.0},
        cost_usd=0.05,
        duration_s=120.0,
    )
    assert vr.variant_name == "full_pipeline"
    assert vr.scores["gemini_pro"]["overall"] == 7.5


def test_suite_report_structure():
    from nexus.engine.evaluation.experiment import SuiteReport, VariantResult
    sr = SuiteReport(
        suite_id="A",
        description="Pipeline vs Baselines",
        claim="Pipeline produces X% higher quality",
        results=[],
        stats={"full_pipeline": {"overall": {"mean": 7.5, "std": 0.5, "n": 5}}},
    )
    assert sr.suite_id == "A"
    assert sr.stats["full_pipeline"]["overall"]["mean"] == 7.5


# ── Report Rendering ─────────────────────────────────────────────────────────

def test_experiment_report_to_json():
    from nexus.engine.evaluation.experiment import ExperimentReport, SuiteReport
    report = ExperimentReport(
        suites={"A": SuiteReport(
            suite_id="A", description="Test", claim="test claim",
            results=[], stats={},
        )},
        total_cost={"gemini": 1.50, "deepseek": 0.30},
        timestamp="2026-03-13T12:00:00",
        duration_s=300.0,
        limitations=["LLM-as-judge"],
    )
    j = report.to_json()
    assert j["timestamp"] == "2026-03-13T12:00:00"
    assert "A" in j["suites"]
    assert j["limitations"] == ["LLM-as-judge"]


def test_experiment_report_to_markdown():
    from nexus.engine.evaluation.experiment import ExperimentReport, SuiteReport
    report = ExperimentReport(
        suites={"A": SuiteReport(
            suite_id="A", description="Pipeline vs Baselines",
            claim="Pipeline quality improvement",
            results=[], stats={"full_pipeline": {"overall": {"mean": 7.5, "std": 0.4, "n": 5}}},
        )},
        total_cost={"gemini": 1.50},
        timestamp="2026-03-13T12:00:00",
        duration_s=300.0,
        limitations=["LLM-as-judge", "N=5 topics"],
    )
    md = report.to_markdown()
    assert "# Nexus Experiment Report" in md
    assert "Pipeline vs Baselines" in md
    assert "LLM-as-judge" in md


def test_experiment_report_to_readme_snippet():
    from nexus.engine.evaluation.experiment import ExperimentReport, SuiteReport
    report = ExperimentReport(
        suites={"A": SuiteReport(
            suite_id="A", description="Pipeline vs Baselines",
            claim="X% improvement",
            results=[], stats={
                "full_pipeline": {"overall": {"mean": 7.5, "std": 0.4, "n": 5}},
                "naive_baseline": {"overall": {"mean": 3.0, "std": 0.7, "n": 5}},
            },
        )},
        total_cost={"gemini": 1.50},
        timestamp="2026-03-13",
        duration_s=300.0,
        limitations=["LLM-as-judge"],
    )
    snippet = report.to_readme_snippet()
    assert "Nexus" in snippet or "Pipeline" in snippet
    assert "Limitation" in snippet or "limitation" in snippet


# ── Model Overrides Context Manager ──────────────────────────────────────────

def test_model_overrides_context_manager():
    from nexus.engine.evaluation.experiment import model_overrides
    config = ModelsConfig()
    assert config.filtering == "gemini-3-flash-preview"
    assert config.knowledge_summary == "gemini-3-flash-preview"

    with model_overrides(config, {"filtering": "deepseek-chat", "knowledge_summary": "gemini-3.1-pro-preview"}):
        assert config.filtering == "deepseek-chat"
        assert config.knowledge_summary == "gemini-3.1-pro-preview"

    # Restored after context manager
    assert config.filtering == "gemini-3-flash-preview"
    assert config.knowledge_summary == "gemini-3-flash-preview"


def test_model_overrides_restores_on_error():
    from nexus.engine.evaluation.experiment import model_overrides
    config = ModelsConfig()
    original = config.filtering

    with pytest.raises(ValueError):
        with model_overrides(config, {"filtering": "deepseek-chat"}):
            assert config.filtering == "deepseek-chat"
            raise ValueError("intentional error")

    assert config.filtering == original


# ── Filter Weight Overrides ──────────────────────────────────────────────────

async def test_filter_weight_overrides_applied():
    """Verify new filter.py params are accepted without error."""
    from nexus.engine.filtering.filter import filter_items
    from nexus.engine.sources.polling import ContentItem

    items = [ContentItem(
        url="https://example.com/1", title="Test article",
        source_id="test", source_affiliation="private",
        source_country="US", source_language="en",
        full_text="Relevant article text about the topic.",
    )]
    topic = TopicConfig(name="Test Topic", subtopics=["subtopic"])

    mock_llm = AsyncMock()
    # Return high relevance score for every call
    mock_llm.complete = AsyncMock(
        return_value='[{"id": 0, "score": 8, "reason": "relevant"}]',
    )

    # Verify the function accepts the new params without error
    result = await filter_items(
        mock_llm, items, topic, threshold=6.0,
        relevance_weight=0.3, significance_weight=0.7,
        diversity_max_items=20,
    )
    assert len(result.accepted) > 0 or len(result.log_entries) > 0


# ── Anchored Rubric Verification ─────────────────────────────────────────────

def test_anchored_synthesis_rubric():
    """Verify judge prompt includes anchor definitions."""
    from nexus.engine.evaluation.judge import JUDGE_SYSTEM_PROMPT
    assert "2" in JUDGE_SYSTEM_PROMPT and "10" in JUDGE_SYSTEM_PROMPT
    assert "Covers 1-2 stories" in JUDGE_SYSTEM_PROMPT  # Completeness anchor at 2
    assert "FULL range" in JUDGE_SYSTEM_PROMPT


def test_anchored_briefing_rubric():
    """Verify text quality prompt includes anchor definitions."""
    from nexus.engine.evaluation.benchmark import BRIEFING_JUDGE_PROMPT
    assert "Dense, jargon-heavy" in BRIEFING_JUDGE_PROMPT  # Clarity anchor at 2
    assert "FULL range" in BRIEFING_JUDGE_PROMPT


# ── Multi-Judge ──────────────────────────────────────────────────────────────

async def test_multi_judge_calls_all_models():
    from nexus.engine.evaluation.experiment import multi_judge
    from nexus.engine.synthesis.knowledge import TopicSynthesis

    synthesis = TopicSynthesis(
        topic_name="Test", threads=[], background=[],
        source_balance={"private": 5}, languages_represented=["en"],
    )

    judges = {"gemini_pro": "gemini-3.1-pro-preview", "ds_reasoner": "deepseek-reasoner"}
    mock_llm = AsyncMock()
    mock_config = ModelsConfig()
    mock_llm._config = mock_config

    # Mock judge to return valid JSON
    with patch("nexus.engine.evaluation.experiment.judge_synthesis", new_callable=AsyncMock) as mock_judge:
        mock_judge.return_value = {"overall": 7.0, "completeness": 7}
        result = await multi_judge(mock_llm, synthesis, judges)

    assert "gemini_pro" in result
    assert "ds_reasoner" in result
    assert mock_judge.call_count == 2


# ── Suite H: Divergence Prompt Variants ──────────────────────────────────────

def test_suite_h_registered():
    """Suite H is registered in SUITE_RUNNERS."""
    from nexus.engine.evaluation.experiment import SUITE_RUNNERS
    assert "H" in SUITE_RUNNERS


def test_run_full_pipeline_variant_accepts_divergence_instructions():
    """run_full_pipeline_variant accepts divergence_instructions param."""
    import inspect
    from nexus.engine.evaluation.experiment import run_full_pipeline_variant
    sig = inspect.signature(run_full_pipeline_variant)
    assert "divergence_instructions" in sig.parameters
    assert "divergence_output_qualifier" in sig.parameters


def test_suite_h_variant_names():
    """DIVERGENCE_VARIANTS has the expected 4 variant names."""
    from nexus.engine.synthesis.knowledge import DIVERGENCE_VARIANTS
    assert len(DIVERGENCE_VARIANTS) == 4
    assert "baseline" in DIVERGENCE_VARIANTS
    assert "broadened" in DIVERGENCE_VARIANTS
    assert "structured" in DIVERGENCE_VARIANTS
    assert "encouraged" in DIVERGENCE_VARIANTS
