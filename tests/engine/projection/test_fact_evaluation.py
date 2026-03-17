"""Tests for fact-based evaluation of structural predictions."""

from __future__ import annotations


import pytest

from nexus.engine.projection.fact_evaluation import (
    evaluate_predictions,
)
from nexus.engine.projection.models import StructuralAssessment


def _make_assessment(
    question: str = "?",
    verdict: str = "yes",
    confidence: str = "medium",
    has_kg: bool = False,
) -> StructuralAssessment:
    return StructuralAssessment(
        question=question,
        verdict=verdict,
        confidence=confidence,
        has_kg_evidence=has_kg,
    )


def _make_result(
    verdict: str = "yes",
    confidence: str = "medium",
    outcome: bool = True,
    has_kg: bool = False,
) -> dict:
    """Build a prediction result dict as evaluate_predictions expects."""
    assessment = _make_assessment(verdict=verdict, confidence=confidence, has_kg=has_kg)
    return {
        "assessment": assessment,
        "outcome": outcome,
    }


class TestEvaluatePredictions:
    def test_perfect_accuracy(self):
        """All correct predictions → accuracy=1.0."""
        results = [
            _make_result(verdict="yes", outcome=True),
            _make_result(verdict="no", outcome=False),
            _make_result(verdict="yes", outcome=True),
        ]
        report = evaluate_predictions(results)
        assert report.accuracy == 1.0

    def test_zero_accuracy(self):
        """All wrong predictions → accuracy=0.0."""
        results = [
            _make_result(verdict="yes", outcome=False),
            _make_result(verdict="no", outcome=True),
        ]
        report = evaluate_predictions(results)
        assert report.accuracy == 0.0

    def test_coverage_excludes_uncertain(self):
        """Uncertain verdicts should NOT count toward accuracy but lower coverage."""
        results = [
            _make_result(verdict="yes", outcome=True),
            _make_result(verdict="uncertain", outcome=True),
            _make_result(verdict="no", outcome=False),
        ]
        report = evaluate_predictions(results)
        assert report.total == 3
        assert report.called == 2  # only yes/no
        assert report.abstained == 1
        assert report.coverage == pytest.approx(2 / 3)
        assert report.accuracy == 1.0  # both called questions correct

    def test_confidence_calibration(self):
        """High-confidence accuracy should be computed separately."""
        results = [
            _make_result(verdict="yes", confidence="high", outcome=True),
            _make_result(verdict="yes", confidence="high", outcome=True),
            _make_result(verdict="no", confidence="low", outcome=True),  # wrong
            _make_result(verdict="yes", confidence="low", outcome=True),  # correct
        ]
        report = evaluate_predictions(results)
        assert report.accuracy_by_confidence["high"] == 1.0
        assert report.accuracy_by_confidence["low"] == 0.5

    def test_brier_backward_compat(self):
        """Should compute Brier scores from implied_probability."""
        results = [
            _make_result(verdict="yes", confidence="high", outcome=True),
            # implied_prob=0.92, outcome=1 → brier=(1-0.92)^2=0.0064
        ]
        report = evaluate_predictions(results)
        assert report.mean_brier == pytest.approx(0.0064, abs=0.001)

    def test_empty_results(self):
        """Empty result set should not crash."""
        report = evaluate_predictions([])
        assert report.total == 0
        assert report.accuracy == 0.0
        assert report.coverage == 0.0

    def test_kg_value_add(self):
        """Should track accuracy split by has_kg_evidence."""
        results = [
            _make_result(verdict="yes", outcome=True, has_kg=True),
            _make_result(verdict="yes", outcome=True, has_kg=True),
            _make_result(verdict="no", outcome=True, has_kg=False),  # wrong
            _make_result(verdict="yes", outcome=True, has_kg=False),  # correct
        ]
        report = evaluate_predictions(results)
        assert report.accuracy_with_kg == 1.0
        assert report.accuracy_without_kg == 0.5

    def test_naive_baseline_comparison(self):
        """Value-add metric should compare against naive 'always NO' baseline."""
        # 3 outcomes: True, True, False
        # Naive "always NO" gets 1/3 correct
        results = [
            _make_result(verdict="yes", outcome=True),   # correct
            _make_result(verdict="yes", outcome=True),   # correct
            _make_result(verdict="no", outcome=False),    # correct
        ]
        report = evaluate_predictions(results)
        assert report.accuracy == 1.0
        naive = 1 / 3  # always NO: only the False outcome is correct
        assert report.naive_baseline == pytest.approx(naive, abs=0.01)
        assert report.value_add == pytest.approx(1.0 - naive, abs=0.01)
