"""Fact-based evaluation — binary accuracy, confidence calibration, KG value-add.

Primary metrics (replacing Brier as primary):
- Coverage: % of questions where verdict ≠ uncertain
- Accuracy: % correct of called questions
- Selective accuracy: accuracy at each confidence level
- Abstention quality: does the system know what it doesn't know?
- Value-add: our accuracy - naive baseline (always NO)
- Brier: backward compatibility from implied_probability
"""

from __future__ import annotations

from dataclasses import dataclass, field

from nexus.engine.projection.models import StructuralAssessment


@dataclass
class FactEvaluationReport:
    """Evaluation results for structural predictions against known outcomes."""

    total: int = 0
    called: int = 0         # verdict != uncertain
    abstained: int = 0      # verdict == uncertain
    correct: int = 0
    wrong: int = 0

    # Primary metrics
    accuracy: float = 0.0          # correct / called
    coverage: float = 0.0          # called / total
    naive_baseline: float = 0.0    # accuracy of "always NO"
    value_add: float = 0.0         # accuracy - naive_baseline

    # Confidence calibration
    accuracy_by_confidence: dict[str, float] = field(default_factory=dict)

    # KG value-add
    accuracy_with_kg: float = 0.0
    accuracy_without_kg: float = 0.0

    # Backward-compat
    mean_brier: float = 0.0
    brier_scores: list[float] = field(default_factory=list)


def evaluate_predictions(results: list[dict]) -> FactEvaluationReport:
    """Evaluate a list of {assessment: StructuralAssessment, outcome: bool} dicts.

    Returns a FactEvaluationReport with all metrics.
    """
    report = FactEvaluationReport()
    report.total = len(results)

    if not results:
        return report

    # Buckets for confidence calibration
    conf_correct: dict[str, int] = {}
    conf_total: dict[str, int] = {}

    # KG split
    kg_correct = 0
    kg_total = 0
    no_kg_correct = 0
    no_kg_total = 0

    # Brier scores
    brier_scores: list[float] = []

    # Naive baseline: count outcomes
    outcome_no_count = sum(1 for r in results if not r["outcome"])

    for r in results:
        assessment: StructuralAssessment = r["assessment"]
        outcome: bool = r["outcome"]
        prediction = assessment.binary_prediction

        # Brier from implied_probability (backward compat)
        outcome_float = 1.0 if outcome else 0.0
        brier = (assessment.implied_probability - outcome_float) ** 2
        brier_scores.append(brier)

        if prediction is None:
            # Abstained (uncertain)
            report.abstained += 1
            continue

        report.called += 1
        is_correct = prediction == outcome

        if is_correct:
            report.correct += 1
        else:
            report.wrong += 1

        # Confidence buckets
        conf = assessment.confidence
        conf_total[conf] = conf_total.get(conf, 0) + 1
        if is_correct:
            conf_correct[conf] = conf_correct.get(conf, 0) + 1

        # KG split
        if assessment.has_kg_evidence:
            kg_total += 1
            if is_correct:
                kg_correct += 1
        else:
            no_kg_total += 1
            if is_correct:
                no_kg_correct += 1

    # Compute metrics
    report.accuracy = report.correct / report.called if report.called else 0.0
    report.coverage = report.called / report.total if report.total else 0.0

    # Naive baseline: "always NO" accuracy
    report.naive_baseline = outcome_no_count / report.total if report.total else 0.0
    report.value_add = report.accuracy - report.naive_baseline

    # Confidence calibration
    report.accuracy_by_confidence = {
        conf: conf_correct.get(conf, 0) / conf_total[conf]
        for conf in conf_total
    }

    # KG value-add
    report.accuracy_with_kg = kg_correct / kg_total if kg_total else 0.0
    report.accuracy_without_kg = no_kg_correct / no_kg_total if no_kg_total else 0.0

    # Brier
    report.brier_scores = brier_scores
    report.mean_brier = sum(brier_scores) / len(brier_scores) if brier_scores else 0.0

    return report
