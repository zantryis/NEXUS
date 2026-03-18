"""Tests for swarm aggregation math: geometric mean of odds, extremization, anchor blend."""

from __future__ import annotations

import pytest

from nexus.engine.projection.swarm import (
    anchor_blend,
    derive_verdict,
    extremize,
    fit_calibration_params,
    geometric_mean_of_odds,
)


# ---------------------------------------------------------------------------
# Pure math: geometric_mean_of_odds
# ---------------------------------------------------------------------------

class TestGeometricMeanOfOdds:
    def test_equal_inputs_returns_same(self):
        result = geometric_mean_of_odds([0.6, 0.6, 0.6])
        assert result == pytest.approx(0.6, abs=0.01)

    def test_symmetric_around_half(self):
        # 0.3 and 0.7 have symmetric log-odds around 0
        result = geometric_mean_of_odds([0.3, 0.7])
        assert result == pytest.approx(0.5, abs=0.01)

    def test_high_values_pull_up(self):
        result = geometric_mean_of_odds([0.8, 0.8, 0.3])
        assert result > 0.5

    def test_low_values_pull_down(self):
        result = geometric_mean_of_odds([0.2, 0.2, 0.7])
        assert result < 0.5

    def test_weighted_shifts_toward_heavy(self):
        unweighted = geometric_mean_of_odds([0.3, 0.8])
        weighted = geometric_mean_of_odds([0.3, 0.8], weights=[1.0, 3.0])
        assert weighted > unweighted

    def test_single_input(self):
        result = geometric_mean_of_odds([0.75])
        assert result == pytest.approx(0.75, abs=0.01)

    def test_clips_to_valid_range(self):
        result = geometric_mean_of_odds([0.01, 0.99])
        assert 0.02 <= result <= 0.98

    def test_all_high(self):
        result = geometric_mean_of_odds([0.9, 0.85, 0.88])
        assert result > 0.8

    def test_all_low(self):
        result = geometric_mean_of_odds([0.1, 0.15, 0.12])
        assert result < 0.2


# ---------------------------------------------------------------------------
# Pure math: extremize
# ---------------------------------------------------------------------------

class TestExtremize:
    def test_pushes_above_half_higher(self):
        assert extremize(0.6, gamma=2.5) > 0.6

    def test_pushes_below_half_lower(self):
        assert extremize(0.4, gamma=2.5) < 0.4

    def test_preserves_half(self):
        assert extremize(0.5, gamma=2.5) == pytest.approx(0.5, abs=0.001)

    def test_gamma_1_is_identity(self):
        for p in [0.1, 0.3, 0.5, 0.7, 0.9]:
            assert extremize(p, gamma=1.0) == pytest.approx(p, abs=0.001)

    def test_higher_gamma_more_extreme(self):
        p = 0.7
        assert extremize(p, gamma=3.0) > extremize(p, gamma=2.0) > extremize(p, gamma=1.5)

    def test_clips_to_valid_range(self):
        assert extremize(0.99, gamma=5.0) <= 0.98
        assert extremize(0.01, gamma=5.0) >= 0.02

    def test_symmetric_around_half(self):
        above = extremize(0.7, gamma=2.5)
        below = extremize(0.3, gamma=2.5)
        # Should be symmetric: above-0.5 == 0.5-below
        assert abs((above - 0.5) - (0.5 - below)) < 0.01

    def test_gamma_below_1_compresses(self):
        """gamma < 1.0 should compress probabilities toward 0.5."""
        assert extremize(0.7, gamma=0.8) < 0.7
        assert extremize(0.3, gamma=0.8) > 0.3


# ---------------------------------------------------------------------------
# Anchor blend
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Verdict derivation
# ---------------------------------------------------------------------------

class TestDeriveVerdict:
    def test_high_probability_yes_high(self):
        verdict, confidence = derive_verdict(0.85)
        assert verdict == "yes"
        assert confidence == "high"

    def test_medium_probability_yes_medium(self):
        verdict, confidence = derive_verdict(0.70)
        assert verdict == "yes"
        assert confidence == "medium"

    def test_slight_lean_yes_low(self):
        verdict, confidence = derive_verdict(0.58)
        assert verdict == "yes"
        assert confidence == "low"

    def test_low_probability_no_high(self):
        verdict, confidence = derive_verdict(0.15)
        assert verdict == "no"
        assert confidence == "high"

    def test_medium_low_no_medium(self):
        verdict, confidence = derive_verdict(0.30)
        assert verdict == "no"
        assert confidence == "medium"

    def test_slight_lean_no_low(self):
        verdict, confidence = derive_verdict(0.42)
        assert verdict == "no"
        assert confidence == "low"

    def test_tossup_uncertain(self):
        verdict, confidence = derive_verdict(0.50)
        assert verdict == "uncertain"
        assert confidence == "low"

    def test_boundary_values(self):
        # Exactly on boundary: 0.80 → yes/high
        assert derive_verdict(0.80) == ("yes", "high")
        # Exactly on boundary: 0.20 → no/high
        assert derive_verdict(0.20) == ("no", "high")


class TestAnchorBlend:
    def test_full_weight_returns_swarm(self):
        result = anchor_blend(0.80, 0.50, swarm_weight=1.0)
        assert result == pytest.approx(0.80, abs=0.01)

    def test_zero_weight_returns_anchor(self):
        result = anchor_blend(0.80, 0.50, swarm_weight=0.0)
        assert result == pytest.approx(0.50, abs=0.01)

    def test_default_weight_blends(self):
        result = anchor_blend(0.80, 0.50, swarm_weight=0.4)
        # 0.50 + 0.4 * (0.80 - 0.50) = 0.50 + 0.12 = 0.62
        assert result == pytest.approx(0.62, abs=0.01)

    def test_clips_to_valid_range(self):
        assert anchor_blend(0.99, 0.98, swarm_weight=1.0) <= 0.98
        assert anchor_blend(0.01, 0.02, swarm_weight=1.0) >= 0.02

    def test_swarm_below_anchor_pulls_down(self):
        result = anchor_blend(0.20, 0.50, swarm_weight=0.4)
        assert result < 0.50


# ---------------------------------------------------------------------------
# Calibration fitting
# ---------------------------------------------------------------------------

class TestFitCalibrationParams:
    def test_too_few_samples_returns_defaults(self):
        samples = [{"raw_probability": 0.7, "resolved_bool": True}] * 5
        result = fit_calibration_params(samples)
        assert result["improved"] is False
        assert result["gamma"] == 0.8
        assert result["swarm_weight"] == 0.45
        assert "need" in result.get("reason", "")

    def test_well_calibrated_data_finds_params(self):
        # Create synthetic resolved data where gamma=0.8 is reasonable
        samples = []
        for _ in range(25):
            samples.append({"raw_probability": 0.8, "resolved_bool": True, "market_probability": None})
            samples.append({"raw_probability": 0.2, "resolved_bool": False, "market_probability": None})
        result = fit_calibration_params(samples)
        assert result["n_samples"] == 50
        assert result["mean_brier"] is not None
        assert result["mean_brier"] < 0.25  # Should be well-calibrated

    def test_overconfident_data_compresses(self):
        # Overconfident: predicts 0.95 but outcome is 50/50
        samples = []
        for _ in range(15):
            samples.append({"raw_probability": 0.95, "resolved_bool": True, "market_probability": None})
            samples.append({"raw_probability": 0.95, "resolved_bool": False, "market_probability": None})
        result = fit_calibration_params(samples)
        assert result["n_samples"] == 30
        # Optimal gamma should compress (< 1.0) since LLM is overconfident
        assert result["gamma"] < 1.0

    def test_exactly_20_samples_runs_tuning(self):
        """Boundary: exactly MIN_RESOLVED_FOR_TUNING samples should trigger tuning."""
        samples = [
            {"raw_probability": 0.7, "resolved_bool": True, "market_probability": None}
        ] * 20
        result = fit_calibration_params(samples)
        assert result["n_samples"] == 20
        assert result["mean_brier"] is not None  # tuning ran, not defaults

    def test_exactly_19_samples_returns_defaults(self):
        """One below threshold should return defaults."""
        samples = [
            {"raw_probability": 0.7, "resolved_bool": True, "market_probability": None}
        ] * 19
        result = fit_calibration_params(samples)
        assert result["improved"] is False
        assert result["gamma"] == 0.8
        assert "need" in result.get("reason", "")

    def test_all_yes_outcomes(self):
        """All-YES outcomes: engine that says 0.9 should be well-calibrated."""
        samples = [
            {"raw_probability": 0.9, "resolved_bool": True, "market_probability": None}
        ] * 25
        result = fit_calibration_params(samples)
        assert result["n_samples"] == 25
        assert result["mean_brier"] is not None
        assert result["mean_brier"] < 0.1  # 0.9 predicting all-yes is good

    def test_all_no_outcomes(self):
        """All-NO outcomes: engine that says 0.1 should be well-calibrated."""
        samples = [
            {"raw_probability": 0.1, "resolved_bool": False, "market_probability": None}
        ] * 25
        result = fit_calibration_params(samples)
        assert result["n_samples"] == 25
        assert result["mean_brier"] < 0.1

    def test_all_half_probability(self):
        """All-0.5 predictions on mixed outcomes: should be moderately calibrated."""
        samples = []
        for i in range(25):
            samples.append({"raw_probability": 0.5, "resolved_bool": i % 2 == 0, "market_probability": None})
        result = fit_calibration_params(samples)
        assert result["n_samples"] == 25
        assert result["mean_brier"] is not None
        # 0.5 on 50/50 outcomes → Brier = 0.25 (maximum entropy)
        assert result["mean_brier"] == pytest.approx(0.25, abs=0.02)

    def test_with_market_anchor(self):
        """Samples with market_probability should use anchor blending."""
        samples = []
        for _ in range(15):
            # Market says 0.6, LLM says 0.8, outcome is yes
            samples.append({"raw_probability": 0.8, "resolved_bool": True, "market_probability": 0.6})
            # Market says 0.3, LLM says 0.2, outcome is no
            samples.append({"raw_probability": 0.2, "resolved_bool": False, "market_probability": 0.3})
        result = fit_calibration_params(samples)
        assert result["n_samples"] == 30
        assert result["mean_brier"] is not None
        assert result["mean_brier"] < 0.15  # Both anchored and raw agree

    def test_skips_entries_with_none_probability(self):
        """Entries missing raw_probability should be excluded."""
        samples = [
            {"raw_probability": 0.7, "resolved_bool": True, "market_probability": None},
        ] * 20 + [
            {"raw_probability": None, "resolved_bool": True},  # should be skipped
        ] * 5
        result = fit_calibration_params(samples)
        assert result["n_samples"] == 20  # only the valid ones

    def test_skips_entries_with_none_outcome(self):
        """Entries missing resolved_bool should be excluded."""
        samples = [
            {"raw_probability": 0.7, "resolved_bool": True, "market_probability": None},
        ] * 20 + [
            {"raw_probability": 0.7, "resolved_bool": None},  # should be skipped
        ] * 5
        result = fit_calibration_params(samples)
        assert result["n_samples"] == 20
