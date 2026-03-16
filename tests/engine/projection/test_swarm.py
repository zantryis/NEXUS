"""Tests for swarm aggregation math: geometric mean of odds, extremization, anchor blend."""

from __future__ import annotations

import pytest

from nexus.engine.projection.swarm import (
    anchor_blend,
    extremize,
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
