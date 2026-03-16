"""Tests for forecast probability calibration and candidate generation."""

from datetime import date

import pytest

from nexus.engine.knowledge.events import Event
from nexus.engine.projection.forecasting import (
    ForecastEngineInput,
    _build_candidate_catalog,
    _clip_probability,
    _empirical_adjusted_base_rate,
    _family_probability,
    _thread_trajectory_base_rate,
    _topic_direction,
)
from nexus.engine.projection.models import CrossTopicSignal
from nexus.engine.synthesis.knowledge import NarrativeThread


def _make_thread(
    *,
    thread_id=1,
    headline="Test Thread",
    trajectory_label="steady",
    momentum_score=0.0,
    snapshot_count=3,
    significance=7,
    events=None,
):
    return NarrativeThread(
        headline=headline,
        significance=significance,
        thread_id=thread_id,
        trajectory_label=trajectory_label,
        momentum_score=momentum_score,
        snapshot_count=snapshot_count,
        events=events or [
            Event(event_id=1, date=date(2026, 3, 10), summary="Test event", entities=["TestEntity"]),
        ],
    )


def _make_payload(
    threads=None,
    recent_events=None,
    cross_topic_signals=None,
    topic_slug="test-topic",
):
    if threads is None:
        threads = [_make_thread()]
    if recent_events is None:
        recent_events = [
            Event(event_id=e.event_id, date=e.date, summary=e.summary, entities=e.entities)
            for t in threads
            for e in t.events
        ]
    return ForecastEngineInput(
        topic_slug=topic_slug,
        topic_name="Test Topic",
        run_date=date(2026, 3, 15),
        threads=threads,
        recent_events=recent_events,
        cross_topic_signals=cross_topic_signals or [],
    )


# --- Trajectory base rate calibration ---


class TestTrajectoryBaseRates:
    """All trajectory base rates must stay within calibrated bounds."""

    def test_about_to_break_under_065(self):
        thread = _make_thread(trajectory_label="about_to_break", snapshot_count=10)
        rate = _thread_trajectory_base_rate(thread)
        assert rate <= 0.65, f"about_to_break base rate {rate} exceeds 0.65"

    def test_accelerating_under_055(self):
        thread = _make_thread(trajectory_label="accelerating", snapshot_count=10)
        rate = _thread_trajectory_base_rate(thread)
        assert rate <= 0.55, f"accelerating base rate {rate} exceeds 0.55"

    def test_steady_under_045(self):
        thread = _make_thread(trajectory_label="steady", snapshot_count=10)
        rate = _thread_trajectory_base_rate(thread)
        assert rate <= 0.45, f"steady base rate {rate} exceeds 0.45"

    def test_decelerating_under_030(self):
        thread = _make_thread(trajectory_label="decelerating", snapshot_count=10)
        rate = _thread_trajectory_base_rate(thread)
        assert rate <= 0.30, f"decelerating base rate {rate} exceeds 0.30"

    def test_rank_ordering_preserved(self):
        labels = ["about_to_break", "accelerating", "steady", "decelerating"]
        rates = [
            _thread_trajectory_base_rate(_make_thread(trajectory_label=label))
            for label in labels
        ]
        assert rates == sorted(rates, reverse=True), f"Rank ordering violated: {dict(zip(labels, rates))}"

    def test_all_rates_above_floor(self):
        for label in ["about_to_break", "accelerating", "steady", "decelerating"]:
            rate = _thread_trajectory_base_rate(_make_thread(trajectory_label=label))
            assert rate >= 0.05, f"{label} base rate {rate} is below 0.05 floor"


# --- Native variant probability ceiling ---


class TestNativeProbabilityCeiling:
    """The native variant must never produce probabilities above 0.75."""

    def test_max_momentum_thread_stays_under_075(self):
        """Even with about_to_break + max momentum, probability should be < 0.75."""
        thread = _make_thread(
            trajectory_label="about_to_break",
            momentum_score=50.0,  # extreme momentum
            snapshot_count=10,
        )
        payload = _make_payload(threads=[thread])
        candidates = _build_candidate_catalog(payload, "native", max_questions=10)
        for q in candidates:
            assert q.probability <= 0.75, (
                f"Native candidate '{q.target_variable}' probability {q.probability} exceeds 0.75"
            )

    def test_moderate_momentum_thread(self):
        """Typical accelerating thread should stay in 0.40-0.60 range."""
        thread = _make_thread(
            trajectory_label="accelerating",
            momentum_score=5.0,
            snapshot_count=4,
        )
        payload = _make_payload(threads=[thread])
        candidates = _build_candidate_catalog(payload, "native", max_questions=10)
        thread_questions = [q for q in candidates if q.target_variable == "thread_new_event_count"]
        for q in thread_questions:
            assert 0.30 <= q.probability <= 0.70, (
                f"Thread question probability {q.probability} outside expected [0.30, 0.70]"
            )

    def test_zero_momentum_thread(self):
        """Thread with no momentum should stay near base rate."""
        thread = _make_thread(
            trajectory_label="steady",
            momentum_score=0.0,
            snapshot_count=3,
        )
        payload = _make_payload(threads=[thread])
        candidates = _build_candidate_catalog(payload, "native", max_questions=10)
        thread_questions = [q for q in candidates if q.target_variable == "thread_new_event_count"]
        for q in thread_questions:
            assert 0.15 <= q.probability <= 0.40, (
                f"Zero-momentum thread probability {q.probability} outside expected [0.15, 0.40]"
            )


# --- Cross-topic follow-on calibration ---


class TestCrossTopicCalibration:
    """Cross-topic forecasts should be evidence-proportional, not flat-boosted."""

    def test_empty_event_ids_near_base_rate(self):
        """Cross-topic signal with no event evidence should stay near 0.46 base rate."""
        signal = CrossTopicSignal(
            signal_id=1,
            topic_slug="test-topic",
            related_topic_slug="other-topic",
            shared_entity="TestEntity",
            observed_at=date(2026, 3, 15),
            event_ids=[],
            related_event_ids=[],
            note="Test signal",
        )
        payload = _make_payload(cross_topic_signals=[signal])
        candidates = _build_candidate_catalog(payload, "native", max_questions=10)
        bridge_questions = [q for q in candidates if q.target_variable == "cross_topic_follow_on"]
        for q in bridge_questions:
            assert q.probability <= 0.55, (
                f"Cross-topic with no event evidence has probability {q.probability}, expected <= 0.55"
            )

    def test_rich_signal_gets_moderate_boost(self):
        """Cross-topic signal with several events gets a moderate boost."""
        signal = CrossTopicSignal(
            signal_id=1,
            topic_slug="test-topic",
            related_topic_slug="other-topic",
            shared_entity="TestEntity",
            observed_at=date(2026, 3, 15),
            event_ids=[1, 2, 3],
            related_event_ids=[4, 5],
            note="Rich signal",
        )
        payload = _make_payload(cross_topic_signals=[signal])
        candidates = _build_candidate_catalog(payload, "native", max_questions=10)
        bridge_questions = [q for q in candidates if q.target_variable == "cross_topic_follow_on"]
        for q in bridge_questions:
            assert q.probability <= 0.60, (
                f"Cross-topic with 3 events has probability {q.probability}, expected <= 0.60"
            )


# --- Topic direction calibration ---


class TestTopicDirectionCalibration:
    """Topic direction momentum bounds should be tight."""

    def test_high_momentum_bounded(self):
        threads = [_make_thread(momentum_score=20.0, snapshot_count=5)]
        payload = _make_payload(threads=threads)
        _, _, prob = _topic_direction(payload, "native")
        assert prob <= 0.65, f"Topic direction probability {prob} exceeds 0.65"

    def test_negative_momentum_bounded(self):
        threads = [_make_thread(momentum_score=-20.0, snapshot_count=5)]
        payload = _make_payload(threads=threads)
        _, _, prob = _topic_direction(payload, "native")
        assert prob >= 0.35, f"Topic direction probability {prob} below 0.35"


# --- Family probability calibration ---


class TestFamilyProbabilityCalibration:
    """Family event probabilities should stay conservative for native."""

    def test_high_momentum_family_bounded(self):
        _, prob = _family_probability(
            "official_statement_event",
            "native",
            context_momentum=20.0,
        )
        assert prob <= 0.85, f"Family probability {prob} exceeds 0.85 with high momentum"

    def test_baseline_family_at_base_rate(self):
        base, prob = _family_probability(
            "official_statement_event",
            "baseline",
            context_momentum=20.0,
        )
        assert prob == base, "Baseline variant should not adjust family probability"


# --- Native thread probability cap ---


class TestNativeThreadProbabilityCap:
    """Native thread forecasts must stay under 0.75 even with extreme momentum."""

    def test_about_to_break_with_max_momentum_under_075(self):
        """about_to_break + extreme momentum should not exceed 0.75."""
        thread = _make_thread(
            trajectory_label="about_to_break",
            momentum_score=50.0,
            snapshot_count=10,
        )
        payload = _make_payload(threads=[thread])
        candidates = _build_candidate_catalog(payload, "native", max_questions=10)
        thread_qs = [q for q in candidates if q.target_variable == "thread_new_event_count"]
        assert thread_qs, "Should generate at least one thread question"
        for q in thread_qs:
            assert q.probability <= 0.75, (
                f"Thread probability {q.probability} exceeds 0.75"
            )

    def test_all_trajectory_labels_under_075(self):
        """No trajectory label + momentum combination should push past 0.75."""
        for label in ["about_to_break", "accelerating", "steady", "decelerating"]:
            thread = _make_thread(
                trajectory_label=label,
                momentum_score=50.0,
                snapshot_count=10,
            )
            payload = _make_payload(threads=[thread])
            candidates = _build_candidate_catalog(payload, "native", max_questions=10)
            thread_qs = [q for q in candidates if q.target_variable == "thread_new_event_count"]
            for q in thread_qs:
                assert q.probability <= 0.75, (
                    f"Thread ({label}) probability {q.probability} exceeds 0.75"
                )


# --- Multi-horizon generation ---


class TestMultiHorizonGeneration:
    """about_to_break threads should generate questions at both 3-day and 7-day horizons."""

    def test_about_to_break_generates_two_horizons(self):
        thread = _make_thread(
            trajectory_label="about_to_break",
            momentum_score=5.0,
            snapshot_count=4,
        )
        payload = _make_payload(threads=[thread])
        candidates = _build_candidate_catalog(payload, "native", max_questions=10)
        thread_qs = [q for q in candidates if q.target_variable == "thread_new_event_count"]
        horizons = sorted(set(q.horizon_days for q in thread_qs))
        assert horizons == [3, 7], (
            f"about_to_break thread should produce [3, 7] horizons, got {horizons}"
        )

    def test_steady_thread_single_horizon(self):
        thread = _make_thread(
            trajectory_label="steady",
            momentum_score=2.0,
            snapshot_count=4,
        )
        payload = _make_payload(threads=[thread])
        candidates = _build_candidate_catalog(payload, "native", max_questions=10)
        thread_qs = [q for q in candidates if q.target_variable == "thread_new_event_count"]
        horizons = sorted(set(q.horizon_days for q in thread_qs))
        assert horizons == [7], (
            f"steady thread should produce [7] horizon only, got {horizons}"
        )


# --- Clip probability ---


class TestClipProbability:
    def test_clips_above(self):
        assert _clip_probability(1.5) == 0.95

    def test_clips_below(self):
        assert _clip_probability(-0.1) == 0.05

    def test_preserves_valid(self):
        assert _clip_probability(0.5) == 0.5


# --- Empirical adjusted base rate ---


class TestEmpiricalAdjustedBaseRate:
    def test_no_data_returns_prior(self):
        """With no historical rows, should return the hardcoded prior unchanged."""
        result = _empirical_adjusted_base_rate(0.58, [])
        assert result == 0.58

    def test_below_min_samples_returns_prior(self):
        """With fewer than min_samples rows, should return prior."""
        rows = [{"resolved_bool": True}] * 5
        result = _empirical_adjusted_base_rate(0.40, rows, min_samples=10)
        assert result == 0.40

    def test_50_samples_shifts_toward_empirical(self):
        """With 50 samples, empirical hit rate should pull the result away from prior."""
        # 50 samples, all resolving true → empirical rate = 1.0
        rows = [{"resolved_bool": True}] * 50
        result = _empirical_adjusted_base_rate(0.40, rows, min_samples=10)
        # With 50 samples, weight = min(50/50, 0.7) = 0.7
        # blended = 0.7 * 1.0 + 0.3 * 0.40 = 0.82
        assert abs(result - 0.82) < 0.01

    def test_10_samples_partial_shift(self):
        """With exactly min_samples, should shift partially."""
        # 10 samples, 3 true → empirical = 0.30
        rows = [{"resolved_bool": True}] * 3 + [{"resolved_bool": False}] * 7
        result = _empirical_adjusted_base_rate(0.50, rows, min_samples=10)
        # weight = min(10/50, 0.7) = 0.2
        # blended = 0.2 * 0.30 + 0.8 * 0.50 = 0.06 + 0.40 = 0.46
        assert abs(result - 0.46) < 0.01

    def test_build_candidate_catalog_unchanged_with_no_calibration(self):
        """Passing no calibration_data should produce the same output as before."""
        thread = _make_thread(trajectory_label="accelerating", momentum_score=5.0, snapshot_count=4)
        payload = _make_payload(threads=[thread])
        without = _build_candidate_catalog(payload, "native", max_questions=10)
        with_empty = _build_candidate_catalog(payload, "native", max_questions=10, calibration_data=None)
        assert len(without) == len(with_empty)
        for q1, q2 in zip(without, with_empty):
            assert q1.probability == q2.probability
