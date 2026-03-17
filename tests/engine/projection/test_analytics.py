"""Tests for projection analytics — convergence detection (Block E)."""

from datetime import date

import pytest

from nexus.engine.knowledge.events import Event
from nexus.engine.projection.analytics import detect_converging_threads
from nexus.engine.projection.engines import (
    NativeProjectionEngine,
    ProjectionEngineInput,
    _build_prompt_context,
)
from nexus.engine.projection.models import TopicProjection
from nexus.engine.synthesis.knowledge import NarrativeThread


# ── E1: detect_converging_threads ────────────────────────────────────────────


def _make_thread(
    thread_id: int,
    headline: str,
    key_entities: list[str],
    trajectory_label: str = "accelerating",
    significance: int = 7,
    event_count: int = 3,
) -> NarrativeThread:
    return NarrativeThread(
        thread_id=thread_id,
        headline=headline,
        key_entities=key_entities,
        trajectory_label=trajectory_label,
        significance=significance,
        events=[Event(date=date(2026, 3, 10), summary=f"ev-{thread_id}", entities=key_entities)]
        * event_count,
    )


def test_detect_converging_threads_finds_shared_entities():
    """Two accelerating threads sharing 2+ entities produce a convergence signal."""
    threads = [
        _make_thread(1, "Thread A", ["Iran", "IAEA", "EU"], trajectory_label="accelerating"),
        _make_thread(2, "Thread B", ["Iran", "IAEA", "US"], trajectory_label="about_to_break"),
    ]
    signals = detect_converging_threads(threads)
    assert len(signals) == 1
    signal = signals[0]
    assert set(signal["threads"]) == {1, 2}
    assert set(signal["shared_entities"]) == {"Iran", "IAEA"}
    assert "pattern" in signal
    assert 0.0 < signal["confidence"] <= 1.0


def test_detect_converging_threads_requires_acceleration():
    """Threads must both have accelerating/about_to_break trajectory to converge."""
    threads = [
        _make_thread(1, "Thread A", ["Iran", "IAEA", "EU"], trajectory_label="accelerating"),
        _make_thread(2, "Thread B", ["Iran", "IAEA", "US"], trajectory_label="steady"),
    ]
    signals = detect_converging_threads(threads)
    assert len(signals) == 0


def test_detect_converging_threads_requires_min_shared():
    """Threads sharing fewer than min_shared_entities are not flagged."""
    threads = [
        _make_thread(1, "Thread A", ["Iran", "EU"], trajectory_label="accelerating"),
        _make_thread(2, "Thread B", ["Iran", "US"], trajectory_label="about_to_break"),
    ]
    # Only 1 shared entity ("Iran"), below the default min_shared_entities=2
    signals = detect_converging_threads(threads)
    assert len(signals) == 0


def test_detect_converging_threads_custom_threshold():
    """Custom min_shared_entities threshold is respected."""
    threads = [
        _make_thread(1, "Thread A", ["Iran", "EU"], trajectory_label="accelerating"),
        _make_thread(2, "Thread B", ["Iran", "US"], trajectory_label="about_to_break"),
    ]
    signals = detect_converging_threads(threads, min_shared_entities=1)
    assert len(signals) == 1


def test_detect_converging_threads_no_thread_id_skipped():
    """Threads without thread_id are excluded."""
    threads = [
        _make_thread(1, "Thread A", ["Iran", "IAEA"], trajectory_label="accelerating"),
        NarrativeThread(
            headline="No ID",
            key_entities=["Iran", "IAEA"],
            trajectory_label="accelerating",
            significance=7,
        ),
    ]
    signals = detect_converging_threads(threads)
    assert len(signals) == 0


# ── E2: TopicProjection model field ──────────────────────────────────────────


def test_projection_model_has_convergence_signals():
    """TopicProjection should accept and default convergence_signals."""
    proj = TopicProjection(
        topic_slug="test",
        topic_name="Test",
        generated_for=date(2026, 3, 15),
    )
    assert hasattr(proj, "convergence_signals")
    assert proj.convergence_signals == []

    proj2 = TopicProjection(
        topic_slug="test",
        topic_name="Test",
        generated_for=date(2026, 3, 15),
        convergence_signals=[{"threads": [1, 2], "shared_entities": ["Iran"]}],
    )
    assert len(proj2.convergence_signals) == 1


# ── E3: Convergence wired into engines ───────────────────────────────────────


def test_prompt_context_includes_convergence_signals():
    """_build_prompt_context should include convergence signal info when present."""
    threads = [
        _make_thread(1, "Thread A", ["Iran", "IAEA", "EU"], trajectory_label="accelerating"),
        _make_thread(2, "Thread B", ["Iran", "IAEA", "US"], trajectory_label="about_to_break"),
    ]
    convergence_signals = detect_converging_threads(threads)

    payload = ProjectionEngineInput(
        topic_slug="iran-us",
        topic_name="Iran-US Relations",
        run_date=date(2026, 3, 15),
        threads=threads,
        trajectory_threads=threads,
        metadata={"convergence_signals": convergence_signals},
    )
    prompt = _build_prompt_context(payload, max_items=3)
    assert "convergence" in prompt.lower() or "converging" in prompt.lower()
    assert "Iran" in prompt
    assert "IAEA" in prompt


@pytest.mark.asyncio
async def test_projection_includes_convergence_signals():
    """NativeProjectionEngine should populate convergence_signals on output."""
    threads = [
        _make_thread(1, "Thread A", ["Iran", "IAEA", "EU"], trajectory_label="accelerating"),
        _make_thread(2, "Thread B", ["Iran", "IAEA", "US"], trajectory_label="about_to_break"),
    ]

    payload = ProjectionEngineInput(
        topic_slug="iran-us",
        topic_name="Iran-US Relations",
        run_date=date(2026, 3, 15),
        threads=threads,
        trajectory_threads=threads,
    )

    engine = NativeProjectionEngine()
    # Use fallback path (no LLM) to verify convergence signals are populated
    projection = await engine.project(None, payload, critic_pass=False, max_items=3)
    assert isinstance(projection, TopicProjection)
    assert len(projection.convergence_signals) > 0
    assert projection.convergence_signals[0]["shared_entities"]
