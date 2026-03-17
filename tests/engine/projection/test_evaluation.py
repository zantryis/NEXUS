"""Tests for projection evaluation: leakage audit, graph export, significance tests."""

from datetime import date
import json

import pytest

from nexus.config.models import FutureProjectionConfig, NexusConfig, TopicConfig, UserConfig
from nexus.engine.knowledge.events import Event
from nexus.engine.knowledge.store import KnowledgeStore
from nexus.engine.projection.evaluation import (
    audit_forecast_leakage,
    export_graph_bundles,
    statistical_significance_test,
)
from nexus.engine.projection.models import CrossTopicSignal, ThreadSnapshot
from nexus.engine.synthesis.knowledge import NarrativeThread, TopicSynthesis


@pytest.fixture
async def store(tmp_path):
    s = KnowledgeStore(tmp_path / "forecast-eval.db")
    await s.initialize()
    yield s
    await s.close()


async def test_audit_forecast_leakage_finds_latest_state_contamination(store):
    iran_event = Event(date=date(2026, 3, 1), summary="Court sanctions filing", entities=["Iran"])
    iran_follow_on = Event(date=date(2026, 3, 12), summary="Naval strike command", entities=["Iran"])
    energy_event = Event(date=date(2026, 3, 12), summary="Energy price statement", entities=["Iran"])
    iran_ids = await store.add_events([iran_event, iran_follow_on], "iran-us")
    energy_ids = await store.add_events([energy_event], "global-energy-transition")

    thread_id = await store.upsert_thread("iran-thread", "Iran Thread", 8, "active")
    await store.link_thread_topic(thread_id, "iran-us")
    await store.link_thread_events(thread_id, iran_ids)
    await store.upsert_thread_snapshot(ThreadSnapshot(
        thread_id=thread_id,
        snapshot_date=date(2026, 3, 1),
        status="active",
        significance=8,
        event_count=1,
        latest_event_date=date(2026, 3, 1),
    ))
    await store.upsert_thread_snapshot(ThreadSnapshot(
        thread_id=thread_id,
        snapshot_date=date(2026, 3, 12),
        status="active",
        significance=8,
        event_count=2,
        latest_event_date=date(2026, 3, 12),
    ))
    await store.save_synthesis(
        TopicSynthesis(
            topic_name="Iran-US",
            threads=[NarrativeThread(
                headline="Iran Thread",
                significance=8,
                thread_id=thread_id,
                slug="iran-thread",
                status="active",
                snapshot_count=1,
                events=[Event(event_id=iran_ids[0], date=date(2026, 3, 1), summary="Court sanctions filing", entities=["Iran"])],
            )],
        ).model_dump(mode="json"),
        "iran-us",
        date(2026, 3, 1),
    )
    await store.save_synthesis(
        TopicSynthesis(
            topic_name="Global Energy Transition",
            threads=[NarrativeThread(
                headline="Energy Thread",
                significance=7,
                events=[Event(event_id=energy_ids[0], date=date(2026, 3, 12), summary="Energy price statement", entities=["Iran"])],
            )],
        ).model_dump(mode="json"),
        "global-energy-transition",
        date(2026, 3, 12),
    )
    await store.detect_and_save_cross_topic_signals(reference_date=date(2026, 3, 12))

    config = NexusConfig(
        user=UserConfig(name="Test User"),
        topics=[TopicConfig(name="Iran-US"), TopicConfig(name="Global Energy Transition")],
        future_projection=FutureProjectionConfig(enabled=True, min_history_days=1, min_thread_snapshots=1),
    )

    audit = await audit_forecast_leakage(
        store,
        config,
        start=date(2026, 3, 1),
        end=date(2026, 3, 12),
        profile="signal-rich",
    )

    assert audit["cutoffs_audited"] > 0
    assert audit["thread_state_leak_cutoffs"] >= 1
    assert audit["future_signal_leak_cutoffs"] >= 1


async def test_export_graph_bundles_writes_versioned_bundle(store, tmp_path):
    event = Event(date=date(2026, 3, 10), summary="Court sanctions filing", entities=["Iran"])
    event_ids = await store.add_events([event], "iran-us")
    thread_id = await store.upsert_thread("iran-thread", "Iran Thread", 8, "active")
    await store.link_thread_topic(thread_id, "iran-us")
    await store.link_thread_events(thread_id, event_ids)
    await store.upsert_thread_snapshot(ThreadSnapshot(
        thread_id=thread_id,
        snapshot_date=date(2026, 3, 10),
        status="active",
        significance=8,
        event_count=1,
        latest_event_date=date(2026, 3, 10),
    ))
    await store.save_synthesis(
        TopicSynthesis(
            topic_name="Iran-US",
            threads=[NarrativeThread(
                headline="Iran Thread",
                significance=8,
                thread_id=thread_id,
                slug="iran-thread",
                status="active",
                snapshot_count=1,
                events=[Event(event_id=event_ids[0], date=date(2026, 3, 10), summary="Court sanctions filing", entities=["Iran"])],
            )],
        ).model_dump(mode="json"),
        "iran-us",
        date(2026, 3, 10),
    )
    config = NexusConfig(
        user=UserConfig(name="Test User"),
        topics=[TopicConfig(name="Iran-US")],
        future_projection=FutureProjectionConfig(enabled=True, min_history_days=1, min_thread_snapshots=1),
    )

    report = await export_graph_bundles(
        store,
        config,
        start=date(2026, 3, 10),
        end=date(2026, 3, 10),
        target_dir=tmp_path / "graph",
    )

    assert report["exports"] == 1
    payload = json.loads((tmp_path / "graph" / "canonical" / "iran-us-2026-03-10.json").read_text())
    assert payload["schema_version"] == 1
    assert payload["topic_slug"] == "iran-us"
    assert payload["events"][0]["event_id"] == event_ids[0]


async def test_leakage_isolation_snapshot_count_differs_by_cutoff(store):
    """Verify load_historical_topic_state returns cutoff-scoped snapshot counts, not latest."""
    from nexus.engine.projection.service import load_historical_topic_state

    events = [
        Event(date=date(2026, 3, 1), summary="Naval sanctions strike", entities=["Iran"]),
        Event(date=date(2026, 3, 5), summary="Ceasefire talks collapse", entities=["Iran"]),
        Event(date=date(2026, 3, 10), summary="Maritime command escalation", entities=["Iran"]),
        Event(date=date(2026, 3, 14), summary="Naval strike response", entities=["Iran"]),
    ]
    event_ids = await store.add_events(events, "iran-us")
    thread_id = await store.upsert_thread("iran-thread", "Iran Thread", 8, "active")
    await store.link_thread_topic(thread_id, "iran-us")
    await store.link_thread_events(thread_id, event_ids)

    # Build growing snapshots: 1 on Mar 1, 2 on Mar 5, 3 on Mar 10, 4 on Mar 14
    for snap_date, ev_count in [
        (date(2026, 3, 1), 1),
        (date(2026, 3, 5), 2),
        (date(2026, 3, 10), 3),
        (date(2026, 3, 14), 4),
    ]:
        await store.upsert_thread_snapshot(ThreadSnapshot(
            thread_id=thread_id,
            snapshot_date=snap_date,
            status="active",
            significance=8,
            event_count=ev_count,
            latest_event_date=snap_date,
        ))

    # Save syntheses at two cutoff dates
    for synth_date, ev_slice in [(date(2026, 3, 5), events[:2]), (date(2026, 3, 14), events[:4])]:
        await store.save_synthesis(
            TopicSynthesis(
                topic_name="Iran-US",
                threads=[NarrativeThread(
                    headline="Iran Thread",
                    significance=8,
                    thread_id=thread_id,
                    slug="iran-thread",
                    status="active",
                    snapshot_count=len(ev_slice),
                    events=[
                        Event(event_id=event_ids[i], date=e.date, summary=e.summary, entities=e.entities)
                        for i, e in enumerate(ev_slice)
                    ],
                )],
            ).model_dump(mode="json"),
            "iran-us",
            synth_date,
        )

    fp_config = FutureProjectionConfig(enabled=True, min_history_days=1, min_thread_snapshots=1)

    # Load at early cutoff (Mar 5) — should see only 2 snapshots
    early = await load_historical_topic_state(
        store, topic_slug="iran-us", topic_name="Iran-US",
        cutoff=date(2026, 3, 5), config=fp_config, profile="signal-rich",
    )
    # Load at late cutoff (Mar 14) — should see all 4 snapshots
    late = await load_historical_topic_state(
        store, topic_slug="iran-us", topic_name="Iran-US",
        cutoff=date(2026, 3, 14), config=fp_config, profile="signal-rich",
    )

    assert early is not None, "Early cutoff should produce a valid state"
    assert late is not None, "Late cutoff should produce a valid state"

    early_max_snaps = max(
        (getattr(t, "snapshot_count", 0) or 0) for t in early.synthesis.threads
    )
    late_max_snaps = max(
        (getattr(t, "snapshot_count", 0) or 0) for t in late.synthesis.threads
    )

    # The late cutoff should see more snapshots than the early cutoff
    assert late_max_snaps > early_max_snaps, (
        f"Late cutoff should see more snapshots ({late_max_snaps}) than early ({early_max_snaps})"
    )


# --- Statistical significance test ---


def test_significance_test_detects_real_difference():
    """With clearly different Brier distributions, should detect significance."""
    # Native consistently better (lower Brier)
    native_briers = [0.10, 0.12, 0.08, 0.11, 0.09, 0.10, 0.12, 0.08, 0.11, 0.10,
                     0.10, 0.12, 0.08, 0.11, 0.09, 0.10, 0.12, 0.08, 0.11, 0.10]
    baseline_briers = [0.30, 0.32, 0.28, 0.31, 0.29, 0.30, 0.32, 0.28, 0.31, 0.30,
                       0.30, 0.32, 0.28, 0.31, 0.29, 0.30, 0.32, 0.28, 0.31, 0.30]
    result = statistical_significance_test(native_briers, baseline_briers)
    assert result["n"] == 20
    assert result["p_value"] < 0.05
    assert result["significant_at_005"] is True
    assert result["native_mean"] < result["baseline_mean"]


def test_significance_test_fails_with_small_n():
    """With too few samples, should not claim significance."""
    native_briers = [0.20, 0.25]
    baseline_briers = [0.30, 0.35]
    result = statistical_significance_test(native_briers, baseline_briers)
    assert result["n"] == 2
    assert result["significant_at_005"] is False
    assert result["note"] == "insufficient samples for reliable test"


def test_significance_test_identical_distributions():
    """Identical scores should not be significant."""
    briers = [0.25] * 20
    result = statistical_significance_test(briers, briers)
    assert result["significant_at_005"] is False
