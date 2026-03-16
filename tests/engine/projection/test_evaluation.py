"""Tests for quantified forecast evaluation and graph helpers."""

from datetime import date
import json

import pytest

from nexus.config.models import FutureProjectionConfig, NexusConfig, TopicConfig, UserConfig
from nexus.engine.knowledge.events import Event
from nexus.engine.knowledge.store import KnowledgeStore
from nexus.engine.projection.evaluation import (
    audit_forecast_leakage,
    benchmark_forecast_engines,
    export_graph_bundles,
    forecast_readiness_report,
    generate_prediction_audit,
    render_prediction_audit_markdown,
    resolve_forecast_question,
    statistical_significance_test,
)
from nexus.engine.projection.graph import (
    build_graph_snapshot,
    get_graph_evidence_adapter,
    rank_graph_evidence,
)
from nexus.engine.projection.models import CrossTopicSignal, ThreadSnapshot
from nexus.engine.synthesis.knowledge import NarrativeThread, TopicSynthesis


@pytest.fixture
async def store(tmp_path):
    s = KnowledgeStore(tmp_path / "forecast-eval.db")
    await s.initialize()
    yield s
    await s.close()


def test_build_graph_snapshot_and_rank_evidence():
    snapshot = build_graph_snapshot(
        topic_slug="iran-us",
        run_date=date(2026, 3, 10),
        threads=[NarrativeThread(
            headline="Iran Thread",
            significance=8,
            thread_id=7,
            snapshot_count=3,
            momentum_score=4.2,
            trajectory_label="accelerating",
            events=[Event(event_id=11, date=date(2026, 3, 10), summary="A", entities=["Iran"])],
        )],
        cross_topic_signals=[CrossTopicSignal(
            signal_id=5,
            topic_slug="iran-us",
            related_topic_slug="global-energy-transition",
            shared_entity="Iran",
            observed_at=date(2026, 3, 10),
            event_ids=[11],
            related_event_ids=[12],
            note="Iran was active across topics.",
        )],
    )

    ranked = rank_graph_evidence(snapshot)
    assert snapshot.metrics["node_count"] > 0
    assert ranked["event_ids"] == [11]
    assert 7 in ranked["thread_ids"]


async def test_resolve_forecast_question_thread_new_event(store):
    event_a = Event(date=date(2026, 3, 8), summary="A", entities=["Iran"])
    event_b = Event(date=date(2026, 3, 10), summary="B", entities=["Iran"])
    event_ids = await store.add_events([event_a, event_b], "iran-us")
    thread_id = await store.upsert_thread("iran-thread", "Iran Thread", 8, "active")
    await store.link_thread_topic(thread_id, "iran-us")
    await store.link_thread_events(thread_id, event_ids)

    resolution = await resolve_forecast_question(
        store,
        {
            "forecast_question_id": 1,
            "topic_slug": "iran-us",
            "generated_for": "2026-03-08",
            "resolution_date": "2026-03-10",
            "horizon_days": 7,
            "probability": 0.68,
            "target_variable": "thread_new_event_count",
            "target_metadata": {"thread_id": thread_id, "threshold": 1, "topic_slug": "iran-us"},
            "expected_direction": None,
        },
    )

    assert resolution.outcome_status == "resolved"
    assert resolution.resolved_bool is True
    assert resolution.brier_score is not None


async def test_resolve_forecast_question_legal_action_event(store):
    event_a = Event(date=date(2026, 3, 8), summary="Dominion Energy permit lawsuit filed", entities=["Dominion Energy", "Virginia"])
    event_b = Event(date=date(2026, 3, 10), summary="Virginia court schedules Dominion Energy permit hearing", entities=["Dominion Energy", "Virginia"])
    await store.add_events([event_a, event_b], "global-energy-transition")

    resolution = await resolve_forecast_question(
        store,
        {
            "forecast_question_id": 2,
            "topic_slug": "global-energy-transition",
            "generated_for": "2026-03-08",
            "resolution_date": "2026-03-10",
            "horizon_days": 7,
            "probability": 0.41,
            "target_variable": "legal_action_event",
            "target_metadata": {
                "topic_slug": "global-energy-transition",
                "keywords": ["court", "lawsuit", "permit", "regulatory"],
                "anchor_entities": ["Dominion Energy"],
            },
            "expected_direction": None,
        },
    )

    assert resolution.outcome_status == "resolved"
    assert resolution.resolved_bool is True
    assert resolution.actual_value == 1.0


async def test_benchmark_forecast_engines_returns_metrics(store):
    event_a = Event(date=date(2026, 3, 1), summary="Sanctions announced", entities=["Iran"])
    event_b = Event(date=date(2026, 3, 8), summary="Follow-on sanctions filing", entities=["Iran"])
    event_c = Event(date=date(2026, 3, 10), summary="Treasury statement", entities=["Iran"])
    event_d = Event(date=date(2026, 3, 16), summary="New sanctions package", entities=["Iran"])
    event_ids = await store.add_events([event_a, event_b, event_c, event_d], "iran-us")
    thread_id = await store.upsert_thread("iran-thread", "Iran Thread", 8, "active")
    await store.link_thread_topic(thread_id, "iran-us")
    await store.link_thread_events(thread_id, event_ids)
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
        snapshot_date=date(2026, 3, 8),
        status="active",
        significance=8,
        event_count=2,
        latest_event_date=date(2026, 3, 8),
    ))
    await store.upsert_thread_snapshot(ThreadSnapshot(
        thread_id=thread_id,
        snapshot_date=date(2026, 3, 9),
        status="active",
        significance=8,
        event_count=2,
        latest_event_date=date(2026, 3, 8),
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
                snapshot_count=3,
                events=[
                    Event(event_id=event_ids[0], date=date(2026, 3, 1), summary="Sanctions announced", entities=["Iran"]),
                    Event(event_id=event_ids[1], date=date(2026, 3, 8), summary="Follow-on sanctions filing", entities=["Iran"]),
                ],
            )],
        ).model_dump(mode="json"),
        "iran-us",
        date(2026, 3, 9),
    )
    await store.save_synthesis(
        TopicSynthesis(
            topic_name="Iran-US",
            threads=[NarrativeThread(
                headline="Iran Thread",
                significance=8,
                thread_id=thread_id,
                slug="iran-thread",
                status="active",
                snapshot_count=3,
                events=[
                    Event(event_id=event_ids[1], date=date(2026, 3, 8), summary="Follow-on sanctions filing", entities=["Iran"]),
                    Event(event_id=event_ids[2], date=date(2026, 3, 10), summary="Treasury statement", entities=["Iran"]),
                ],
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

    report = await benchmark_forecast_engines(
        store,
        config,
        start=date(2026, 3, 9),
        end=date(2026, 3, 10),
        engines=["baseline", "trajectory", "native"],
        llm=None,
        mode="audit",
        profile="signal-rich",
        strict=True,
        max_questions=3,
    )

    assert report["meta"]["mode"] == "audit"
    assert report["meta"]["validity_label"] == "infrastructure-valid, statistically-insufficient"
    assert report["engines"]["baseline"]["total"] > 0
    assert "mean_brier" in report["engines"]["native"]
    assert "calibration" in report["engines"]["trajectory"]


async def test_benchmark_forecast_engines_is_deterministic_and_does_not_write(store):
    event_a = Event(date=date(2026, 3, 1), summary="Court sanctions filing", entities=["Iran"])
    event_b = Event(date=date(2026, 3, 8), summary="Treasury policy statement", entities=["Iran"])
    event_c = Event(date=date(2026, 3, 16), summary="New sanctions package", entities=["Iran"])
    event_ids = await store.add_events([event_a, event_b, event_c], "iran-us")
    thread_id = await store.upsert_thread("iran-thread", "Iran Thread", 8, "active")
    await store.link_thread_topic(thread_id, "iran-us")
    await store.link_thread_events(thread_id, event_ids)
    for snapshot_date, event_count in [(date(2026, 3, 1), 1), (date(2026, 3, 8), 2), (date(2026, 3, 9), 2)]:
        await store.upsert_thread_snapshot(ThreadSnapshot(
            thread_id=thread_id,
            snapshot_date=snapshot_date,
            status="active",
            significance=8,
            event_count=event_count,
            latest_event_date=min(snapshot_date, date(2026, 3, 8)),
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
                snapshot_count=3,
                events=[
                    Event(event_id=event_ids[0], date=date(2026, 3, 1), summary="Court sanctions filing", entities=["Iran"]),
                    Event(event_id=event_ids[1], date=date(2026, 3, 8), summary="Treasury policy statement", entities=["Iran"]),
                ],
            )],
        ).model_dump(mode="json"),
        "iran-us",
        date(2026, 3, 9),
    )
    config = NexusConfig(
        user=UserConfig(name="Test User"),
        topics=[TopicConfig(name="Iran-US")],
        future_projection=FutureProjectionConfig(enabled=True, min_history_days=1, min_thread_snapshots=1),
    )

    before = await store.get_latest_forecast_run("iran-us")
    first = await benchmark_forecast_engines(
        store,
        config,
        start=date(2026, 3, 9),
        end=date(2026, 3, 9),
        engines=["baseline", "trajectory", "native"],
        llm=None,
        mode="audit",
        profile="signal-rich",
        strict=True,
        max_questions=3,
    )
    second = await benchmark_forecast_engines(
        store,
        config,
        start=date(2026, 3, 9),
        end=date(2026, 3, 9),
        engines=["baseline", "trajectory", "native"],
        llm=None,
        mode="audit",
        profile="signal-rich",
        strict=True,
        max_questions=3,
    )
    after = await store.get_latest_forecast_run("iran-us")

    assert first == second
    assert before is None
    assert after is None


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


async def test_graph_adapter_soft_skip_when_dependency_missing(monkeypatch):
    async def _run():
        bundle = {
            "schema_version": 1,
            "topic_slug": "iran-us",
            "topic_name": "Iran-US",
            "cutoff": "2026-03-10",
            "threads": [],
            "events": [],
            "causal_links": [],
            "cross_topic_signals": [],
            "nodes": [],
            "edges": [],
            "evidence_catalog": {},
            "metadata": {},
        }
        from nexus.engine.projection.models import GraphExportBundle

        monkeypatch.setattr("importlib.util.find_spec", lambda _: None)
        result = await get_graph_evidence_adapter("kuzu").query(GraphExportBundle(**bundle))
        assert result.status == "skipped"

    await _run()


async def test_forecast_readiness_report_surfaces_graph_and_kalshi_status(store, tmp_path):
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

    report = await forecast_readiness_report(
        store,
        config,
        start=date(2026, 3, 10),
        end=date(2026, 3, 10),
        base_dir=tmp_path / "benchmarks",
    )

    assert "benchmark_trusted" in report["readiness"]
    assert report["graph_export"]["exports"] == 1
    assert "kuzu" in report["graph_sidecars"]
    assert report["kalshi"]["mapping_count"] >= 0
    assert report["readiness"]["kalshi_auth_ready"] is False
    assert report["readiness"]["kalshi_compare_ready"] is False


async def test_benchmark_includes_topic_with_min_thread_snapshots_2(store):
    """Iran-US-style topic with 2 snapshots at early cutoffs should be included when min_thread_snapshots=2."""
    # Set up a topic with events spanning enough days
    events = [
        Event(date=date(2026, 3, 1), summary="Naval sanctions strike in Iran", entities=["Iran"]),
        Event(date=date(2026, 3, 8), summary="Ceasefire policy statement", entities=["Iran"]),
        Event(date=date(2026, 3, 14), summary="Maritime naval command dispatch", entities=["Iran"]),
    ]
    event_ids = await store.add_events(events, "iran-us")
    thread_id = await store.upsert_thread("iran-thread", "Iran Thread", 8, "active")
    await store.link_thread_topic(thread_id, "iran-us")
    await store.link_thread_events(thread_id, event_ids)

    # Only 2 snapshots — would fail min_thread_snapshots=3 but pass min_thread_snapshots=2
    for snap_date, event_count in [(date(2026, 3, 8), 2), (date(2026, 3, 10), 2)]:
        await store.upsert_thread_snapshot(ThreadSnapshot(
            thread_id=thread_id,
            snapshot_date=snap_date,
            status="active",
            significance=8,
            event_count=event_count,
            latest_event_date=snap_date,
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
                snapshot_count=2,
                events=[
                    Event(event_id=event_ids[1], date=date(2026, 3, 8), summary="Ceasefire policy statement", entities=["Iran"]),
                ],
            )],
        ).model_dump(mode="json"),
        "iran-us",
        date(2026, 3, 10),
    )

    # With min_thread_snapshots=2, this topic SHOULD appear in the benchmark
    config_pass = NexusConfig(
        user=UserConfig(name="Test User"),
        topics=[TopicConfig(name="Iran-US")],
        future_projection=FutureProjectionConfig(enabled=True, min_history_days=1, min_thread_snapshots=2),
    )
    report_pass = await benchmark_forecast_engines(
        store, config_pass,
        start=date(2026, 3, 9), end=date(2026, 3, 10),
        engines=["baseline"], llm=None, mode="audit", profile="signal-rich", strict=True,
    )
    assert "iran-us" in report_pass["meta"]["domains"], (
        f"iran-us should be in domains with min_thread_snapshots=2, got {report_pass['meta']['domains']}"
    )

    # With min_thread_snapshots=3, this topic should be EXCLUDED
    config_fail = NexusConfig(
        user=UserConfig(name="Test User"),
        topics=[TopicConfig(name="Iran-US")],
        future_projection=FutureProjectionConfig(enabled=True, min_history_days=1, min_thread_snapshots=3),
    )
    report_fail = await benchmark_forecast_engines(
        store, config_fail,
        start=date(2026, 3, 9), end=date(2026, 3, 10),
        engines=["baseline"], llm=None, mode="audit", profile="signal-rich", strict=True,
    )
    assert "iran-us" not in report_fail["meta"].get("domains", []), (
        "iran-us should NOT be in domains with min_thread_snapshots=3"
    )


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


# --- Prediction audit ---


async def test_generate_prediction_audit_returns_per_question_detail(store):
    """generate_prediction_audit should produce per-question rows with Brier scores."""
    event_a = Event(date=date(2026, 3, 1), summary="Sanctions announced", entities=["Iran"])
    event_b = Event(date=date(2026, 3, 8), summary="Follow-on sanctions filing", entities=["Iran"])
    event_c = Event(date=date(2026, 3, 10), summary="Treasury statement", entities=["Iran"])
    event_d = Event(date=date(2026, 3, 16), summary="New sanctions package", entities=["Iran"])
    event_ids = await store.add_events([event_a, event_b, event_c, event_d], "iran-us")
    thread_id = await store.upsert_thread("iran-thread", "Iran Thread", 8, "active")
    await store.link_thread_topic(thread_id, "iran-us")
    await store.link_thread_events(thread_id, event_ids)
    for snap_date, ev_count in [(date(2026, 3, 1), 1), (date(2026, 3, 8), 2), (date(2026, 3, 9), 2)]:
        await store.upsert_thread_snapshot(ThreadSnapshot(
            thread_id=thread_id, snapshot_date=snap_date, status="active",
            significance=8, event_count=ev_count,
            latest_event_date=min(snap_date, date(2026, 3, 8)),
        ))
    await store.save_synthesis(
        TopicSynthesis(
            topic_name="Iran-US",
            threads=[NarrativeThread(
                headline="Iran Thread", significance=8,
                thread_id=thread_id, slug="iran-thread", status="active", snapshot_count=3,
                events=[
                    Event(event_id=event_ids[0], date=date(2026, 3, 1), summary="Sanctions announced", entities=["Iran"]),
                    Event(event_id=event_ids[1], date=date(2026, 3, 8), summary="Follow-on sanctions filing", entities=["Iran"]),
                ],
            )],
        ).model_dump(mode="json"),
        "iran-us",
        date(2026, 3, 9),
    )
    config = NexusConfig(
        user=UserConfig(name="Test User"),
        topics=[TopicConfig(name="Iran-US")],
        future_projection=FutureProjectionConfig(enabled=True, min_history_days=1, min_thread_snapshots=1),
    )

    audit = await generate_prediction_audit(
        store, config,
        start=date(2026, 3, 9), end=date(2026, 3, 9),
        engines=["baseline", "native"],
    )

    assert audit["meta"]["total_questions"] > 0
    assert audit["meta"]["engines"] == ["baseline", "native"]
    assert len(audit["per_question"]) > 0
    for q in audit["per_question"]:
        assert "brier_score" in q
        assert "topic_slug" in q
        assert "target_variable" in q
    assert "iran-us" in audit["topic_breakdown"]
    assert len(audit["target_variable_breakdown"]) > 0
    assert len(audit["best_predictions"]) > 0


def test_render_prediction_audit_markdown_includes_tables():
    """Markdown renderer should include engine summary and topic tables."""
    audit = {
        "meta": {
            "start": "2026-03-01",
            "end": "2026-03-15",
            "engines": ["native", "baseline"],
            "total_questions": 10,
        },
        "benchmark_summary": {
            "meta": {"significance_test": {"p_value": 0.03, "n": 10, "significant_at_005": True}},
            "engines": {
                "native": {"total": 10, "accuracy": 0.7, "mean_brier": 0.18, "mean_log_loss": 0.5},
                "baseline": {"total": 10, "accuracy": 0.5, "mean_brier": 0.25, "mean_log_loss": 0.7},
            },
        },
        "topic_breakdown": {
            "iran-us": {"total": 6, "accuracy": 0.8, "mean_brier": 0.15},
            "ai-ml-research": {"total": 4, "accuracy": 0.5, "mean_brier": 0.22},
        },
        "target_variable_breakdown": {
            "thread_new_event_count": {"total": 7, "accuracy": 0.7, "mean_brier": 0.17},
            "entity_recurrence": {"total": 3, "accuracy": 0.6, "mean_brier": 0.21},
        },
        "best_predictions": [{
            "topic_slug": "iran-us",
            "question": "Will sanctions continue?",
            "probability": 0.7,
            "resolved_bool": True,
            "brier_score": 0.09,
            "engine": "native",
            "cutoff": "2026-03-09",
            "horizon_days": 7,
        }],
        "worst_predictions": [{
            "topic_slug": "ai-ml-research",
            "question": "Will a new model be released?",
            "probability": 0.8,
            "resolved_bool": False,
            "brier_score": 0.64,
            "engine": "native",
            "cutoff": "2026-03-09",
            "horizon_days": 7,
        }],
    }

    md = render_prediction_audit_markdown(audit)
    assert "# Prediction Audit Report" in md
    assert "Engine Performance Summary" in md
    assert "Per-Topic Breakdown" in md
    assert "Per-Target Variable" in md
    assert "Best Predictions" in md
    assert "Worst Predictions" in md
    assert "iran-us" in md
    assert "significant" in md.lower()
