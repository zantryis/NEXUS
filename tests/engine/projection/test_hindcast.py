"""Tests for the hindcast benchmark pipeline."""

from datetime import date, timedelta
import json

import pytest

from nexus.engine.knowledge.events import Event
from nexus.engine.knowledge.store import KnowledgeStore
from nexus.engine.projection.hindcast import (
    BacktestReport,
    HindcastCase,
    _run_engine,
    backtest_forecasts,
    compute_calibration,
    generate_hindcast_questions,
    sample_negative_cases,
    select_hindcast_events,
)


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
async def store(tmp_path):
    s = KnowledgeStore(tmp_path / "hindcast.db")
    await s.initialize()
    yield s
    await s.close()


class _MockLLM:
    """Minimal mock returning valid JSON for question generation and engines."""

    def __init__(self, responses: list[str] | None = None):
        self.calls: list[dict] = []
        self._responses = list(responses) if responses else []
        self._idx = 0
        self.usage = type("Usage", (), {"cost_summary": lambda self: {}})()

    async def complete(self, config_key, system_prompt, user_prompt, **kwargs):
        self.calls.append({
            "config_key": config_key,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
        })
        if self._responses:
            resp = self._responses[self._idx % len(self._responses)]
            self._idx += 1
            return resp
        # Default: return a structural-engine-shaped response
        return json.dumps({
            "verdict": "yes",
            "confidence": "medium",
            "factors": [{"factor": "test", "direction": "supports_yes",
                         "weight": "moderate", "source_type": "world_knowledge"}],
            "reasoning": "Test reasoning.",
            "contrarian_view": "None.",
            "key_uncertainties": ["unknown"],
            "signposts": ["watch for X"],
            "base_rate_reasoning": "Base rate ~50%.",
            "contrarian_argument": "Could go either way.",
            "wildcards": ["surprise event"],
            "base_rate_critique": "Base rate may be off.",
        })


async def _seed_topic_with_events(store, topic_slug="test-topic", n_events=5):
    """Seed a topic with events at various significance levels."""
    events = []
    base = date(2026, 2, 1)
    entities_pool = ["Alpha Corp", "Beta Inc", "Gamma Ltd", "Delta Co", "Epsilon AG"]
    for i in range(n_events):
        sig = 8 if i < 3 else 4  # first 3 high significance, last 2 low
        events.append(Event(
            date=base + timedelta(days=i * 3),
            summary=f"Event {i}: {'Major' if sig >= 7 else 'Minor'} development involving {entities_pool[i]}",
            entities=[entities_pool[i]],
            significance=sig,
        ))
    event_ids = await store.add_events(events, topic_slug)

    # Register entities so they're findable
    for name in entities_pool:
        await store.upsert_entity(name, entity_type="org")

    return event_ids, events, entities_pool


# ── Test 1: Event selection by significance ──────────────────────────


async def test_select_hindcast_events_filters_by_significance(store):
    """Only events with significance >= threshold are selected."""
    await _seed_topic_with_events(store, "test-topic", n_events=5)

    selected = await select_hindcast_events(store, "test-topic",
                                            start=date(2026, 1, 1), end=date(2026, 3, 1),
                                            min_significance=7)
    assert len(selected) == 3
    assert all(e.significance >= 7 for e in selected)


# ── Test 2: Question generation with LLM ────────────────────────────


async def test_generate_questions_llm_path(store):
    """LLM converts events into natural prediction questions."""
    await _seed_topic_with_events(store)
    events = [
        Event(date=date(2026, 2, 15), summary="Alpha Corp announced tariffs",
              entities=["Alpha Corp"], significance=8),
    ]
    llm = _MockLLM(responses=[
        json.dumps({
            "question": "Will Alpha Corp impose new tariffs by February 15, 2026?",
            "primary_entity": "Alpha Corp",
        }),
    ])

    cases = await generate_hindcast_questions(
        llm, events, "test-topic", "Test Topic", horizon_days=7,
    )
    assert len(cases) == 1
    case = cases[0]
    assert case.outcome is True
    assert case.case_type == "positive"
    assert case.cutoff_date == date(2026, 2, 15) - timedelta(days=7)
    assert case.resolution_date == date(2026, 2, 15)
    assert "Alpha Corp" in case.question or "tariff" in case.question.lower()
    assert llm.calls  # LLM was actually called


# ── Test 3: Question generation template fallback ────────────────────


async def test_generate_questions_template_fallback(store):
    """When llm=None, template-based questions are generated."""
    events = [
        Event(date=date(2026, 2, 15), summary="Alpha Corp announced tariffs",
              entities=["Alpha Corp"], significance=8),
        Event(date=date(2026, 2, 20), summary="Beta Inc signed deal",
              entities=["Beta Inc"], significance=9),
    ]

    cases = await generate_hindcast_questions(
        None, events, "test-topic", "Test Topic", horizon_days=7,
    )
    assert len(cases) == 2
    for case in cases:
        assert case.outcome is True
        assert case.case_type == "positive"
        assert case.source_entity  # entity name is populated
        assert case.source_entity in case.question


# ── Test 4: Negative sampling produces balanced set ──────────────────


async def test_negative_sampling_balanced(store):
    """Entities active before cutoff but quiet during window become negatives."""
    base = date(2026, 2, 1)
    # Entity A: active + significant event in forecast window
    events_a = [
        Event(date=base, summary="A setup", entities=["Alpha Corp"], significance=6),
        Event(date=base + timedelta(days=10), summary="A significant",
              entities=["Alpha Corp"], significance=8),
    ]
    # Entity B: active before cutoff, quiet during window
    events_b = [
        Event(date=base, summary="B setup", entities=["Beta Inc"], significance=6),
        Event(date=base + timedelta(days=1), summary="B minor",
              entities=["Beta Inc"], significance=3),
    ]
    await store.add_events(events_a + events_b, "test-topic")
    await store.upsert_entity("Alpha Corp", entity_type="org")
    await store.upsert_entity("Beta Inc", entity_type="org")

    positive_cases = [
        HindcastCase(
            case_id="test-topic:2026-02-04:Alpha Corp",
            topic_slug="test-topic",
            question="Will Alpha Corp do something?",
            cutoff_date=base + timedelta(days=3),  # Feb 4
            resolution_date=base + timedelta(days=10),  # Feb 11
            horizon_days=7,
            outcome=True,
            source_event_id=None,
            source_entity="Alpha Corp",
            case_type="positive",
        ),
    ]

    negatives = await sample_negative_cases(
        store, "test-topic", "Test Topic", positive_cases, horizon_days=7,
    )
    # Beta Inc was active but had no sig>=7 events in window → negative case
    negative_entities = {c.source_entity for c in negatives}
    assert "Alpha Corp" not in negative_entities  # excluded (in positive set)
    assert len(negatives) >= 1


# ── Test 5: Negative cases exclude positive entities ─────────────────


async def test_negative_excludes_positive_entities(store):
    """An entity used in a positive case is not reused as negative for same window."""
    base = date(2026, 2, 1)
    events = [
        Event(date=base, summary="Setup", entities=["Alpha Corp"], significance=6),
        Event(date=base + timedelta(days=8), summary="Big event",
              entities=["Alpha Corp"], significance=9),
    ]
    await store.add_events(events, "test-topic")
    await store.upsert_entity("Alpha Corp", entity_type="org")

    positive_cases = [
        HindcastCase(
            case_id="test-topic:2026-02-04:Alpha Corp",
            topic_slug="test-topic",
            question="Will Alpha Corp do something?",
            cutoff_date=base + timedelta(days=3),
            resolution_date=base + timedelta(days=10),
            horizon_days=7,
            outcome=True,
            source_event_id=None,
            source_entity="Alpha Corp",
            case_type="positive",
        ),
    ]

    negatives = await sample_negative_cases(
        store, "test-topic", "Test Topic", positive_cases, horizon_days=7,
    )
    for case in negatives:
        assert case.source_entity != "Alpha Corp"


# ── Test 6: End-to-end backtest ──────────────────────────────────────


async def test_backtest_end_to_end(store):
    """Full backtest run with mocked LLM produces valid BacktestReport."""
    event_ids, events, entities = await _seed_topic_with_events(store, "test-topic")

    # Need a thread + snapshot for the structural engine evidence path
    thread_id = await store.upsert_thread("test-thread", "Test Thread", 8, "active")
    await store.link_thread_topic(thread_id, "test-topic")
    await store.link_thread_events(thread_id, event_ids[:3])

    from nexus.engine.projection.models import ThreadSnapshot
    await store.upsert_thread_snapshot(ThreadSnapshot(
        thread_id=thread_id, snapshot_date=date(2026, 2, 1),
        status="active", significance=8, event_count=1,
        latest_event_date=date(2026, 2, 1),
    ))

    # Mock LLM that handles both question generation and engine calls
    llm = _MockLLM()

    report = await backtest_forecasts(
        store, llm, topics=[("test-topic", "Test Topic")],
        start=date(2026, 1, 1), end=date(2026, 3, 1),
        engines=["structural"],
        horizon_days=7,
        min_significance=7,
        max_cases_per_topic=5,
        persist=False,
    )

    assert isinstance(report, BacktestReport)
    assert report.total_cases > 0
    assert report.positive_cases > 0
    assert "structural" in report.engine_results
    briers = report.engine_results["structural"]["brier_scores"]
    assert all(0.0 <= b <= 1.0 for b in briers)


# ── Test 7: Leakage safety ──────────────────────────────────────────


async def test_leakage_safety(store):
    """Evidence assembly receives correct as_of cutoff, never the event date."""
    base = date(2026, 2, 1)
    events = [
        Event(date=base, summary="Early event", entities=["Alpha Corp"], significance=5),
        Event(date=base + timedelta(days=4), summary="Mid event",
              entities=["Alpha Corp"], significance=6),
        Event(date=base + timedelta(days=9), summary="Target event",
              entities=["Alpha Corp"], significance=8),
    ]
    await store.add_events(events, "test-topic")
    await store.upsert_entity("Alpha Corp", entity_type="org")

    # Generate cases for the day-9 event with horizon=7 → cutoff = day 2
    target_events = [e for e in events if e.significance >= 7]
    cases = await generate_hindcast_questions(
        None, target_events, "test-topic", "Test Topic", horizon_days=7,
    )

    assert len(cases) == 1
    case = cases[0]
    # Cutoff must be event.date - horizon, not the event date itself
    assert case.cutoff_date == base + timedelta(days=9) - timedelta(days=7)
    assert case.cutoff_date == date(2026, 2, 3)
    # Resolution date is the event date
    assert case.resolution_date == base + timedelta(days=9)


# ── Test 8: Persistence ─────────────────────────────────────────────


async def test_persistence(store):
    """When persist=True, forecast runs and resolutions are saved to the store."""
    event_ids, events, entities = await _seed_topic_with_events(store, "test-topic")

    thread_id = await store.upsert_thread("test-thread", "Test Thread", 8, "active")
    await store.link_thread_topic(thread_id, "test-topic")
    await store.link_thread_events(thread_id, event_ids[:3])

    from nexus.engine.projection.models import ThreadSnapshot
    await store.upsert_thread_snapshot(ThreadSnapshot(
        thread_id=thread_id, snapshot_date=date(2026, 2, 1),
        status="active", significance=8, event_count=1,
        latest_event_date=date(2026, 2, 1),
    ))

    llm = _MockLLM()

    report = await backtest_forecasts(
        store, llm, topics=[("test-topic", "Test Topic")],
        start=date(2026, 1, 1), end=date(2026, 3, 1),
        engines=["structural"],
        horizon_days=7,
        persist=True,
    )

    # Check forecasts were persisted
    saved = await store.get_forecast_questions_between(
        start=date(2026, 1, 1), end=date(2026, 3, 1),
    )
    hindcast_questions = [q for q in saved if q.get("target_variable") == "hindcast"]
    assert len(hindcast_questions) > 0

    # Check resolutions were saved (resolved_bool should be set)
    resolved = [q for q in hindcast_questions if q.get("outcome_status") == "resolved"]
    assert len(resolved) > 0


# ── Test 9: Calibration computation ─────────────────────────────────


def test_calibration_computation():
    """Decile buckets are computed correctly from known inputs."""
    results = [
        {"probability": 0.9, "outcome": True},
        {"probability": 0.85, "outcome": True},
        {"probability": 0.8, "outcome": False},
        {"probability": 0.2, "outcome": False},
        {"probability": 0.15, "outcome": False},
        {"probability": 0.1, "outcome": True},
    ]

    cal = compute_calibration(results)
    assert isinstance(cal, list)
    assert all("bucket" in b and "predicted_mean" in b and "actual_rate" in b for b in cal)

    # The 0.8-0.9 bucket should have 3 entries (0.8, 0.85, 0.9)
    high_bucket = [b for b in cal if 0.8 <= b["bucket"] < 1.0]
    assert len(high_bucket) >= 1
    # 2 out of 3 high-confidence predictions were correct
    for b in high_bucket:
        if b["count"] == 3:
            assert abs(b["actual_rate"] - 2 / 3) < 0.01

    # Low bucket should show low actual rate
    low_bucket = [b for b in cal if b["bucket"] < 0.3]
    assert len(low_bucket) >= 1


# ── Test 10: Engine dispatch covers all engines ──────────────────────


async def test_run_engine_graphrag(store):
    """GraphRAG engine dispatch returns a probability."""
    await _seed_topic_with_events(store)
    llm = _MockLLM(responses=[
        # Entity extraction response
        json.dumps({"entities": ["Alpha Corp"]}),
        # Graph reasoning response
        json.dumps({"probability": 0.65, "reasoning": "test", "key_uncertainties": [], "signposts": []}),
    ])
    prob = await _run_engine("graphrag", store, llm, "Will Alpha Corp expand?", cutoff=date(2026, 2, 15))
    assert 0.02 <= prob <= 0.98


async def test_run_engine_naked(store):
    """Naked engine dispatch returns a probability."""
    await _seed_topic_with_events(store)
    llm = _MockLLM(responses=[
        json.dumps({"probability": 0.55}),
    ])
    prob = await _run_engine("naked", store, llm, "Will something happen?", cutoff=date(2026, 2, 15))
    assert 0.02 <= prob <= 0.98


async def test_run_engine_unknown_raises():
    """Unknown engine name should raise ValueError."""
    from unittest.mock import AsyncMock
    with pytest.raises(ValueError, match="Unknown engine"):
        await _run_engine("nonexistent", AsyncMock(), None, "test?", cutoff=date(2026, 1, 1))
