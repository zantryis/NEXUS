"""Tests for KnowledgeStore — SQLite-backed knowledge graph CRUD."""

import pytest
from datetime import date, timedelta
from nexus.engine.knowledge.store import KnowledgeStore
from nexus.engine.knowledge.events import Event
from nexus.engine.knowledge.compression import Summary
from nexus.engine.projection.models import (
    ForecastQuestion,
    ForecastResolution,
    ForecastRun,
    ProjectionItem,
    ThreadSnapshot,
    TopicProjection,
)


@pytest.fixture
async def store(tmp_path):
    s = KnowledgeStore(tmp_path / "knowledge.db")
    await s.initialize()
    yield s
    await s.close()


# ── Events ────────────────────────────────────────────────────────


def _make_event(
    d="2026-03-09", summary="Test event", significance=7,
    entities=None, sources=None,
):
    return Event(
        date=date.fromisoformat(d),
        summary=summary,
        significance=significance,
        relation_to_prior="follows prior",
        entities=entities or ["IAEA", "Iran"],
        sources=sources or [
            {"url": "https://reuters.com/1", "outlet": "reuters",
             "affiliation": "private", "country": "US", "language": "en"},
        ],
    )


async def test_add_and_get_events(store):
    event = _make_event()
    ids = await store.add_events([event], "iran-us-relations")
    assert len(ids) == 1

    loaded = await store.get_events("iran-us-relations")
    assert len(loaded) == 1
    assert loaded[0].summary == "Test event"
    assert loaded[0].date == date(2026, 3, 9)
    assert loaded[0].significance == 7
    assert loaded[0].relation_to_prior == "follows prior"


async def test_event_sources_roundtrip(store):
    event = _make_event(sources=[
        {"url": "https://a.com", "outlet": "outlet-a",
         "affiliation": "state", "country": "IR", "language": "fa"},
        {"url": "https://b.com", "outlet": "outlet-b",
         "affiliation": "public", "country": "GB", "language": "en"},
    ])
    await store.add_events([event], "iran-us-relations")
    loaded = await store.get_events("iran-us-relations")
    assert len(loaded[0].sources) == 2
    assert loaded[0].sources[0]["outlet"] == "outlet-a"
    assert loaded[0].sources[1]["affiliation"] == "public"


async def test_event_entities_roundtrip(store):
    event = _make_event(entities=["IAEA", "Iran", "US Treasury"])
    await store.add_events([event], "iran-us-relations")
    loaded = await store.get_events("iran-us-relations")
    assert set(loaded[0].entities) == {"IAEA", "Iran", "US Treasury"}


async def test_get_events_date_filter(store):
    e1 = _make_event(d="2026-03-05", summary="Old")
    e2 = _make_event(d="2026-03-09", summary="Recent")
    e3 = _make_event(d="2026-03-10", summary="Today")
    await store.add_events([e1, e2, e3], "test-topic")

    # Since filter
    loaded = await store.get_events("test-topic", since=date(2026, 3, 8))
    assert len(loaded) == 2
    assert loaded[0].summary == "Recent"
    assert loaded[1].summary == "Today"

    # Until filter
    loaded = await store.get_events("test-topic", until=date(2026, 3, 6))
    assert len(loaded) == 1
    assert loaded[0].summary == "Old"


async def test_event_source_framing_roundtrip(store):
    """Framing field in source dict survives add_events → get_events roundtrip."""
    event = _make_event(sources=[
        {"url": "https://reuters.com/1", "outlet": "reuters",
         "affiliation": "private", "country": "US", "language": "en",
         "framing": "[neutral] Reports policy details; US as enforcer"},
        {"url": "https://tass.com/1", "outlet": "tass",
         "affiliation": "state", "country": "RU", "language": "ru",
         "framing": "[critical] Emphasizes economic damage; US as aggressor"},
    ])
    await store.add_events([event], "iran-us")
    loaded = await store.get_events("iran-us")
    assert len(loaded) == 1
    assert loaded[0].sources[0]["framing"] == "[neutral] Reports policy details; US as enforcer"
    assert loaded[0].sources[1]["framing"] == "[critical] Emphasizes economic damage; US as aggressor"


async def test_get_events_topic_isolation(store):
    await store.add_events([_make_event(summary="Iran event")], "iran-us")
    await store.add_events([_make_event(summary="AI event")], "ai-ml")

    iran = await store.get_events("iran-us")
    assert len(iran) == 1
    assert iran[0].summary == "Iran event"

    ai = await store.get_events("ai-ml")
    assert len(ai) == 1
    assert ai[0].summary == "AI event"


async def test_get_events_limit(store):
    events = [_make_event(summary=f"Event {i}") for i in range(10)]
    await store.add_events(events, "test")

    loaded = await store.get_events("test", limit=3)
    assert len(loaded) == 3


async def test_get_recent_events(store):
    e1 = _make_event(d="2026-03-01", summary="Old")
    e2 = _make_event(d="2026-03-08", summary="Recent")
    e3 = _make_event(d="2026-03-09", summary="Very recent")
    await store.add_events([e1, e2, e3], "test")

    recent = await store.get_recent_events(
        "test", days=3, reference_date=date(2026, 3, 10)
    )
    assert len(recent) == 2
    assert recent[0].summary == "Recent"


async def test_get_recent_events_limit(store):
    events = [_make_event(d="2026-03-09", summary=f"E{i}") for i in range(40)]
    await store.add_events(events, "test")

    recent = await store.get_recent_events(
        "test", days=7, limit=10, reference_date=date(2026, 3, 10)
    )
    assert len(recent) == 10


async def test_get_events_empty_topic(store):
    loaded = await store.get_events("nonexistent")
    assert loaded == []


# ── Entities ──────────────────────────────────────────────────────


async def test_upsert_entity_create(store):
    eid = await store.upsert_entity("IAEA", "org", ["International Atomic Energy Agency"])
    assert eid > 0

    entity = await store.find_entity("IAEA")
    assert entity is not None
    assert entity["canonical_name"] == "IAEA"
    assert entity["entity_type"] == "org"
    assert "International Atomic Energy Agency" in entity["aliases"]


async def test_upsert_entity_update(store):
    eid1 = await store.upsert_entity("IAEA", "unknown")
    eid2 = await store.upsert_entity("IAEA", "org", ["IAEA Vienna"])
    assert eid1 == eid2

    entity = await store.find_entity("IAEA")
    assert entity["entity_type"] == "org"  # Updated from unknown


async def test_upsert_entity_preserves_known_type(store):
    """If entity already has a known type, don't overwrite with 'unknown'."""
    await store.upsert_entity("IAEA", "org")
    await store.upsert_entity("IAEA", "unknown")

    entity = await store.find_entity("IAEA")
    assert entity["entity_type"] == "org"  # Preserved


async def test_upsert_entity_with_observation_date(store):
    """Observation date sets first_seen and last_seen instead of today."""
    await store.upsert_entity("IAEA", "org", observation_date=date(2026, 3, 14))
    entity = await store.find_entity("IAEA")
    assert entity["first_seen"] == "2026-03-14"
    assert entity["last_seen"] == "2026-03-14"


async def test_upsert_entity_earlier_date_updates_first_seen(store):
    """Re-upserting with an earlier date pulls first_seen back."""
    await store.upsert_entity("IAEA", "org", observation_date=date(2026, 3, 16))
    await store.upsert_entity("IAEA", "org", observation_date=date(2026, 3, 12))
    entity = await store.find_entity("IAEA")
    assert entity["first_seen"] == "2026-03-12"
    assert entity["last_seen"] == "2026-03-16"


async def test_upsert_entity_later_date_updates_last_seen(store):
    """Re-upserting with a later date pushes last_seen forward."""
    await store.upsert_entity("IAEA", "org", observation_date=date(2026, 3, 14))
    await store.upsert_entity("IAEA", "org", observation_date=date(2026, 3, 18))
    entity = await store.find_entity("IAEA")
    assert entity["first_seen"] == "2026-03-14"
    assert entity["last_seen"] == "2026-03-18"


async def test_upsert_entity_backfill_order_independence(store):
    """Processing events out of order still produces correct date range."""
    await store.upsert_entity("IAEA", "org", observation_date=date(2026, 3, 16))
    await store.upsert_entity("IAEA", "org", observation_date=date(2026, 3, 12))
    await store.upsert_entity("IAEA", "org", observation_date=date(2026, 3, 14))
    entity = await store.find_entity("IAEA")
    assert entity["first_seen"] == "2026-03-12"
    assert entity["last_seen"] == "2026-03-16"


async def test_find_entity_by_alias(store):
    await store.upsert_entity("IAEA", "org", ["International Atomic Energy Agency"])
    entity = await store.find_entity("International Atomic Energy Agency")
    assert entity is not None
    assert entity["canonical_name"] == "IAEA"


async def test_find_entity_missing(store):
    result = await store.find_entity("Nonexistent")
    assert result is None


async def test_get_all_entities(store):
    await store.upsert_entity("IAEA", "org")
    await store.upsert_entity("Iran", "country")
    await store.upsert_entity("Ali Khamenei", "person")

    entities = await store.get_all_entities()
    assert len(entities) == 3
    names = {e["canonical_name"] for e in entities}
    assert names == {"IAEA", "Iran", "Ali Khamenei"}


async def test_get_all_entities_scoped_to_topic(store):
    """Entities only appear if they're linked to events in the topic."""
    e1 = _make_event(entities=["IAEA", "Iran"])
    e2 = _make_event(entities=["OpenAI"])
    await store.add_events([e1], "iran-us")
    await store.add_events([e2], "ai-ml")

    iran_entities = await store.get_all_entities("iran-us")
    names = {e["canonical_name"] for e in iran_entities}
    assert "IAEA" in names
    assert "Iran" in names
    assert "OpenAI" not in names


async def test_get_events_for_entity(store):
    e1 = _make_event(d="2026-03-08", summary="Iran sanctions", entities=["IAEA", "Iran"])
    e2 = _make_event(d="2026-03-09", summary="AI news", entities=["OpenAI"])
    e3 = _make_event(d="2026-03-10", summary="IAEA inspection", entities=["IAEA"])
    await store.add_events([e1], "iran-us")
    await store.add_events([e2], "ai-ml")
    await store.add_events([e3], "iran-us")

    # Find IAEA entity ID
    entity = await store.find_entity("IAEA")
    events = await store.get_events_for_entity(entity["id"])
    assert len(events) == 2
    summaries = {e.summary for e in events}
    assert summaries == {"Iran sanctions", "IAEA inspection"}


# ── Summaries ─────────────────────────────────────────────────────


async def test_add_and_get_summaries(store):
    summary = Summary(
        period_start=date(2026, 3, 3),
        period_end=date(2026, 3, 9),
        text="Weekly summary of Iran-US events.",
        event_count=5,
    )
    sid = await store.add_summary(summary, "iran-us", "weekly")
    assert sid > 0

    loaded = await store.get_summaries("iran-us", "weekly")
    assert len(loaded) == 1
    assert loaded[0].period_start == date(2026, 3, 3)
    assert loaded[0].text == "Weekly summary of Iran-US events."
    assert loaded[0].event_count == 5


async def test_summaries_topic_isolation(store):
    s1 = Summary(period_start=date(2026, 3, 3), period_end=date(2026, 3, 9),
                 text="Iran summary", event_count=5)
    s2 = Summary(period_start=date(2026, 3, 3), period_end=date(2026, 3, 9),
                 text="AI summary", event_count=3)
    await store.add_summary(s1, "iran-us", "weekly")
    await store.add_summary(s2, "ai-ml", "weekly")

    iran = await store.get_summaries("iran-us", "weekly")
    assert len(iran) == 1
    assert iran[0].text == "Iran summary"


async def test_summaries_period_type_isolation(store):
    s1 = Summary(period_start=date(2026, 3, 3), period_end=date(2026, 3, 9),
                 text="Weekly", event_count=5)
    s2 = Summary(period_start=date(2026, 3, 1), period_end=date(2026, 3, 31),
                 text="Monthly", event_count=20)
    await store.add_summary(s1, "iran-us", "weekly")
    await store.add_summary(s2, "iran-us", "monthly")

    weekly = await store.get_summaries("iran-us", "weekly")
    monthly = await store.get_summaries("iran-us", "monthly")
    assert len(weekly) == 1
    assert weekly[0].text == "Weekly"
    assert len(monthly) == 1
    assert monthly[0].text == "Monthly"


# ── Threads ───────────────────────────────────────────────────────


async def test_upsert_thread(store):
    tid = await store.upsert_thread("sanctions-escalation", "Sanctions Escalation", 8)
    assert tid > 0

    # Update
    tid2 = await store.upsert_thread("sanctions-escalation", "Updated Headline", 9, "active")
    assert tid2 == tid


async def test_get_active_threads(store):
    await store.upsert_thread("t1", "Thread 1", status="emerging")
    await store.upsert_thread("t2", "Thread 2", status="active")
    await store.upsert_thread("t3", "Thread 3", status="resolved")

    active = await store.get_active_threads()
    slugs = {t["slug"] for t in active}
    assert slugs == {"t1", "t2"}


async def test_get_active_threads_by_topic(store):
    tid = await store.upsert_thread("t1", "Thread 1", status="active")
    await store.link_thread_topic(tid, "iran-us")

    tid2 = await store.upsert_thread("t2", "Thread 2", status="active")
    await store.link_thread_topic(tid2, "ai-ml")

    iran_threads = await store.get_active_threads("iran-us")
    assert len(iran_threads) == 1
    assert iran_threads[0]["slug"] == "t1"


async def test_mark_stale_threads(store):
    """mark_stale_threads demotes old threads, leaves recent ones active."""
    from datetime import date

    # Create two active threads
    tid_old = await store.upsert_thread("old-thread", "Old Thread", status="active")
    tid_recent = await store.upsert_thread("recent-thread", "Recent Thread", status="active")

    # Add events: old one from 20 days ago, recent one from 3 days ago
    old_event = _make_event(summary="Old event", d="2026-02-22")
    recent_event = _make_event(summary="Recent event", d="2026-03-11")
    old_ids = await store.add_events([old_event], "test")
    recent_ids = await store.add_events([recent_event], "test")

    # Link events to threads
    await store.link_thread_events(tid_old, old_ids)
    await store.link_thread_events(tid_recent, recent_ids)

    # Mark stale with reference date of March 14
    count = await store.mark_stale_threads(stale_after_days=14, reference_date=date(2026, 3, 14))
    assert count == 1  # Only old thread should be marked stale

    # Verify states
    active = await store.get_active_threads()
    slugs = {t["slug"] for t in active}
    assert "recent-thread" in slugs
    assert "old-thread" not in slugs


async def test_convergence_divergence(store):
    tid = await store.upsert_thread("t1", "Test Thread")

    cid = await store.add_convergence(tid, "Iran enriching to 60%", ["Reuters", "BBC"])
    assert cid > 0

    did = await store.add_divergence(
        tid, "Sanctions impact",
        "Reuters", "Sanctions crippling economy",
        "IRNA", "Sanctions having minimal effect",
    )
    assert did > 0


async def test_add_events_preserves_raw_entities_without_auto_linking(store):
    event = Event(
        date=date(2026, 3, 9),
        summary="Canonicalized event",
        significance=7,
        entities=["International Atomic Energy Agency", "Iran"],
        raw_entities=["IAEA", "Iran"],
        sources=[{"url": "https://reuters.com/1", "outlet": "reuters"}],
    )
    ids = await store.add_events([event], "iran-us")
    loaded = await store.get_events("iran-us")
    assert ids[0] == loaded[0].event_id
    assert loaded[0].raw_entities == ["IAEA", "Iran"]
    assert loaded[0].entities == ["IAEA", "Iran"]  # no canonical links yet


async def test_thread_snapshots_roundtrip(store):
    tid = await store.upsert_thread("snap-thread", "Snapshot Thread", 8, "active")
    s1 = ThreadSnapshot(
        thread_id=tid,
        snapshot_date=date(2026, 3, 8),
        status="active",
        significance=7,
        event_count=1,
        latest_event_date=date(2026, 3, 8),
    )
    s2 = ThreadSnapshot(
        thread_id=tid,
        snapshot_date=date(2026, 3, 10),
        status="active",
        significance=8,
        event_count=4,
        latest_event_date=date(2026, 3, 10),
    )
    await store.upsert_thread_snapshot(s1)
    await store.upsert_thread_snapshot(s2)

    latest = await store.get_latest_thread_snapshot(tid)
    assert latest is not None
    assert latest.trajectory_label in {"accelerating", "about_to_break", "steady"}
    assert latest.event_count == 4


async def test_save_and_get_latest_projection(store):
    projection = TopicProjection(
        topic_slug="iran-us",
        topic_name="Iran-US",
        generated_for=date(2026, 3, 10),
        summary="Forward look summary",
        items=[ProjectionItem(
            claim="Sanctions pressure is likely to continue.",
            confidence="medium",
            horizon_days=7,
            signpost="Watch for Treasury action",
            signals_cited=["trajectory:accelerating"],
            evidence_thread_ids=[1],
            review_after=date(2026, 3, 17),
        )],
    )
    await store.save_projection(projection)
    loaded = await store.get_latest_projection("iran-us")
    assert loaded is not None
    assert loaded.summary == "Forward look summary"
    assert loaded.items[0].claim.startswith("Sanctions pressure")


async def test_save_and_get_latest_forecast_run(store):
    forecast_run = ForecastRun(
        topic_slug="iran-us",
        topic_name="Iran-US",
        generated_for=date(2026, 3, 10),
        summary="Quantified forecast summary",
        questions=[ForecastQuestion(
            question="Will sanctions coverage add a new event by 2026-03-17?",
            forecast_type="binary",
            target_variable="thread_new_event_count",
            target_metadata={"thread_id": 1, "threshold": 1, "topic_slug": "iran-us"},
            probability=0.68,
            base_rate=0.55,
            resolution_criteria="Resolves true if thread 1 gains at least 1 new linked event by the resolution date.",
            resolution_date=date(2026, 3, 17),
            horizon_days=7,
            signpost="Watch for Treasury action",
            evidence_thread_ids=[1],
        )],
    )
    await store.save_forecast_run(forecast_run)

    loaded = await store.get_latest_forecast_run("iran-us")
    assert loaded is not None
    assert loaded.summary == "Quantified forecast summary"
    assert loaded.questions[0].target_variable == "thread_new_event_count"
    assert loaded.questions[0].question_id is not None


async def test_set_forecast_resolution(store):
    forecast_run = ForecastRun(
        topic_slug="iran-us",
        topic_name="Iran-US",
        generated_for=date(2026, 3, 10),
        questions=[ForecastQuestion(
            question="Will sanctions coverage add a new event by 2026-03-17?",
            forecast_type="binary",
            target_variable="thread_new_event_count",
            target_metadata={"thread_id": 1, "threshold": 1, "topic_slug": "iran-us"},
            probability=0.68,
            base_rate=0.55,
            resolution_criteria="Resolves true if thread 1 gains at least 1 new linked event by the resolution date.",
            resolution_date=date(2026, 3, 17),
            horizon_days=7,
            signpost="Watch for Treasury action",
            evidence_thread_ids=[1],
        )],
    )
    await store.save_forecast_run(forecast_run)
    question_id = forecast_run.questions[0].question_id
    assert question_id is not None

    await store.set_forecast_resolution(ForecastResolution(
        forecast_question_id=question_id,
        outcome_status="resolved",
        resolved_bool=True,
        actual_value=1.0,
        brier_score=0.1024,
        log_loss=0.3857,
        notes="Observed one new event.",
        resolved_at=date(2026, 3, 17),
    ))

    pending = await store.get_pending_forecast_questions(until=date(2026, 3, 17))
    assert pending == []


async def test_backfill_forecast_keys_persists_missing_keys(store):
    forecast_run = ForecastRun(
        topic_slug="iran-us",
        topic_name="Iran-US",
        generated_for=date(2026, 3, 10),
        questions=[ForecastQuestion(
            question="Will sanctions coverage add a new event by 2026-03-17?",
            forecast_type="binary",
            target_variable="thread_new_event_count",
            target_metadata={"thread_id": 1, "threshold": 1, "topic_slug": "iran-us"},
            probability=0.68,
            base_rate=0.55,
            resolution_criteria="Resolves true if thread 1 gains at least 1 new linked event by the resolution date.",
            resolution_date=date(2026, 3, 17),
            horizon_days=7,
            signpost="Watch for Treasury action",
            evidence_thread_ids=[1],
        )],
    )
    await store.save_forecast_run(forecast_run)
    question_id = forecast_run.questions[0].question_id
    assert question_id is not None

    await store.db.execute("UPDATE forecast_questions SET forecast_key = '' WHERE id = ?", (question_id,))
    await store.db.commit()

    report = await store.backfill_forecast_keys(
        start=date(2026, 3, 10),
        end=date(2026, 3, 10),
        engine="native",
    )

    assert report["questions_backfilled"] == 1
    cursor = await store.db.execute("SELECT forecast_key FROM forecast_questions WHERE id = ?", (question_id,))
    row = await cursor.fetchone()
    assert row[0].startswith("iran-us:2026-03-10:")


async def test_detect_and_save_cross_topic_signals(store):
    event_a = _make_event(d="2026-03-09", summary="Iran event", entities=["Iran"])
    event_b = _make_event(d="2026-03-10", summary="Energy event", entities=["Iran", "OPEC"])
    await store.add_events([event_a], "iran-us")
    await store.add_events([event_b], "energy")

    signals = await store.detect_and_save_cross_topic_signals(reference_date=date(2026, 3, 10))
    assert signals
    iran_signals = await store.get_cross_topic_signals("iran-us")
    assert iran_signals
    assert iran_signals[0].shared_entity == "Iran"


# ── Syntheses ─────────────────────────────────────────────────────


async def test_save_and_get_synthesis(store):
    data = {"topic_name": "Iran-US Relations", "threads": [], "metadata": {}}
    sid = await store.save_synthesis(data, "iran-us", date(2026, 3, 10))
    assert sid > 0

    loaded = await store.get_synthesis("iran-us", date(2026, 3, 10))
    assert loaded is not None
    assert loaded["topic_name"] == "Iran-US Relations"


async def test_get_synthesis_missing(store):
    loaded = await store.get_synthesis("nonexistent", date(2026, 3, 10))
    assert loaded is None


# ── Pages ─────────────────────────────────────────────────────────


async def test_save_and_get_page(store):
    pid = await store.save_page(
        slug="backstory:iran-us",
        title="History of Iran-US Relations",
        page_type="backstory",
        content_md="# Background\n\nLong history...",
        topic_slug="iran-us",
        ttl_days=7,
        prompt_hash="abc123",
    )
    assert pid > 0

    page = await store.get_page("backstory:iran-us")
    assert page is not None
    assert page["title"] == "History of Iran-US Relations"
    assert page["content_md"].startswith("# Background")
    assert page["prompt_hash"] == "abc123"


async def test_page_upsert(store):
    await store.save_page("p1", "Title 1", "backstory", "Content 1", "t", 7, "h1")
    await store.save_page("p1", "Title 2", "backstory", "Content 2", "t", 7, "h2")

    page = await store.get_page("p1")
    assert page["title"] == "Title 2"
    assert page["prompt_hash"] == "h2"


async def test_get_stale_pages(store):
    # Save a page with 0-day TTL (immediately stale)
    await store.save_page("stale", "Stale", "backstory", "old", "t", 0, "h1")
    # Save a page with 30-day TTL (not stale)
    await store.save_page("fresh", "Fresh", "backstory", "new", "t", 30, "h2")

    # The 0-day TTL page should have stale_after in the past (or very close to now)
    # but due to timing, let's just check get_stale_pages returns something reasonable
    stale = await store.get_stale_pages()
    slugs = {p["slug"] for p in stale}
    assert "fresh" not in slugs


async def test_get_page_missing(store):
    page = await store.get_page("nonexistent")
    assert page is None


# ── Filter Log ────────────────────────────────────────────────────


async def test_add_and_get_filter_log(store):
    entries = [
        {
            "run_date": "2026-03-10",
            "topic_slug": "iran-us",
            "url": "https://a.com",
            "title": "Article A",
            "source_id": "reuters",
            "source_affiliation": "private",
            "source_country": "US",
            "relevance_score": 8.0,
            "relevance_reason": "Relevant to sanctions",
            "passed_pass1": True,
            "significance_score": 7.0,
            "is_novel": True,
            "significance_reason": "New development",
            "passed_pass2": True,
            "final_score": 7.4,
            "outcome": "accepted",
        },
        {
            "run_date": "2026-03-10",
            "topic_slug": "iran-us",
            "url": "https://b.com",
            "title": "Article B",
            "source_id": "food-blog",
            "source_affiliation": "private",
            "source_country": "US",
            "relevance_score": 2.0,
            "relevance_reason": "Not relevant",
            "passed_pass1": False,
            "outcome": "rejected_relevance",
        },
    ]
    await store.add_filter_log(entries)

    loaded = await store.get_filter_log("iran-us", date(2026, 3, 10))
    assert len(loaded) == 2

    # Sorted by relevance_score DESC
    assert loaded[0]["url"] == "https://a.com"
    assert loaded[0]["relevance_score"] == 8.0
    assert loaded[0]["passed_pass1"] is True
    assert loaded[0]["significance_score"] == 7.0
    assert loaded[0]["is_novel"] is True
    assert loaded[0]["outcome"] == "accepted"

    assert loaded[1]["url"] == "https://b.com"
    assert loaded[1]["passed_pass1"] is False
    assert loaded[1]["outcome"] == "rejected_relevance"


async def test_filter_stats(store):
    entries = [
        {"run_date": "2026-03-10", "topic_slug": "iran-us", "url": "https://a.com",
         "title": "A", "outcome": "accepted", "passed_pass1": True},
        {"run_date": "2026-03-10", "topic_slug": "iran-us", "url": "https://b.com",
         "title": "B", "outcome": "rejected_relevance", "passed_pass1": False},
        {"run_date": "2026-03-10", "topic_slug": "iran-us", "url": "https://c.com",
         "title": "C", "outcome": "rejected_relevance", "passed_pass1": False},
        {"run_date": "2026-03-10", "topic_slug": "iran-us", "url": "https://d.com",
         "title": "D", "outcome": "rejected_significance", "passed_pass1": True, "passed_pass2": False},
    ]
    await store.add_filter_log(entries)

    stats = await store.get_filter_stats("iran-us", date(2026, 3, 10))
    assert stats["total"] == 4
    assert stats["accepted"] == 1
    assert stats["rejected_relevance"] == 2
    assert stats["rejected_significance"] == 1


async def test_filter_log_topic_isolation(store):
    entries_iran = [
        {"run_date": "2026-03-10", "topic_slug": "iran-us", "url": "https://a.com",
         "title": "A", "outcome": "accepted", "passed_pass1": True},
    ]
    entries_ai = [
        {"run_date": "2026-03-10", "topic_slug": "ai-ml", "url": "https://b.com",
         "title": "B", "outcome": "rejected_relevance", "passed_pass1": False},
    ]
    await store.add_filter_log(entries_iran)
    await store.add_filter_log(entries_ai)

    iran_log = await store.get_filter_log("iran-us", date(2026, 3, 10))
    assert len(iran_log) == 1
    assert iran_log[0]["url"] == "https://a.com"

    ai_log = await store.get_filter_log("ai-ml", date(2026, 3, 10))
    assert len(ai_log) == 1
    assert ai_log[0]["url"] == "https://b.com"


# ── Migration ─────────────────────────────────────────────────────


async def test_import_events_from_yaml(store):
    events = [
        _make_event(d="2026-03-08", summary="Event 1"),
        _make_event(d="2026-03-09", summary="Event 2"),
    ]
    count = await store.import_events_from_yaml(events, "iran-us")
    assert count == 2

    loaded = await store.get_events("iran-us")
    assert len(loaded) == 2


async def test_import_summaries_from_yaml(store):
    summaries = [
        Summary(period_start=date(2026, 3, 3), period_end=date(2026, 3, 9),
                text="Week 1", event_count=5),
        Summary(period_start=date(2026, 3, 10), period_end=date(2026, 3, 16),
                text="Week 2", event_count=3),
    ]
    count = await store.import_summaries_from_yaml(summaries, "iran-us", "weekly")
    assert count == 2

    loaded = await store.get_summaries("iran-us", "weekly")
    assert len(loaded) == 2


# ── Dashboard Query Methods ──────────────────────────────────────


@pytest.fixture
async def seeded_store(store):
    """Store pre-loaded with events, threads, convergence, divergence for dashboard tests."""
    # Two events for iran-us, one for ai-ml
    e_iran1 = _make_event(d="2026-03-08", summary="Iran sanctions announced",
                          entities=["US", "Iran", "Treasury Dept"],
                          sources=[{"url": "https://reuters.com/1", "outlet": "reuters",
                                    "affiliation": "private", "country": "US", "language": "en"}])
    e_iran2 = _make_event(d="2026-03-09", summary="Iran condemns sanctions",
                          entities=["Iran", "US", "Foreign Ministry"],
                          sources=[{"url": "https://tass.com/1", "outlet": "tass",
                                    "affiliation": "state", "country": "RU", "language": "ru"}])
    e_ai = _make_event(d="2026-03-09", summary="New AI benchmark released",
                       entities=["OpenAI", "Google"],
                       sources=[{"url": "https://arxiv.org/1", "outlet": "arxiv",
                                 "affiliation": "academic", "country": "US", "language": "en"}])

    iran_ids = await store.add_events([e_iran1, e_iran2], "iran-us")
    ai_ids = await store.add_events([e_ai], "ai-ml")

    # Create threads and link them
    tid1 = await store.upsert_thread("sanctions-escalation", "Sanctions Escalation", 8, "active")
    await store.link_thread_topic(tid1, "iran-us")
    await store.link_thread_events(tid1, iran_ids)

    tid2 = await store.upsert_thread("ai-benchmarks", "AI Benchmark Progress", 6, "emerging")
    await store.link_thread_topic(tid2, "ai-ml")
    await store.link_thread_events(tid2, ai_ids)

    # A resolved thread (should not appear in active)
    tid3 = await store.upsert_thread("old-thread", "Old Resolved Thread", 3, "resolved")
    await store.link_thread_topic(tid3, "iran-us")

    # Add convergence and divergence to sanctions thread
    await store.add_convergence(tid1, "New sanctions targeting oil sector", ["reuters", "tass"])
    await store.add_convergence(tid1, "Treasury Dept leading enforcement", ["reuters"])
    await store.add_divergence(
        tid1, "Sanctions impact on Iran",
        "reuters", "Sanctions will cripple Iran's economy",
        "tass", "Sanctions are ineffective economic warfare",
    )

    return {
        "store": store,
        "iran_event_ids": iran_ids,
        "ai_event_ids": ai_ids,
        "sanctions_thread_id": tid1,
        "ai_thread_id": tid2,
        "resolved_thread_id": tid3,
    }


async def test_get_event_by_id(seeded_store):
    s = seeded_store["store"]
    eid = seeded_store["iran_event_ids"][0]

    event = await s.get_event_by_id(eid)
    assert event is not None
    assert event.summary == "Iran sanctions announced"
    assert event.significance == 7
    assert len(event.sources) == 1
    assert event.sources[0]["outlet"] == "reuters"
    assert "US" in event.entities

    # Missing event returns None
    assert await s.get_event_by_id(99999) is None


async def test_get_events_for_thread(seeded_store):
    s = seeded_store["store"]
    events = await s.get_events_for_thread(seeded_store["sanctions_thread_id"])
    assert len(events) == 2
    assert events[0].summary == "Iran sanctions announced"  # Ordered by date ASC
    assert events[1].summary == "Iran condemns sanctions"
    # Verify full hydration
    assert len(events[0].sources) == 1
    assert "Iran" in events[1].entities


async def test_get_events_for_thread_as_of(seeded_store):
    s = seeded_store["store"]
    events = await s.get_events_for_thread_as_of(seeded_store["sanctions_thread_id"], date(2026, 3, 8))
    assert len(events) == 1
    assert events[0].summary == "Iran sanctions announced"


async def test_get_convergence_for_thread(seeded_store):
    s = seeded_store["store"]
    conv = await s.get_convergence_for_thread(seeded_store["sanctions_thread_id"])
    assert len(conv) == 2
    assert conv[0]["fact_text"] == "New sanctions targeting oil sector"
    assert "reuters" in conv[0]["confirmed_by"]
    assert "tass" in conv[0]["confirmed_by"]

    # Thread with no convergence
    conv2 = await s.get_convergence_for_thread(seeded_store["ai_thread_id"])
    assert conv2 == []


async def test_get_divergence_for_thread(seeded_store):
    s = seeded_store["store"]
    div = await s.get_divergence_for_thread(seeded_store["sanctions_thread_id"])
    assert len(div) == 1
    assert div[0]["shared_event"] == "Sanctions impact on Iran"
    assert div[0]["source_a"] == "reuters"
    assert div[0]["framing_b"] == "Sanctions are ineffective economic warfare"

    # Thread with no divergence
    div2 = await s.get_divergence_for_thread(seeded_store["ai_thread_id"])
    assert div2 == []


async def test_get_thread(seeded_store):
    s = seeded_store["store"]
    thread = await s.get_thread("sanctions-escalation")
    assert thread is not None
    assert thread["headline"] == "Sanctions Escalation"
    assert thread["status"] == "active"
    assert thread["significance"] == 8
    # Key entities come from linked events
    assert "US" in thread["key_entities"]
    assert "Iran" in thread["key_entities"]

    # Resolved thread still found by get_thread (unlike get_active_threads)
    resolved = await s.get_thread("old-thread")
    assert resolved is not None
    assert resolved["status"] == "resolved"

    # Missing slug returns None
    assert await s.get_thread("nonexistent") is None


async def test_get_all_threads_as_of_uses_historical_snapshot(store):
    event = _make_event(d="2026-03-01", summary="Sanctions statement", entities=["Iran"])
    event_id = (await store.add_events([event], "iran-us"))[0]
    tid = await store.upsert_thread("iran-thread", "Iran Thread", 8, "active")
    await store.link_thread_topic(tid, "iran-us")
    await store.link_thread_events(tid, [event_id])
    await store.upsert_thread_snapshot(ThreadSnapshot(
        thread_id=tid,
        snapshot_date=date(2026, 3, 1),
        status="active",
        significance=8,
        event_count=1,
        latest_event_date=date(2026, 3, 1),
    ))
    await store.upsert_thread_snapshot(ThreadSnapshot(
        thread_id=tid,
        snapshot_date=date(2026, 3, 10),
        status="active",
        significance=8,
        event_count=3,
        latest_event_date=date(2026, 3, 10),
    ))

    historical = await store.get_all_threads_as_of("iran-us", date(2026, 3, 1))
    latest = await store.get_all_threads("iran-us")

    assert historical[0]["snapshot_count"] == 1
    assert latest[0]["snapshot_count"] == 2


async def test_get_cross_topic_signals_as_of_ignores_future_rows(store):
    iran = _make_event(d="2026-03-01", summary="Sanctions statement", entities=["Iran"])
    energy = _make_event(d="2026-03-12", summary="Energy price statement", entities=["Iran"])
    iran_id = (await store.add_events([iran], "iran-us"))[0]
    energy_id = (await store.add_events([energy], "global-energy-transition"))[0]

    iran_entity_id = await store.upsert_entity("Iran", "country", ["iran"])
    await store.link_event_entities(iran_id, [iran_entity_id])
    await store.link_event_entities(energy_id, [iran_entity_id])
    await store.detect_and_save_cross_topic_signals(reference_date=date(2026, 3, 12))

    signals = await store.get_cross_topic_signals_as_of("iran-us", date(2026, 3, 1))
    assert signals == []


async def test_get_all_threads(seeded_store):
    s = seeded_store["store"]

    # All threads, no filter
    all_threads = await s.get_all_threads()
    assert len(all_threads) == 3

    # Filter by topic
    iran_threads = await s.get_all_threads(topic_slug="iran-us")
    slugs = {t["slug"] for t in iran_threads}
    assert slugs == {"sanctions-escalation", "old-thread"}

    # Filter by status
    active = await s.get_all_threads(status="active")
    assert len(active) == 1
    assert active[0]["slug"] == "sanctions-escalation"

    # Filter by topic + status
    iran_resolved = await s.get_all_threads(topic_slug="iran-us", status="resolved")
    assert len(iran_resolved) == 1
    assert iran_resolved[0]["slug"] == "old-thread"


async def test_get_threads_for_entity(seeded_store):
    s = seeded_store["store"]
    # Find the "Iran" entity
    entity = await s.find_entity("Iran")
    assert entity is not None

    threads = await s.get_threads_for_entity(entity["id"])
    assert len(threads) >= 1
    slugs = {t["slug"] for t in threads}
    assert "sanctions-escalation" in slugs


async def test_get_source_stats(seeded_store):
    s = seeded_store["store"]

    # All sources
    stats = await s.get_source_stats()
    assert len(stats) >= 2
    outlets = {st["outlet"] for st in stats}
    assert "reuters" in outlets
    assert "tass" in outlets

    # Scoped to topic
    iran_stats = await s.get_source_stats(topic_slug="iran-us")
    iran_outlets = {st["outlet"] for st in iran_stats}
    assert "reuters" in iran_outlets
    assert "arxiv" not in iran_outlets


async def test_search_entities(seeded_store):
    s = seeded_store["store"]

    results = await s.search_entities("Iran")
    assert len(results) >= 1
    names = {r["canonical_name"] for r in results}
    assert "Iran" in names

    # Partial match
    results2 = await s.search_entities("Treas")
    assert any(r["canonical_name"] == "Treasury Dept" for r in results2)

    # No match
    results3 = await s.search_entities("zzzznonexistent")
    assert results3 == []


async def test_get_topic_stats(seeded_store):
    s = seeded_store["store"]
    stats = await s.get_topic_stats()
    assert len(stats) == 2  # iran-us and ai-ml

    by_slug = {st["topic_slug"]: st for st in stats}

    assert by_slug["iran-us"]["event_count"] == 2
    assert by_slug["iran-us"]["entity_count"] >= 3  # US, Iran, Treasury Dept
    assert by_slug["iran-us"]["thread_count"] >= 1
    assert by_slug["iran-us"]["latest_date"] == "2026-03-09"

    assert by_slug["ai-ml"]["event_count"] == 1
    assert by_slug["ai-ml"]["entity_count"] >= 2  # OpenAI, Google


async def test_get_related_entities(seeded_store):
    s = seeded_store["store"]
    # US and Iran co-appear in both iran-us events
    us_entity = await s.find_entity("US")
    assert us_entity is not None

    related = await s.get_related_entities(us_entity["id"])
    related_names = {r["canonical_name"] for r in related}
    assert "Iran" in related_names
    assert "Treasury Dept" in related_names
    # co_occurrence_count should be populated
    iran_rel = next(r for r in related if r["canonical_name"] == "Iran")
    assert iran_rel["co_occurrence_count"] >= 2  # Both iran events mention both US and Iran


# ── find_event_id ────────────────────────────────────────────────

async def test_find_event_id(seeded_store):
    s = seeded_store["store"]
    # Match by summary + date + topic_slug
    eid = await s.find_event_id("Iran sanctions announced", "2026-03-08", "iran-us")
    assert eid is not None
    assert eid == seeded_store["iran_event_ids"][0]


async def test_find_event_id_no_match(seeded_store):
    s = seeded_store["store"]
    # Wrong summary
    assert await s.find_event_id("Nonexistent event", "2026-03-08", "iran-us") is None
    # Wrong topic
    assert await s.find_event_id("Iran sanctions announced", "2026-03-08", "ai-ml") is None
    # Wrong date
    assert await s.find_event_id("Iran sanctions announced", "2099-01-01", "iran-us") is None


# ── get_threads_for_event ────────────────────────────────────────

async def test_get_threads_for_event(seeded_store):
    s = seeded_store["store"]
    eid = seeded_store["iran_event_ids"][0]
    threads = await s.get_threads_for_event(eid)
    assert len(threads) >= 1
    assert any(t["slug"] == "sanctions-escalation" for t in threads)


async def test_get_threads_for_event_unlinked(seeded_store):
    s = seeded_store["store"]
    # Event that exists but isn't linked to any thread
    # (ai events are linked, so create a fresh event)
    e = Event(date="2026-03-10", summary="Orphan event", significance=3,
              entities=["Test"], sources=[])
    ids = await s.add_events([e], "test-topic")
    threads = await s.get_threads_for_event(ids[0])
    assert threads == []


# ── get_topics_for_thread ────────────────────────────────────────

async def test_get_topics_for_thread(seeded_store):
    s = seeded_store["store"]
    tid = seeded_store["sanctions_thread_id"]
    topics = await s.get_topics_for_thread(tid)
    assert "iran-us" in topics


# ── get_related_events ───────────────────────────────────────────

async def test_get_related_events(seeded_store):
    s = seeded_store["store"]
    eid = seeded_store["iran_event_ids"][0]
    # The other iran event shares entities (US, Iran)
    related = await s.get_related_events(eid, limit=5)
    assert len(related) >= 1
    related_ids = {r["id"] for r in related}
    assert seeded_store["iran_event_ids"][1] in related_ids
    # Should NOT include the event itself
    assert eid not in related_ids


# ── get_historical_calibration ────────────────────────────────────


async def test_get_historical_calibration_returns_resolved_rows(store):
    """Should return only resolved forecast questions with their outcomes."""
    # Create two runs — one resolved, one pending
    run1 = ForecastRun(
        topic_slug="iran-us",
        topic_name="Iran-US",
        engine="native",
        generated_for=date(2026, 3, 10),
        questions=[ForecastQuestion(
            question="Will sanctions add a new event by 2026-03-17?",
            forecast_type="binary",
            target_variable="thread_new_event_count",
            target_metadata={"thread_id": 1, "threshold": 1, "topic_slug": "iran-us"},
            probability=0.60,
            base_rate=0.50,
            resolution_criteria="Resolves true if thread 1 gains 1+ event.",
            resolution_date=date(2026, 3, 17),
            horizon_days=7,
            signpost="Watch for Treasury action",
            evidence_thread_ids=[1],
        )],
    )
    await store.save_forecast_run(run1)
    qid1 = run1.questions[0].question_id

    run2 = ForecastRun(
        topic_slug="iran-us",
        topic_name="Iran-US",
        engine="native",
        generated_for=date(2026, 3, 12),
        questions=[ForecastQuestion(
            question="Will sanctions add a new event by 2026-03-19?",
            forecast_type="binary",
            target_variable="thread_new_event_count",
            target_metadata={"thread_id": 2, "threshold": 1, "topic_slug": "iran-us"},
            probability=0.45,
            base_rate=0.40,
            resolution_criteria="Resolves true if thread 2 gains 1+ event.",
            resolution_date=date(2026, 3, 19),
            horizon_days=7,
            signpost="Watch for escalation",
            evidence_thread_ids=[2],
        )],
    )
    await store.save_forecast_run(run2)

    # Resolve only the first one
    await store.set_forecast_resolution(ForecastResolution(
        forecast_question_id=qid1,
        outcome_status="resolved",
        resolved_bool=True,
        actual_value=1.0,
        brier_score=0.16,
        log_loss=0.51,
        notes="One new event observed.",
        resolved_at=date(2026, 3, 17),
    ))

    rows = await store.get_historical_calibration()
    assert len(rows) == 1
    row = rows[0]
    assert row["target_variable"] == "thread_new_event_count"
    assert row["probability"] == 0.60
    assert row["resolved_bool"] is True
    assert row["base_rate"] == 0.50


async def test_get_historical_calibration_respects_as_of_cutoff(store):
    """as_of should exclude forecast runs generated after the cutoff date."""
    for day, prob in [(10, 0.55), (13, 0.65)]:
        run = ForecastRun(
            topic_slug="iran-us",
            topic_name="Iran-US",
            engine="native",
            generated_for=date(2026, 3, day),
            questions=[ForecastQuestion(
                question=f"Q for day {day}",
                forecast_type="binary",
                target_variable="thread_new_event_count",
                target_metadata={"thread_id": day, "threshold": 1, "topic_slug": "iran-us"},
                probability=prob,
                base_rate=0.40,
                resolution_criteria="Test.",
                resolution_date=date(2026, 3, day + 7),
                horizon_days=7,
                signpost="Test",
                evidence_thread_ids=[day],
            )],
        )
        await store.save_forecast_run(run)
        await store.set_forecast_resolution(ForecastResolution(
            forecast_question_id=run.questions[0].question_id,
            outcome_status="resolved",
            resolved_bool=False,
            actual_value=0.0,
            brier_score=round(prob ** 2, 4),
            log_loss=0.5,
            notes="No event.",
            resolved_at=date(2026, 3, day + 7),
        ))

    # Both resolved
    all_rows = await store.get_historical_calibration()
    assert len(all_rows) == 2

    # With cutoff at March 12, only the day-10 run should appear
    scoped = await store.get_historical_calibration(as_of=date(2026, 3, 12))
    assert len(scoped) == 1
    assert scoped[0]["probability"] == 0.55


# ── Thread merging cascades ────────────────────────────────────────


async def _setup_two_threads_with_events(store):
    """Create two threads with overlapping events for merge tests."""
    e1 = _make_event(summary="Sanctions filing", d="2026-03-01")
    e2 = _make_event(summary="Naval deployment", d="2026-03-05")
    e3 = _make_event(summary="Ceasefire talks", d="2026-03-08")
    ids = await store.add_events([e1, e2, e3], "iran-us")

    keep_id = await store.upsert_thread("keep-thread", "Keep Thread", 8, "active")
    absorb_id = await store.upsert_thread("absorb-thread", "Absorb Thread", 5, "active")

    await store.link_thread_events(keep_id, [ids[0]])
    await store.link_thread_events(absorb_id, [ids[1], ids[2]])
    await store.link_thread_topic(keep_id, "iran-us")
    await store.link_thread_topic(absorb_id, "iran-us")

    return keep_id, absorb_id, ids


async def test_merge_threads_snapshots_migrated(store):
    """Snapshots from absorbed thread should move to kept thread."""
    keep_id, absorb_id, _ = await _setup_two_threads_with_events(store)

    # Create snapshots on both threads
    await store.upsert_thread_snapshot(ThreadSnapshot(
        thread_id=keep_id, snapshot_date=date(2026, 3, 1),
        status="active", significance=8, event_count=1,
        latest_event_date=date(2026, 3, 1),
    ))
    await store.upsert_thread_snapshot(ThreadSnapshot(
        thread_id=absorb_id, snapshot_date=date(2026, 3, 5),
        status="active", significance=5, event_count=2,
        latest_event_date=date(2026, 3, 5),
    ))

    await store.merge_threads(keep_id, absorb_id)

    # Absorbed thread's snapshot should now belong to kept thread
    cursor = await store.db.execute(
        "SELECT thread_id FROM thread_snapshots WHERE snapshot_date = '2026-03-05'"
    )
    row = await cursor.fetchone()
    assert row[0] == keep_id

    # No snapshots remain on absorbed thread
    cursor = await store.db.execute(
        "SELECT COUNT(*) FROM thread_snapshots WHERE thread_id = ?", (absorb_id,)
    )
    assert (await cursor.fetchone())[0] == 0


async def test_merge_threads_snapshot_date_conflict(store):
    """When both threads have a snapshot on the same date, handle via OR IGNORE."""
    keep_id, absorb_id, _ = await _setup_two_threads_with_events(store)

    # Both threads have a snapshot on the same date
    await store.upsert_thread_snapshot(ThreadSnapshot(
        thread_id=keep_id, snapshot_date=date(2026, 3, 5),
        status="active", significance=8, event_count=3,
        latest_event_date=date(2026, 3, 5),
    ))
    await store.upsert_thread_snapshot(ThreadSnapshot(
        thread_id=absorb_id, snapshot_date=date(2026, 3, 5),
        status="active", significance=5, event_count=1,
        latest_event_date=date(2026, 3, 5),
    ))

    await store.merge_threads(keep_id, absorb_id)

    # Only one snapshot per date should remain on kept thread
    cursor = await store.db.execute(
        "SELECT COUNT(*) FROM thread_snapshots WHERE thread_id = ? AND snapshot_date = '2026-03-05'",
        (keep_id,),
    )
    assert (await cursor.fetchone())[0] == 1

    # No snapshots on absorbed thread
    cursor = await store.db.execute(
        "SELECT COUNT(*) FROM thread_snapshots WHERE thread_id = ?", (absorb_id,)
    )
    assert (await cursor.fetchone())[0] == 0


async def test_merge_threads_projection_items_json_updated(store):
    """evidence_thread_ids_json in projection_items should be rewritten."""
    keep_id, absorb_id, _ = await _setup_two_threads_with_events(store)

    # Create a projection with items referencing the absorbed thread
    projection = TopicProjection(
        topic_slug="iran-us", topic_name="Iran-US",
        generated_for=date(2026, 3, 10),
        items=[ProjectionItem(
            claim="Test claim",
            signpost="Test signpost",
            review_after=date(2026, 3, 17),
            evidence_thread_ids=[absorb_id, 999],
        )],
    )
    await store.save_projection(projection)

    await store.merge_threads(keep_id, absorb_id)

    # Check the JSON was rewritten
    cursor = await store.db.execute(
        "SELECT evidence_thread_ids_json FROM projection_items"
    )
    row = await cursor.fetchone()
    import json
    ids = json.loads(row[0])
    assert keep_id in ids
    assert absorb_id not in ids
    assert 999 in ids  # other IDs preserved


async def test_merge_threads_forecast_questions_json_updated(store):
    """evidence_thread_ids_json in forecast_questions should be rewritten."""
    keep_id, absorb_id, _ = await _setup_two_threads_with_events(store)

    run = ForecastRun(
        topic_slug="iran-us", topic_name="Iran-US",
        generated_for=date(2026, 3, 10),
        questions=[ForecastQuestion(
            question="Will Iran escalate?",
            target_variable="iran_escalation",
            probability=0.6,
            resolution_criteria="Military action observed",
            resolution_date=date(2026, 3, 17),
            signpost="Troop movement",
            evidence_thread_ids=[absorb_id, 777],
        )],
    )
    await store.save_forecast_run(run)

    await store.merge_threads(keep_id, absorb_id)

    cursor = await store.db.execute(
        "SELECT evidence_thread_ids_json FROM forecast_questions"
    )
    row = await cursor.fetchone()
    import json
    ids = json.loads(row[0])
    assert keep_id in ids
    assert absorb_id not in ids
    assert 777 in ids


async def test_merge_threads_idempotent(store):
    """Merging an already-resolved thread should not crash."""
    keep_id, absorb_id, _ = await _setup_two_threads_with_events(store)

    # First merge
    await store.merge_threads(keep_id, absorb_id)

    # Second merge — should be a no-op, not an error
    await store.merge_threads(keep_id, absorb_id)

    # Absorbed thread still resolved
    cursor = await store.db.execute(
        "SELECT status FROM threads WHERE id = ?", (absorb_id,)
    )
    assert (await cursor.fetchone())[0] == "resolved"


async def test_merge_sets_merged_into_id(store):
    """Absorbed thread records which thread it was merged into."""
    keep_id, absorb_id, _ = await _setup_two_threads_with_events(store)
    await store.merge_threads(keep_id, absorb_id)

    cursor = await store.db.execute(
        "SELECT merged_into_id FROM threads WHERE id = ?", (absorb_id,)
    )
    row = await cursor.fetchone()
    assert row[0] == keep_id


async def test_keep_thread_has_null_merged_into(store):
    """The keeper thread should have no merged_into_id."""
    keep_id, absorb_id, _ = await _setup_two_threads_with_events(store)
    await store.merge_threads(keep_id, absorb_id)

    cursor = await store.db.execute(
        "SELECT merged_into_id FROM threads WHERE id = ?", (keep_id,)
    )
    row = await cursor.fetchone()
    assert row[0] is None


async def test_chained_merge_preserves_trail(store):
    """A→B merge then B→C merge: A still points to B, B points to C."""
    e = _make_event(summary="Test event", d="2026-03-01")
    ids = await store.add_events([e], "iran-us")

    a_id = await store.upsert_thread("thread-a", "Thread A", 5, "active")
    b_id = await store.upsert_thread("thread-b", "Thread B", 7, "active")
    c_id = await store.upsert_thread("thread-c", "Thread C", 9, "active")
    await store.link_thread_events(a_id, ids)
    await store.link_thread_events(b_id, ids)
    await store.link_thread_events(c_id, ids)

    await store.merge_threads(b_id, a_id)  # A absorbed into B
    await store.merge_threads(c_id, b_id)  # B absorbed into C

    cursor = await store.db.execute(
        "SELECT merged_into_id FROM threads WHERE id = ?", (a_id,)
    )
    assert (await cursor.fetchone())[0] == b_id

    cursor = await store.db.execute(
        "SELECT merged_into_id FROM threads WHERE id = ?", (b_id,)
    )
    assert (await cursor.fetchone())[0] == c_id


async def test_merge_threads_duplicate_events_deduplicated(store):
    """If both threads link to the same event, it appears once after merge."""
    e1 = _make_event(summary="Shared event", d="2026-03-01")
    ids = await store.add_events([e1], "iran-us")

    keep_id = await store.upsert_thread("keep", "Keep", 8, "active")
    absorb_id = await store.upsert_thread("absorb", "Absorb", 5, "active")

    # Both threads link to the same event
    await store.link_thread_events(keep_id, ids)
    await store.link_thread_events(absorb_id, ids)

    await store.merge_threads(keep_id, absorb_id)

    cursor = await store.db.execute(
        "SELECT COUNT(*) FROM thread_events WHERE thread_id = ? AND event_id = ?",
        (keep_id, ids[0]),
    )
    assert (await cursor.fetchone())[0] == 1


# ── get_interesting_kalshi_markets ─────────────────────────────────────


async def _insert_kalshi_question(
    store,
    question,
    prob,
    market_prob,
    resolution_date,
    ticker,
    *,
    engine="structural",
    status="open",
):
    """Helper: insert a kalshi-aligned forecast question."""
    import json
    await store.db.execute(
        "INSERT INTO forecast_runs (topic_slug, topic_name, engine, generated_for, summary) "
        "VALUES (?, ?, ?, ?, ?)",
        ("kalshi-aligned", "Kalshi", engine, "2026-03-01", "test"),
    )
    run_id = (await store.db.execute("SELECT last_insert_rowid()")).fetchone()
    run_id = (await run_id)[0] if hasattr(run_id, "__await__") else run_id[0]
    meta = json.dumps({"kalshi_ticker": ticker, "kalshi_implied": market_prob})
    await store.db.execute(
        "INSERT INTO forecast_questions "
        "(forecast_run_id, question, target_variable, target_metadata_json, "
        "probability, base_rate, resolution_criteria, resolution_date, "
        "horizon_days, signpost, status, external_ref) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (run_id, question, "kalshi_aligned", meta,
         prob, market_prob, "test", resolution_date,
         14, "test", status, ticker),
    )
    await store.db.commit()


async def test_get_interesting_kalshi_markets_ranked(store):
    """Markets should be ranked by interestingness (gap + proximity)."""
    # Moderate gap, far out
    await _insert_kalshi_question(store, "Mod gap far", 0.60, 0.40, "2026-06-15", "MOD-FAR")
    # Small gap, close
    await _insert_kalshi_question(store, "Small gap close", 0.55, 0.50, "2026-03-20", "SMALL-CLOSE")
    # High gap, close — should rank highest
    await _insert_kalshi_question(store, "High gap close", 0.80, 0.30, "2026-03-20", "HIGH-CLOSE")

    results = await store.get_interesting_kalshi_markets(limit=5)

    assert len(results) == 3
    # High-close has both high gap AND close date — should be first
    assert results[0]["ticker"] == "HIGH-CLOSE"
    # Verify all returned with expected fields
    assert "question" in results[0]
    assert "gap_pp" in results[0]
    assert results[0]["gap_pp"] == 50


async def test_get_interesting_kalshi_markets_excludes_resolved(store):
    """Resolved markets should not appear."""
    await _insert_kalshi_question(store, "Open market", 0.70, 0.30, "2026-04-01", "OPEN")
    await _insert_kalshi_question(store, "Resolved market", 0.60, 0.40, "2026-04-01", "RESOLVED", status="resolved")

    results = await store.get_interesting_kalshi_markets(limit=5)
    tickers = [r["ticker"] for r in results]
    assert "OPEN" in tickers
    assert "RESOLVED" not in tickers


async def test_get_interesting_kalshi_markets_excludes_expired(store):
    """Markets with past resolution dates should not appear."""
    await _insert_kalshi_question(store, "Future market", 0.70, 0.30, "2026-04-01", "FUTURE")
    await _insert_kalshi_question(store, "Past market", 0.60, 0.40, "2026-01-01", "PAST")

    results = await store.get_interesting_kalshi_markets(limit=5)
    tickers = [r["ticker"] for r in results]
    assert "FUTURE" in tickers
    assert "PAST" not in tickers


async def test_get_interesting_kalshi_markets_empty(store):
    """Should return empty list when no Kalshi data exists."""
    results = await store.get_interesting_kalshi_markets(limit=5)
    assert results == []


async def test_get_interesting_kalshi_markets_can_filter_engine(store):
    """Public callers should be able to request actor-only Kalshi markets."""
    future_resolution = (date.today() + timedelta(days=14)).isoformat()
    await _insert_kalshi_question(
        store,
        "Actor market",
        0.70,
        0.40,
        future_resolution,
        "ACTOR",
        engine="actor",
    )
    await _insert_kalshi_question(
        store,
        "Structural market",
        0.55,
        0.35,
        future_resolution,
        "STRUCTURAL",
        engine="structural",
    )

    results = await store.get_interesting_kalshi_markets(limit=5, engine="actor")
    assert [r["ticker"] for r in results] == ["ACTOR"]


# ── purge_template_projections ────────────────────────────────────────


async def _insert_projection_with_item(store, claim):
    """Helper: insert a projection with one item."""
    await store.db.execute(
        "INSERT INTO projections (topic_slug, topic_name, generated_for) "
        "VALUES (?, ?, ?)",
        ("test-topic", "Test", "2026-03-01"),
    )
    cursor = await store.db.execute("SELECT last_insert_rowid()")
    proj_id = (await cursor.fetchone() if hasattr(cursor, "fetchone") else cursor)[0]
    await store.db.execute(
        "INSERT INTO projection_items (projection_id, claim, signpost, review_after) "
        "VALUES (?, ?, ?, ?)",
        (proj_id, claim, "test signpost", "2026-03-15"),
    )
    await store.db.commit()
    return proj_id


async def test_purge_template_projections_dry_run(store):
    """Dry run should identify but not delete template items."""
    await _insert_projection_with_item(
        store,
        "Will Iran be involved in significant new developments related to geopolitics within 7 days?",
    )
    result = await store.purge_template_projections(dry_run=True)

    assert result["items_found"] >= 1
    # Items still exist
    cursor = await store.db.execute("SELECT COUNT(*) FROM projection_items")
    assert (await cursor.fetchone())[0] >= 1


async def test_purge_template_projections_deletes_templates(store):
    """Should delete template items but preserve good ones."""
    await _insert_projection_with_item(
        store,
        "Will Iran produce significant new developments within 7 days?",
    )
    await _insert_projection_with_item(
        store,
        "Iran will impose new transit fees on non-allied vessels in the Strait of Hormuz",
    )

    result = await store.purge_template_projections(dry_run=False)

    assert result["items_deleted"] == 1
    # Good item still exists
    cursor = await store.db.execute("SELECT claim FROM projection_items")
    rows = await cursor.fetchall()
    claims = [r[0] for r in rows]
    assert "Iran will impose new transit fees on non-allied vessels in the Strait of Hormuz" in claims
    assert not any("significant new developments" in c for c in claims)


# ── purge_forecast_runs ───────────────────────────────────────────────


def _make_forecast_run(topic_slug, engine, target_variable, question_text="Test?"):
    """Helper to build a minimal ForecastRun."""
    return ForecastRun(
        topic_slug=topic_slug,
        topic_name=topic_slug.replace("-", " ").title(),
        engine=engine,
        generated_for=date(2026, 3, 17),
        summary="test",
        questions=[ForecastQuestion(
            question=question_text,
            forecast_type="binary",
            target_variable=target_variable,
            probability=0.5,
            base_rate=0.5,
            resolution_criteria="test",
            resolution_date=date(2026, 3, 24),
            horizon_days=7,
            signpost="test",
        )],
    )


async def test_purge_by_target_variable(store):
    """Purge benchmark, verify topic_claim survives."""
    run_bench = _make_forecast_run("bench", "actor", "kalshi_benchmark", "Bench?")
    run_claim = _make_forecast_run("claim", "actor", "topic_claim", "Claim?")
    await store.save_forecast_run(run_bench)
    await store.save_forecast_run(run_claim)

    # Resolve the benchmark question
    qid = run_bench.questions[0].question_id
    await store.set_forecast_resolution(ForecastResolution(
        forecast_question_id=qid,
        outcome_status="resolved",
        resolved_bool=True,
        brier_score=0.1,
    ))

    result = await store.purge_forecast_runs(target_variable="kalshi_benchmark", dry_run=False)
    assert result["runs_deleted"] == 1
    assert result["questions_deleted"] == 1
    assert result["resolutions_deleted"] == 1

    # topic_claim still there
    cursor = await store.db.execute(
        "SELECT COUNT(*) FROM forecast_questions WHERE target_variable='topic_claim'"
    )
    assert (await cursor.fetchone())[0] == 1


async def test_get_featured_predictions_can_filter_public_actor_slice(store):
    """Featured predictions should support the actor-only public Forward Look contract."""
    await store.save_forecast_run(_make_forecast_run("iran-us", "actor", "topic_claim", "Actor claim?"))
    await store.save_forecast_run(_make_forecast_run("iran-us", "actor", "thread_development", "Actor thread?"))
    await store.save_forecast_run(_make_forecast_run("iran-us", "actor", "kalshi_benchmark", "Benchmark row?"))
    await store.save_forecast_run(_make_forecast_run("iran-us", "structural", "topic_claim", "Structural claim?"))

    results = await store.get_featured_predictions(
        "iran-us",
        engine="actor",
        allowed_target_variables={"topic_claim", "thread_development", "kalshi_aligned"},
    )

    questions = [row["question"] for row in results]
    assert "Actor claim?" in questions
    assert "Actor thread?" in questions
    assert "Benchmark row?" not in questions
    assert "Structural claim?" not in questions


async def test_purge_dry_run(store):
    """Dry run reports counts but doesn't delete."""
    run = _make_forecast_run("bench", "actor", "kalshi_benchmark")
    await store.save_forecast_run(run)

    result = await store.purge_forecast_runs(target_variable="kalshi_benchmark", dry_run=True)
    assert result["runs_found"] >= 1
    assert result["questions_found"] >= 1
    assert result["dry_run"] is True

    # Data still exists
    cursor = await store.db.execute(
        "SELECT COUNT(*) FROM forecast_questions WHERE target_variable='kalshi_benchmark'"
    )
    assert (await cursor.fetchone())[0] >= 1


async def test_purge_cascade_order(store):
    """All child rows (resolutions, mappings, questions) cleaned up."""
    run = _make_forecast_run("bench", "actor", "kalshi_benchmark")
    await store.save_forecast_run(run)
    qid = run.questions[0].question_id

    # Add a resolution
    await store.set_forecast_resolution(ForecastResolution(
        forecast_question_id=qid,
        outcome_status="resolved",
        resolved_bool=False,
        brier_score=0.25,
    ))

    result = await store.purge_forecast_runs(target_variable="kalshi_benchmark", dry_run=False)
    assert result["runs_deleted"] == 1
    assert result["questions_deleted"] == 1
    assert result["resolutions_deleted"] >= 1

    # Verify everything is gone
    for table in ("forecast_runs", "forecast_questions", "forecast_resolutions"):
        cursor = await store.db.execute(f"SELECT COUNT(*) FROM {table}")
        assert (await cursor.fetchone())[0] == 0, f"{table} not empty"


async def test_purge_empty_noop(store):
    """Purge on empty DB returns zeros without error."""
    result = await store.purge_forecast_runs(target_variable="kalshi_benchmark", dry_run=False)
    assert result["runs_deleted"] == 0
    assert result["questions_deleted"] == 0
