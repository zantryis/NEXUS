"""Tests for projection runtime services."""

from datetime import date
import json

import pytest

from nexus.config.models import FutureProjectionConfig, NexusConfig, TopicConfig, UserConfig
from nexus.engine.knowledge.events import Event
from nexus.engine.knowledge.store import KnowledgeStore
from nexus.engine.projection.service import (
    backfill_signal_rich_profile,
    backfill_syntheses,
    backfill_thread_snapshots,
    generate_projections_from_store,
    projection_eligibility,
    render_projection_markdown,
    run_projection_pass,
)
from nexus.engine.projection.models import ProjectionItem, TopicProjection
from nexus.engine.projection.models import ThreadSnapshot
from nexus.engine.synthesis.knowledge import NarrativeThread, TopicSynthesis


@pytest.fixture
async def store(tmp_path):
    s = KnowledgeStore(tmp_path / "projection.db")
    await s.initialize()
    yield s
    await s.close()


async def test_backfill_thread_snapshots(store):
    event_a = Event(date=date(2026, 3, 8), summary="A", entities=["Iran"])
    event_b = Event(date=date(2026, 3, 10), summary="B", entities=["Iran"])
    event_ids = await store.add_events([event_a, event_b], "iran-us")
    thread_id = await store.upsert_thread("iran-thread", "Iran Thread", 8, "active")
    await store.link_thread_topic(thread_id, "iran-us")
    await store.link_thread_events(thread_id, event_ids)
    await store.save_synthesis({"topic_name": "Iran-US", "threads": []}, "iran-us", date(2026, 3, 10))

    result = await backfill_thread_snapshots(store)
    assert result["snapshots"] >= 2

    snapshots = await store.get_thread_snapshots(thread_id)
    assert len(snapshots) >= 2


async def test_projection_eligibility(store):
    thread_id = await store.upsert_thread("eligible", "Eligible", 8, "active")
    await store.upsert_thread_snapshot(
        ThreadSnapshot(
            thread_id=thread_id,
            snapshot_date=date(2026, 3, 1),
            status="active",
            significance=8,
            event_count=1,
            latest_event_date=date(2026, 3, 1),
        )
    )
    await store.upsert_thread_snapshot(
        ThreadSnapshot(
            thread_id=thread_id,
            snapshot_date=date(2026, 3, 8),
            status="active",
            significance=8,
            event_count=3,
            latest_event_date=date(2026, 3, 8),
        )
    )
    await store.upsert_thread_snapshot(
        ThreadSnapshot(
            thread_id=thread_id,
            snapshot_date=date(2026, 3, 10),
            status="active",
            significance=8,
            event_count=4,
            latest_event_date=date(2026, 3, 10),
        )
    )
    await store.add_events([Event(date=date(2026, 3, 1), summary="A", entities=["Iran"])], "iran-us")
    await store.add_events([Event(date=date(2026, 3, 10), summary="B", entities=["Iran"])], "iran-us")

    thread = {"snapshot_count": 3}
    eligible, meta = await projection_eligibility(store, "iran-us", [thread], FutureProjectionConfig(enabled=True))
    assert eligible is True
    assert meta["history_days"] >= 7


async def test_run_projection_pass_persists_projection_page(store, tmp_path):
    event = Event(date=date(2026, 3, 10), summary="A", entities=["Iran"])
    event_ids = await store.add_events([event], "iran-us")
    thread_id = await store.upsert_thread("iran-thread", "Iran Thread", 8, "active")
    await store.link_thread_topic(thread_id, "iran-us")
    await store.link_thread_events(thread_id, event_ids)

    synthesis = TopicSynthesis(
        topic_name="Iran-US",
        threads=[NarrativeThread(
            headline="Iran Thread",
            events=[Event(event_id=event_ids[0], date=date(2026, 3, 10), summary="A", entities=["Iran"])],
            significance=8,
            thread_id=thread_id,
            slug="iran-thread",
            status="active",
            snapshot_count=3,
        )],
    )
    config = FutureProjectionConfig(enabled=True, min_history_days=1, min_thread_snapshots=1)
    await run_projection_pass(store, None, [synthesis], run_date=date(2026, 3, 10), config=config, experiments_dir=tmp_path)

    projection = await store.get_latest_projection("iran-us")
    forecast_run = await store.get_latest_forecast_run("iran-us")
    page = await store.get_page("projection:iran-us")
    assert projection is not None
    assert forecast_run is not None
    assert all(question.forecast_key for question in forecast_run.questions)
    assert page is not None
    assert "Forward Look" in page["title"]


async def test_generate_projections_from_store_respects_snapshot_override(store, tmp_path):
    event_a = Event(date=date(2026, 3, 1), summary="A", entities=["Iran"])
    event_b = Event(date=date(2026, 3, 10), summary="B", entities=["Iran"])
    event_ids = await store.add_events([event_a, event_b], "iran-us")
    thread_id = await store.upsert_thread("iran-thread", "Iran Thread", 8, "active")
    await store.link_thread_topic(thread_id, "iran-us")
    await store.link_thread_events(thread_id, event_ids)
    await store.upsert_thread_snapshot(
        ThreadSnapshot(
            thread_id=thread_id,
            snapshot_date=date(2026, 3, 1),
            status="active",
            significance=8,
            event_count=1,
            latest_event_date=date(2026, 3, 1),
        )
    )
    await store.upsert_thread_snapshot(
        ThreadSnapshot(
            thread_id=thread_id,
            snapshot_date=date(2026, 3, 10),
            status="active",
            significance=8,
            event_count=2,
            latest_event_date=date(2026, 3, 10),
        )
    )
    await store.save_synthesis(
        TopicSynthesis(
            topic_name="Iran-US",
            threads=[NarrativeThread(
                headline="Iran Thread",
                events=[
                    Event(event_id=event_ids[0], date=date(2026, 3, 1), summary="A", entities=["Iran"]),
                    Event(event_id=event_ids[1], date=date(2026, 3, 10), summary="B", entities=["Iran"]),
                ],
                significance=8,
            )],
        ).model_dump(mode="json"),
        "iran-us",
        date(2026, 3, 10),
    )
    config = NexusConfig(
        user=UserConfig(name="Test User"),
        topics=[TopicConfig(name="Iran-US")],
        future_projection=FutureProjectionConfig(enabled=False, min_history_days=1, min_thread_snapshots=3),
    )

    results = await generate_projections_from_store(
        store,
        None,
        config,
        min_thread_snapshots_override=2,
        experiments_dir=tmp_path,
    )

    assert results[0]["status"] == "ready"
    projection = await store.get_latest_projection("iran-us")
    page = await store.get_page("projection:iran-us")
    assert projection is not None
    assert page is not None
    assert projection.status == "ready"


async def test_backfill_signal_rich_profile_exports_graph_snapshots(store, tmp_path):
    event = Event(date=date(2026, 3, 10), summary="Court filing on sanctions policy", entities=["Iran"])
    event_ids = await store.add_events([event], "iran-us")
    thread_id = await store.upsert_thread("iran-thread", "Iran Thread", 8, "active")
    await store.link_thread_topic(thread_id, "iran-us")
    await store.link_thread_events(thread_id, event_ids)
    await store.upsert_thread_snapshot(
        ThreadSnapshot(
            thread_id=thread_id,
            snapshot_date=date(2026, 3, 10),
            status="active",
            significance=8,
            event_count=1,
            latest_event_date=date(2026, 3, 10),
        )
    )
    await store.save_synthesis(
        TopicSynthesis(
            topic_name="Iran-US",
            threads=[NarrativeThread(
                headline="Iran Thread",
                events=[Event(event_id=event_ids[0], date=date(2026, 3, 10), summary="Court filing on sanctions policy", entities=["Iran"])],
                significance=8,
            )],
        ).model_dump(mode="json"),
        "iran-us",
        date(2026, 3, 10),
    )
    config = NexusConfig(
        user=UserConfig(name="Test User"),
        topics=[TopicConfig(name="Iran-US"), TopicConfig(name="Formula 1")],
        future_projection=FutureProjectionConfig(enabled=False, min_history_days=1, min_thread_snapshots=1),
    )

    result = await backfill_signal_rich_profile(store, config, target_dir=tmp_path / "signal-rich")

    assert result["exports"] == 1
    export_path = tmp_path / "signal-rich" / "iran-us-2026-03-10.json"
    assert export_path.exists()
    payload = json.loads(export_path.read_text())
    assert payload["topic_slug"] == "iran-us"
    assert payload["run_date"] == "2026-03-10"
    assert payload["graph_snapshot"]["topic_slug"] == "iran-us"


def test_render_projection_markdown():
    projection = TopicProjection(
        topic_slug="iran-us",
        topic_name="Iran-US",
        generated_for=date(2026, 3, 10),
        items=[ProjectionItem(
            claim="Sanctions pressure is likely to continue.",
            confidence="medium",
            horizon_days=7,
            signpost="Watch for Treasury action",
            review_after=date(2026, 3, 17),
        )],
    )
    rendered = render_projection_markdown(projection)
    assert "Sanctions pressure" in rendered


# --- Backfill syntheses ---


class _MockLLM:
    """Captures the events passed to synthesize_topic via the LLM prompt."""

    def __init__(self):
        self.calls: list[dict] = []
        self.usage = type("Usage", (), {"cost_summary": lambda self: {}})()

    async def complete(self, *, config_key, system_prompt, user_prompt, json_response=False):
        self.calls.append({"user_prompt": user_prompt})
        # Return a minimal valid synthesis JSON
        return json.dumps({"threads": [{"headline": "Test Thread", "event_indices": [0], "significance": 7}]})


async def test_backfill_syntheses_creates_synthesis_for_each_date(store):
    """Should create a synthesis for each unique event date that doesn't have one."""
    events = [
        Event(date=date(2026, 3, 1), summary="Event A", significance=7, entities=["Iran"]),
        Event(date=date(2026, 3, 5), summary="Event B", significance=7, entities=["Iran"]),
        Event(date=date(2026, 3, 10), summary="Event C", significance=7, entities=["Iran"]),
    ]
    await store.add_events(events, "iran-us")

    llm = _MockLLM()
    config = NexusConfig(
        user=UserConfig(name="Test"),
        topics=[TopicConfig(name="Iran-US")],
    )

    result = await backfill_syntheses(
        store, llm, config,
        topic_slug="iran-us",
    )

    assert result["dates_backfilled"] == 3

    # Count synthesis calls (contain "Events to analyze"); thread persistence may add extra LLM calls
    synthesis_calls = [c for c in llm.calls if "Events to analyze" in c["user_prompt"]]
    assert len(synthesis_calls) == 3

    # Verify syntheses were saved
    synth_dates = sorted(await store.get_synthesis_dates("iran-us"))
    assert len(synth_dates) == 3


async def test_backfill_syntheses_skips_existing_dates(store):
    """Dates that already have syntheses should be skipped."""
    events = [
        Event(date=date(2026, 3, 1), summary="Event A", significance=7, entities=["Iran"]),
        Event(date=date(2026, 3, 5), summary="Event B", significance=7, entities=["Iran"]),
    ]
    await store.add_events(events, "iran-us")

    # Pre-create a synthesis for March 1
    await store.save_synthesis(
        {"topic_name": "Iran-US", "threads": []},
        "iran-us",
        date(2026, 3, 1),
    )

    llm = _MockLLM()
    config = NexusConfig(
        user=UserConfig(name="Test"),
        topics=[TopicConfig(name="Iran-US")],
    )

    result = await backfill_syntheses(
        store, llm, config,
        topic_slug="iran-us",
    )

    assert result["dates_backfilled"] == 1  # Only March 5
    synthesis_calls = [c for c in llm.calls if "Events to analyze" in c["user_prompt"]]
    assert len(synthesis_calls) == 1


async def test_backfill_syntheses_never_sees_future_events(store):
    """Events on day 1, 5, 10; backfill at day 5 should only see day 1+5 events."""
    events = [
        Event(date=date(2026, 3, 1), summary="Day1 event", significance=7, entities=["Iran"]),
        Event(date=date(2026, 3, 5), summary="Day5 event", significance=7, entities=["Iran"]),
        Event(date=date(2026, 3, 10), summary="Day10 event FUTURE", significance=7, entities=["Iran"]),
    ]
    await store.add_events(events, "iran-us")

    llm = _MockLLM()
    config = NexusConfig(
        user=UserConfig(name="Test"),
        topics=[TopicConfig(name="Iran-US")],
    )

    await backfill_syntheses(
        store, llm, config,
        topic_slug="iran-us",
        start=date(2026, 3, 1),
        end=date(2026, 3, 5),
    )

    synthesis_calls = [c for c in llm.calls if "Events to analyze" in c["user_prompt"]]
    assert len(synthesis_calls) == 2  # Day 1 and Day 5

    # The Day 5 synthesis call should NOT mention "Day10" or "FUTURE"
    day5_prompt = synthesis_calls[1]["user_prompt"]
    assert "Day10" not in day5_prompt
    assert "FUTURE" not in day5_prompt
    # But should see Day1 and Day5
    assert "Day1" in day5_prompt
    assert "Day5" in day5_prompt


# ── Kalshi Loop Topic-Keyword Fallback ────────────────────────────


class TestKalshiLoopKeywordFallback:

    async def test_extracts_keywords_from_topic_config(self):
        """run_kalshi_loop with topic_configs should match markets using topic keywords."""
        from unittest.mock import AsyncMock, patch
        from nexus.engine.projection.service import run_kalshi_loop

        mock_store = AsyncMock()
        mock_store.save_forecast_run = AsyncMock()

        mock_client = AsyncMock()
        mock_client.list_events = AsyncMock(return_value={
            "events": [{
                "event_ticker": "IRAN-DEAL",
                "title": "Will Iran nuclear deal happen?",
                "markets": [{
                    "ticker": "IRAN-DEAL-Y",
                    "title": "Iran nuclear deal before June 2026",
                    "subtitle": "Resolves Yes if deal signed",
                    "status": "open",
                    "last_price": 0.25,
                    "volume": 5000,
                }],
            }],
        })

        mock_config = type("Config", (), {
            "max_markets_per_topic": 10,
            "auto_match_min_score": 1,
        })()

        topic_configs = [
            TopicConfig(
                name="Iran-US Relations",
                subtopics=["sanctions", "nuclear program", "diplomacy"],
            ),
        ]

        result = await run_kalshi_loop(
            mock_store, None, [],  # empty syntheses
            run_date=date(2026, 3, 16),
            kalshi_client=mock_client,
            kalshi_config=mock_config,
            topic_configs=topic_configs,
        )
        # Should have matched the Iran market using topic keywords
        assert result["markets_matched"] >= 1

    async def test_empty_syntheses_and_no_topics_matches_nothing(self):
        """With empty syntheses and no topic_configs, should match nothing."""
        from unittest.mock import AsyncMock
        from nexus.engine.projection.service import run_kalshi_loop

        mock_store = AsyncMock()
        mock_client = AsyncMock()
        mock_config = type("Config", (), {
            "max_markets_per_topic": 10,
            "auto_match_min_score": 1,
        })()

        result = await run_kalshi_loop(
            mock_store, None, [],
            run_date=date(2026, 3, 16),
            kalshi_client=mock_client,
            kalshi_config=mock_config,
        )
        assert result["markets_matched"] == 0
