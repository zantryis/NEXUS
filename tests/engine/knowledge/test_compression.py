"""Tests for knowledge layer compression — weekly/monthly rollups."""

import pytest
from datetime import date
from pathlib import Path
from unittest.mock import AsyncMock
from nexus.engine.knowledge.events import Event
from nexus.engine.knowledge.compression import (
    Summary,
    group_events_by_week,
    compress_to_weekly,
    load_summaries,
    save_summaries,
)


def make_events(dates_and_summaries):
    return [
        Event(date=d, summary=s, significance=5)
        for d, s in dates_and_summaries
    ]


def test_group_events_by_week():
    events = make_events([
        (date(2026, 2, 16), "Monday week 1"),
        (date(2026, 2, 18), "Wednesday week 1"),
        (date(2026, 2, 23), "Monday week 2"),
    ])
    groups = group_events_by_week(events)
    assert len(groups) == 2


@pytest.mark.asyncio
async def test_compress_to_weekly():
    mock_llm = AsyncMock()
    mock_llm.complete.return_value = "Week summary: Two events happened."

    events = make_events([
        (date(2026, 2, 2), "Event A"),  # Monday
        (date(2026, 2, 4), "Event B"),  # Wednesday, same ISO week
    ])

    summaries = await compress_to_weekly(mock_llm, events, "AI Research")
    assert len(summaries) == 1
    assert "Two events" in summaries[0].text


def test_save_and_load_summaries(tmp_path):
    path = tmp_path / "weekly.yaml"
    summaries = [
        Summary(
            period_start=date(2026, 2, 1),
            period_end=date(2026, 2, 7),
            text="Summary of week 1",
            event_count=3,
        ),
    ]
    save_summaries(path, summaries)
    loaded = load_summaries(path)
    assert len(loaded) == 1
    assert loaded[0].text == "Summary of week 1"
    assert loaded[0].event_count == 3
