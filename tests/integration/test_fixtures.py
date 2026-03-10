"""Tests for fixture capture and replay."""

from datetime import date, datetime
from nexus.testing.fixtures import FixtureCapture, FixtureReplay, list_captured_days, partition_by_date
from nexus.engine.sources.polling import ContentItem
from nexus.engine.knowledge.events import Event


def test_capture_and_replay_roundtrip(tmp_path):
    """Capture pipeline data and replay it — verifies serialization."""
    fixture_dir = tmp_path / "fixtures"

    # Capture
    capture = FixtureCapture(fixture_dir, "iran-us-relations", "day-1")

    items = [
        ContentItem(
            title="Sanctions article",
            url="https://example.com/article",
            source_id="nyt",
            snippet="US sanctions...",
            full_text="Full text about sanctions",
            source_language="en",
            source_affiliation="private",
            source_country="US",
            detected_language="en",
            extraction_status="ok",
            relevance_score=8.0,
        ),
    ]
    events = [
        Event(
            date=date(2026, 3, 9),
            summary="US announces new Iran sanctions",
            entities=["US", "Iran", "Treasury"],
            sources=[{"url": "https://example.com/article", "outlet": "nyt"}],
            significance=8,
        ),
    ]

    capture.save_polled(items)
    capture.save_ingested(items)
    capture.save_filtered(items)
    capture.save_events(events)

    # Replay
    replay = FixtureReplay(fixture_dir, "iran-us-relations", "day-1")

    polled = replay.load_polled()
    assert len(polled) == 1
    assert polled[0].title == "Sanctions article"
    assert polled[0].source_affiliation == "private"

    ingested = replay.load_ingested()
    assert len(ingested) == 1
    assert ingested[0].full_text == "Full text about sanctions"

    loaded_events = replay.load_events()
    assert len(loaded_events) == 1
    assert loaded_events[0].summary == "US announces new Iran sanctions"
    assert loaded_events[0].entities == ["US", "Iran", "Treasury"]


def test_list_captured_days(tmp_path):
    fixture_dir = tmp_path / "fixtures"
    FixtureCapture(fixture_dir, "test-topic", "day-1")
    FixtureCapture(fixture_dir, "test-topic", "day-2")

    days = list_captured_days(fixture_dir, "test-topic")
    assert days == ["day-1", "day-2"]


def test_list_captured_days_empty(tmp_path):
    assert list_captured_days(tmp_path / "fixtures", "nonexistent") == []


def test_replay_missing_files(tmp_path):
    """Replay with missing files returns empty lists."""
    fixture_dir = tmp_path / "fixtures"
    FixtureCapture(fixture_dir, "test-topic", "day-1")
    replay = FixtureReplay(fixture_dir, "test-topic", "day-1")

    # Dir exists but files haven't been written
    # save_polled was not called, so polled.json doesn't exist
    # Actually the capture creates the dir, let's test replay directly
    empty_dir = fixture_dir / "empty-topic" / "day-1"
    empty_dir.mkdir(parents=True)
    replay2 = FixtureReplay(fixture_dir, "empty-topic", "day-1")
    assert replay2.load_polled() == []
    assert replay2.load_events() == []


def test_partition_by_date():
    """Items are grouped by published date, sorted chronologically."""
    items = [
        ContentItem(title="A", url="https://a.com", source_id="s1",
                    published=datetime(2026, 3, 5, 10, 0)),
        ContentItem(title="B", url="https://b.com", source_id="s2",
                    published=datetime(2026, 3, 7, 14, 30)),
        ContentItem(title="C", url="https://c.com", source_id="s3",
                    published=datetime(2026, 3, 5, 18, 0)),
        ContentItem(title="D", url="https://d.com", source_id="s4",
                    published=datetime(2026, 3, 6, 9, 0)),
    ]
    groups = partition_by_date(items)
    days = list(groups.keys())
    assert days == [date(2026, 3, 5), date(2026, 3, 6), date(2026, 3, 7)]
    assert len(groups[date(2026, 3, 5)]) == 2
    assert len(groups[date(2026, 3, 6)]) == 1
    assert len(groups[date(2026, 3, 7)]) == 1


def test_partition_by_date_no_published():
    """Items without published date fall into reference date bucket."""
    items = [
        ContentItem(title="A", url="https://a.com", source_id="s1"),
    ]
    ref = date(2026, 3, 10)
    groups = partition_by_date(items, reference_date=ref)
    assert ref in groups
    assert len(groups[ref]) == 1


def test_partition_by_date_drops_stale_articles():
    """Articles older than max_age_days are dropped."""
    ref = date(2026, 3, 10)
    items = [
        ContentItem(title="Current", url="https://a.com", source_id="s1",
                    published=datetime(2026, 3, 9, 10, 0)),
        ContentItem(title="Stale 2017", url="https://b.com", source_id="s2",
                    published=datetime(2017, 3, 30, 12, 0)),
        ContentItem(title="Old 30 days", url="https://c.com", source_id="s3",
                    published=datetime(2026, 2, 8, 12, 0)),
    ]
    groups = partition_by_date(items, max_age_days=14, reference_date=ref)
    # Only the current article should remain
    assert len(groups) == 1
    assert date(2026, 3, 9) in groups
    assert groups[date(2026, 3, 9)][0].title == "Current"


def test_partition_by_date_drops_future_articles():
    """Articles with future dates are dropped."""
    ref = date(2026, 3, 10)
    items = [
        ContentItem(title="Today", url="https://a.com", source_id="s1",
                    published=datetime(2026, 3, 10, 8, 0)),
        ContentItem(title="Future", url="https://b.com", source_id="s2",
                    published=datetime(2026, 3, 15, 12, 0)),
    ]
    groups = partition_by_date(items, reference_date=ref)
    assert len(groups) == 1
    assert date(2026, 3, 10) in groups
