"""Tests for global source registry and topic registry builder."""

import pytest
from pathlib import Path
from nexus.engine.sources.registry import (
    GlobalSource,
    load_global_registry,
    sources_for_topic,
    check_feed_health,
    build_topic_registry,
)
from nexus.config.models import TopicConfig


@pytest.fixture
def registry_path(tmp_path):
    import yaml

    data = {
        "sources": [
            {
                "id": "bbc-world",
                "name": "BBC World",
                "url": "https://feeds.bbci.co.uk/news/world/rss.xml",
                "language": "en",
                "tier": "A",
                "tags": ["world", "politics"],
            },
            {
                "id": "bbc-persian",
                "name": "BBC Persian",
                "url": "https://feeds.bbci.co.uk/persian/rss.xml",
                "language": "fa",
                "tier": "A",
                "tags": ["world", "middle-east", "iran"],
            },
            {
                "id": "arxiv-cs-ai",
                "name": "arXiv CS.AI",
                "url": "https://rss.arxiv.org/rss/cs.AI",
                "language": "en",
                "tier": "A",
                "tags": ["ai", "research"],
            },
            {
                "id": "carbon-brief",
                "name": "Carbon Brief",
                "url": "https://www.carbonbrief.org/feed",
                "language": "en",
                "tier": "B",
                "tags": ["climate", "energy", "policy"],
            },
        ]
    }
    path = tmp_path / "global_registry.yaml"
    path.write_text(yaml.dump(data))
    return path


def test_load_global_registry(registry_path):
    sources = load_global_registry(registry_path)
    assert len(sources) == 4
    assert sources[0].id == "bbc-world"
    assert sources[0].language == "en"
    assert "world" in sources[0].tags
    # Defaults for affiliation/country
    assert sources[0].affiliation == "private"
    assert sources[0].country == ""


def test_affiliation_and_country():
    s = GlobalSource(
        id="cgtn", name="CGTN", url="https://cgtn.com/rss",
        language="en", tier="A", tags=["world"],
        affiliation="state", country="CN",
    )
    assert s.affiliation == "state"
    assert s.country == "CN"


def test_load_global_registry_missing_file(tmp_path):
    sources = load_global_registry(tmp_path / "nope.yaml")
    assert sources == []


def test_sources_for_topic_matches_tags(registry_path):
    sources = load_global_registry(registry_path)
    topic = TopicConfig(
        name="Iran-US Relations",
        priority="high",
        subtopics=["sanctions", "nuclear"],
        source_languages=["en", "fa"],
    )
    matched = sources_for_topic(sources, topic, tag_hints=["middle-east", "iran", "world"])
    ids = [s.id for s in matched]
    # Should match bbc-world (world tag) and bbc-persian (iran tag, fa language)
    assert "bbc-world" in ids
    assert "bbc-persian" in ids
    # Should NOT match arxiv or carbon-brief
    assert "arxiv-cs-ai" not in ids
    assert "carbon-brief" not in ids


def test_sources_for_topic_filters_by_language(registry_path):
    sources = load_global_registry(registry_path)
    topic = TopicConfig(
        name="Iran-US Relations",
        priority="high",
        subtopics=["sanctions"],
        source_languages=["en"],  # English only
    )
    matched = sources_for_topic(sources, topic, tag_hints=["iran", "middle-east", "world"])
    ids = [s.id for s in matched]
    assert "bbc-world" in ids
    assert "bbc-persian" not in ids  # Farsi excluded


def test_sources_for_topic_ai(registry_path):
    sources = load_global_registry(registry_path)
    topic = TopicConfig(
        name="AI/ML Research",
        priority="medium",
        subtopics=["agents"],
        source_languages=["en"],
    )
    matched = sources_for_topic(sources, topic, tag_hints=["ai", "research", "tech"])
    ids = [s.id for s in matched]
    assert "arxiv-cs-ai" in ids
    assert "bbc-persian" not in ids


def test_check_feed_health_with_mock(monkeypatch):
    """Test health check without hitting real network."""
    import feedparser

    fake_feed = feedparser.FeedParserDict()
    fake_feed["entries"] = [{"title": "Test"}] * 10
    fake_feed["feed"] = {"title": "Test Feed"}
    fake_feed["bozo"] = 0

    monkeypatch.setattr(feedparser, "parse", lambda url: fake_feed)

    source = GlobalSource(
        id="test", name="Test", url="https://example.com/feed",
        language="en", tier="A", tags=["test"],
    )
    result = check_feed_health(source)
    assert result["status"] == "ok"
    assert result["entries"] == 10


def test_build_topic_registry_includes_metadata():
    """build_topic_registry must propagate affiliation and country."""
    sources = [
        GlobalSource(
            id="aljazeera", name="Al Jazeera", url="https://aljazeera.com/rss",
            language="en", tier="A", tags=["world"],
            affiliation="state", country="QA",
        ),
        GlobalSource(
            id="bbc-world", name="BBC World", url="https://bbc.com/rss",
            language="en", tier="A", tags=["world"],
            affiliation="public", country="GB",
        ),
    ]
    registry = build_topic_registry(sources)
    entries = registry["sources"]
    assert len(entries) == 2
    assert entries[0]["affiliation"] == "state"
    assert entries[0]["country"] == "QA"
    assert entries[1]["affiliation"] == "public"
    assert entries[1]["country"] == "GB"


def test_check_feed_health_failure(monkeypatch):
    import feedparser

    fake_feed = feedparser.FeedParserDict()
    fake_feed["entries"] = []
    fake_feed["feed"] = {}
    fake_feed["bozo"] = 1

    monkeypatch.setattr(feedparser, "parse", lambda url: fake_feed)

    source = GlobalSource(
        id="test", name="Test", url="https://example.com/feed",
        language="en", tier="A", tags=["test"],
    )
    result = check_feed_health(source)
    assert result["status"] == "empty"
