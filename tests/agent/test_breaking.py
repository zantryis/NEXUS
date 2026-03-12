"""Tests for topic-scoped breaking news poller."""

import json
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from nexus.config.models import (
    NexusConfig, UserConfig, BreakingNewsConfig, TopicConfig,
)
from nexus.agent.breaking import check_breaking_news, _hash_headline, MAX_ALERTS_PER_TOPIC
from nexus.engine.sources.polling import ContentItem


def _make_config(topics=None, **bn_kw) -> NexusConfig:
    if topics is None:
        topics = [
            TopicConfig(
                name="Iran-US Relations",
                subtopics=["sanctions", "nuclear"],
            ),
        ]
    return NexusConfig(
        user=UserConfig(name="Tristan"),
        breaking_news=BreakingNewsConfig(**bn_kw),
        topics=topics,
    )


def _make_items(headlines: list[str]) -> list[ContentItem]:
    return [
        ContentItem(title=h, url=f"https://test.com/{i}", source_id="test")
        for i, h in enumerate(headlines)
    ]


def _mock_store(alerted_hashes: set | None = None):
    """Create a mock store with get_alerted_hashes returning given set."""
    store = AsyncMock()
    store.get_alerted_hashes = AsyncMock(return_value=alerted_hashes or set())
    store.add_breaking_alert = AsyncMock(return_value=1)
    return store


def test_hash_headline():
    h = _hash_headline("Major event happened")
    assert isinstance(h, str)
    assert len(h) == 16
    assert _hash_headline("Major event happened") == h
    assert _hash_headline("Different headline") != h


async def test_breaking_news_disabled():
    config = _make_config(enabled=False)
    result = await check_breaking_news(AsyncMock(), config, AsyncMock())
    assert result == {}


@patch("nexus.agent.breaking._poll_all_feeds")
async def test_breaking_news_returns_topic_grouped_dict(mock_poll):
    """Alerts are grouped by topic slug."""
    mock_poll.return_value = _make_items(["Iran sanctions expanded"])

    store = _mock_store()

    llm = AsyncMock()
    llm.complete = AsyncMock(return_value=json.dumps([
        {"id": 0, "score": 9, "reason": "highly relevant"}
    ]))

    config = _make_config(threshold=7)
    result = await check_breaking_news(llm, config, store)

    assert isinstance(result, dict)
    assert "iran-us-relations" in result
    assert len(result["iran-us-relations"]) == 1
    assert result["iran-us-relations"][0]["headline"] == "Iran sanctions expanded"
    assert result["iran-us-relations"][0]["significance_score"] == 9


@patch("nexus.agent.breaking._poll_all_feeds")
async def test_breaking_news_skips_alerted(mock_poll):
    """Already-alerted headlines per topic are skipped."""
    items = _make_items(["Old news"])
    mock_poll.return_value = items

    # Return the hash of "Old news" as already alerted
    alerted = {_hash_headline("Old news")}
    store = _mock_store(alerted_hashes=alerted)

    config = _make_config()
    result = await check_breaking_news(AsyncMock(), config, store)

    assert result == {}


@patch("nexus.agent.breaking._poll_all_feeds")
async def test_breaking_news_below_threshold(mock_poll):
    """Headlines below threshold don't become alerts."""
    mock_poll.return_value = _make_items(["Minor update"])

    store = _mock_store()

    llm = AsyncMock()
    llm.complete = AsyncMock(return_value=json.dumps([
        {"id": 0, "score": 3, "reason": "not relevant"}
    ]))

    config = _make_config(threshold=7)
    result = await check_breaking_news(llm, config, store)

    assert result == {}


@patch("nexus.agent.breaking._poll_all_feeds")
async def test_breaking_news_multi_topic(mock_poll):
    """Headlines scored against multiple topics concurrently."""
    mock_poll.return_value = _make_items(["AI breakthrough announced"])

    store = _mock_store()

    # LLM scores high for both topics
    llm = AsyncMock()
    llm.complete = AsyncMock(return_value=json.dumps([
        {"id": 0, "score": 8, "reason": "relevant"}
    ]))

    topics = [
        TopicConfig(name="Iran-US Relations", subtopics=["sanctions"]),
        TopicConfig(name="AI/ML Research", subtopics=["agents"]),
    ]
    config = _make_config(topics=topics, threshold=7)
    result = await check_breaking_news(llm, config, store)

    # Both topics should have scored the headline
    assert store.add_breaking_alert.call_count == 2


@patch("nexus.agent.breaking._poll_all_feeds")
async def test_breaking_stores_alerts_with_topic_slug(mock_poll):
    """Alerts are stored with topic_slug."""
    mock_poll.return_value = _make_items(["Crisis erupts"])

    store = _mock_store()

    llm = AsyncMock()
    llm.complete = AsyncMock(return_value=json.dumps([
        {"id": 0, "score": 9, "reason": "highly relevant"}
    ]))

    config = _make_config(threshold=7)
    await check_breaking_news(llm, config, store)

    # Verify topic_slug was passed
    call_kwargs = store.add_breaking_alert.call_args
    assert call_kwargs.kwargs.get("topic_slug") == "iran-us-relations"


@patch("nexus.agent.breaking._poll_all_feeds")
async def test_breaking_caps_alerts_per_topic(mock_poll):
    """No more than MAX_ALERTS_PER_TOPIC alerts per topic."""
    # Generate many headlines
    headlines = [f"Breaking event {i}" for i in range(20)]
    mock_poll.return_value = _make_items(headlines)

    store = _mock_store()

    # LLM scores ALL headlines high
    def mock_complete(**kwargs):
        # Parse out how many articles in the prompt
        prompt = kwargs.get("user_prompt", "")
        scores = []
        for i in range(20):
            if f"[{i}]" in prompt:
                scores.append({"id": i, "score": 9, "reason": "all high"})
        return json.dumps(scores)

    llm = AsyncMock()
    llm.complete = AsyncMock(side_effect=mock_complete)

    config = _make_config(threshold=7)
    result = await check_breaking_news(llm, config, store)

    assert "iran-us-relations" in result
    assert len(result["iran-us-relations"]) <= MAX_ALERTS_PER_TOPIC
