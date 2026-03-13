"""Tests for source auto-discovery."""

from unittest.mock import AsyncMock, patch, MagicMock

from nexus.engine.sources.discovery import (
    _generate_search_queries,
    _find_rss_feeds,
    _validate_feed,
    _evaluate_feed_relevance,
    _generate_refined_queries,
    discover_sources,
)


@patch("nexus.engine.sources.discovery.web_search")
async def test_find_rss_feeds_extracts_urls(mock_search):
    """Finds RSS feed URLs from web search results."""
    mock_search.return_value = [
        {"title": "Best Horticulture RSS Feeds", "snippet": "Top feeds...",
         "url": "https://example.com/feeds"},
        {"title": "Garden RSS", "snippet": "RSS at https://garden.org/rss",
         "url": "https://garden.org"},
    ]

    feeds = await _find_rss_feeds("horticulture RSS feeds")
    # Should return URLs from search results
    assert isinstance(feeds, list)


async def test_generate_search_queries():
    """LLM generates useful search queries for a topic."""
    llm = AsyncMock()
    llm.complete = AsyncMock(return_value='["horticulture news RSS feed", "gardening RSS feed"]')

    queries = await _generate_search_queries(llm, "horticulture", ["gardening", "plant science"])
    assert len(queries) >= 1
    assert any("horticulture" in q.lower() or "garden" in q.lower() for q in queries)


@patch("nexus.engine.sources.discovery.feedparser")
async def test_validate_feed_valid(mock_fp):
    """Valid feed returns source dict."""
    mock_fp.parse.return_value = MagicMock(
        bozo=False,
        feed=MagicMock(title="Garden News", language="en"),
        entries=[MagicMock(), MagicMock()],
    )

    result = await _validate_feed("https://garden.org/rss")
    assert result is not None
    assert result["url"] == "https://garden.org/rss"
    assert result["type"] == "rss"


@patch("nexus.engine.sources.discovery.feedparser")
async def test_validate_feed_invalid(mock_fp):
    """Invalid/empty feed returns None."""
    mock_fp.parse.return_value = MagicMock(
        bozo=True,
        entries=[],
    )

    result = await _validate_feed("https://bad.com/rss")
    assert result is None


@patch("nexus.engine.sources.discovery.classify_feed_metadata", new_callable=AsyncMock)
@patch("nexus.engine.sources.discovery._evaluate_feed_relevance", new_callable=AsyncMock)
@patch("nexus.engine.sources.discovery._validate_feed")
@patch("nexus.engine.sources.discovery._find_rss_feeds")
@patch("nexus.engine.sources.discovery._generate_search_queries")
@patch("nexus.engine.sources.discovery._discover_from_google_news")
async def test_discover_sources_pipeline(mock_gnews, mock_queries, mock_find, mock_validate, mock_evaluate, mock_classify):
    """Full discovery pipeline: queries -> search -> validate -> evaluate."""
    mock_gnews.return_value = []
    mock_queries.return_value = ["horticulture RSS"]
    mock_find.return_value = ["https://garden.org/rss", "https://plants.com/feed"]
    mock_validate.side_effect = [
        {"id": "garden-org", "url": "https://garden.org/rss", "type": "rss",
         "language": "en", "affiliation": "unknown", "country": "unknown", "tier": "B",
         "name": "Garden Org"},
        None,  # invalid feed
    ]
    mock_evaluate.side_effect = lambda llm, feeds, topic: feeds  # passthrough
    mock_classify.side_effect = lambda llm, feeds: feeds  # passthrough

    llm = AsyncMock()
    result = await discover_sources(
        llm, "horticulture", subtopics=["gardening"],
        max_rounds=1,
    )

    assert len(result.feeds) == 1
    assert result.feeds[0]["url"] == "https://garden.org/rss"
    assert result.sources_from_web == 1


@patch("nexus.engine.sources.discovery._sample_feed_titles", new_callable=AsyncMock)
async def test_evaluate_feed_relevance_filters_low_scores(mock_titles):
    """Feeds scoring below 5 are dropped."""
    mock_titles.side_effect = [
        ["TSMC announces new 2nm fab", "Intel gets CHIPS Act funding"],
        ["Celebrity gossip today", "Sports scores roundup"],
    ]
    llm = AsyncMock()
    llm.complete = AsyncMock(return_value='[{"score": 9, "reason": "dedicated semiconductor coverage"}, {"score": 2, "reason": "unrelated tabloid"}]')

    feeds = [
        {"id": "semi-feed", "url": "https://semi.com/rss", "name": "Semi News"},
        {"id": "tabloid", "url": "https://gossip.com/rss", "name": "Gossip Daily"},
    ]
    result = await _evaluate_feed_relevance(llm, feeds, "Semiconductor Supply Chain")
    assert len(result) == 1
    assert result[0]["id"] == "semi-feed"


@patch("nexus.engine.sources.discovery._sample_feed_titles", new_callable=AsyncMock)
async def test_evaluate_feed_relevance_keeps_all_on_failure(mock_titles):
    """All feeds kept when LLM evaluation fails."""
    mock_titles.return_value = ["Some title"]
    llm = AsyncMock()
    llm.complete = AsyncMock(side_effect=Exception("LLM error"))

    feeds = [
        {"id": "a", "url": "https://a.com/rss", "name": "Feed A"},
    ]
    result = await _evaluate_feed_relevance(llm, feeds, "Any Topic")
    assert len(result) == 1


async def test_generate_refined_queries():
    """Refined queries are generated from previous context."""
    llm = AsyncMock()
    llm.complete = AsyncMock(
        return_value='["semiconductor trade journal RSS", "chip industry association news feed"]'
    )

    queries = await _generate_refined_queries(
        llm, "Semiconductor Supply Chain",
        subtopics=["EUV lithography"],
        previous_queries=["semiconductor news RSS feed"],
    )
    assert len(queries) >= 1
    assert all(isinstance(q, str) for q in queries)


async def test_discover_sources_deduplicates():
    """Discovery deduplicates by URL."""
    from nexus.engine.sources.discovery import _deduplicate
    feeds = [
        {"url": "https://a.com/rss", "id": "a"},
        {"url": "https://a.com/rss", "id": "a-dup"},
        {"url": "https://b.com/rss", "id": "b"},
    ]
    deduped = _deduplicate(feeds)
    assert len(deduped) == 2
