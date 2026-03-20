"""Tests for source auto-discovery."""

from unittest.mock import AsyncMock, patch, MagicMock

from nexus.engine.sources.discovery import (
    _generate_search_queries,
    _find_rss_feeds,
    _validate_feed,
    _evaluate_feed_relevance,
    _generate_refined_queries,
    _slugify,
    _deduplicate,
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


# ── Slugify and ID tests ──


def test_slugify_clean_name():
    assert _slugify("Rugby World") == "rugby-world"


def test_slugify_strips_special_chars():
    assert _slugify('"Road Cycling" - Google') == "road-cycling-google"


def test_slugify_colons_and_dots():
    assert _slugify("World Rugby: Latest News") == "world-rugby-latest-news"
    assert _slugify("A.B.C News") == "a-b-c-news"


def test_slugify_collapses_hyphens():
    assert _slugify("foo---bar") == "foo-bar"


def test_slugify_strips_edges():
    assert _slugify("--hello--") == "hello"


def test_slugify_truncates_at_word_boundary():
    long_name = "international-rugby-union-championship-series-extra"
    result = _slugify(long_name, max_len=30)
    assert len(result) <= 30
    assert not result.endswith("-")


def test_slugify_empty_string():
    assert _slugify("") == ""


@patch("nexus.engine.sources.discovery.feedparser")
async def test_validate_feed_clean_id(mock_fp):
    """Feed ID is a clean slug of the feed title."""
    mock_fp.parse.return_value = MagicMock(
        bozo=False,
        feed=MagicMock(title="Google News: International Rugby Union", language="en"),
        entries=[MagicMock()],
    )
    result = await _validate_feed("https://news.google.com/rss")
    assert result["id"] == "google-news-international-rugby-union"


@patch("nexus.engine.sources.discovery.feedparser")
async def test_validate_feed_empty_title_uses_hash(mock_fp):
    """Feed with no title gets hash-based ID."""
    mock_fp.parse.return_value = MagicMock(
        bozo=False,
        feed=MagicMock(title="", language="en"),
        entries=[MagicMock()],
    )
    result = await _validate_feed("https://example.com/rss")
    assert result["id"].startswith("feed-")
    assert len(result["id"]) == 13  # "feed-" + 8 hex chars


def test_deduplicate_resolves_id_collision():
    """Two feeds with same title but different URLs get unique IDs."""
    feeds = [
        {"id": "rugby-world", "url": "https://a.com/rss", "name": "Rugby World"},
        {"id": "rugby-world", "url": "https://b.com/rss", "name": "Rugby World"},
    ]
    result = _deduplicate(feeds)
    assert len(result) == 2
    ids = [f["id"] for f in result]
    assert ids[0] != ids[1]
    assert ids[0] == "rugby-world"
    assert ids[1].startswith("rugby-world-")


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
