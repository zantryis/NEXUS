"""Live integration test for source discovery — hits real Gemini API + DuckDuckGo.

Run with: .venv/bin/pytest tests/integration/test_discovery_live.py -v -s -m integration
"""

import logging
import os
import pytest
from pathlib import Path

from dotenv import load_dotenv

# Load .env before anything else
_env_path = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(_env_path)

from nexus.config.models import ModelsConfig
from nexus.llm.client import LLMClient
from nexus.engine.sources.discovery import (
    _generate_search_queries,
    _find_rss_feeds,
    _validate_feed,
    _discover_from_google_news,
    _match_from_global_registry,
    _load_global_registry,
    discover_sources,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Use the actual data dir for global registry
DATA_DIR = Path(__file__).resolve().parents[2] / "data"

_gemini_key = os.getenv("GEMINI_API_KEY", "")
pytestmark = pytest.mark.integration


@pytest.fixture
def llm():
    """Real LLM client with Gemini keys from .env."""
    if not _gemini_key:
        pytest.skip("GEMINI_API_KEY not set")
    models = ModelsConfig(discovery="gemini-3-flash-preview")
    return LLMClient(models_config=models, api_key=_gemini_key)


# ── Individual step tests ───────────────────────────────────────────


@pytest.mark.integration
async def test_step1_global_registry_matching(llm):
    """Test LLM-scored matching from curated global registry."""
    global_sources = _load_global_registry(DATA_DIR)
    assert len(global_sources) > 0, "Global registry should have sources"
    logger.info(f"Global registry has {len(global_sources)} sources")

    matches = await _match_from_global_registry(
        llm, "Iran-US Relations",
        subtopics=["sanctions", "nuclear negotiations", "IRGC"],
        global_sources=global_sources,
        existing_urls=set(),
    )
    logger.info(f"Registry matches: {len(matches)}")
    for m in matches:
        logger.info(f"  - {m['name']} ({m.get('tier', '?')}) {m['url']}")

    assert len(matches) > 0, "Should find some relevant sources in global registry"


@pytest.mark.integration
async def test_step2_google_news_rss():
    """Test Google News RSS feed generation and validation."""
    feeds = await _discover_from_google_news(
        "Iran-US Relations",
        subtopics=["sanctions", "nuclear negotiations"],
        existing_urls=set(),
    )
    logger.info(f"Google News feeds: {len(feeds)}")
    for f in feeds:
        logger.info(f"  - {f['name']} — {f['url'][:80]}...")

    # Google News should always return at least the main topic feed
    assert len(feeds) >= 1, "Google News should return at least 1 feed"


@pytest.mark.integration
async def test_step3a_query_generation(llm):
    """Test LLM query generation for RSS feed search."""
    queries = await _generate_search_queries(
        llm, "Iran-US Relations",
        subtopics=["sanctions", "nuclear negotiations", "IRGC"],
    )
    logger.info(f"Generated {len(queries)} queries:")
    for q in queries:
        logger.info(f"  - {q}")

    assert len(queries) >= 2, "Should generate at least 2 queries"
    assert all(isinstance(q, str) for q in queries)


@pytest.mark.integration
async def test_step3b_web_search_and_validation():
    """Test DuckDuckGo search → feed validation pipeline.

    This is the suspected weak point: DDG returns web pages, not RSS feeds.
    """
    query = "Iran US relations news RSS feed"
    logger.info(f"Searching: {query}")

    urls = await _find_rss_feeds(query)
    logger.info(f"Found {len(urls)} candidate URLs from web search")
    for u in urls:
        logger.info(f"  - {u}")

    # Now validate each as a feed
    valid_count = 0
    for url in urls[:15]:  # cap to avoid slow test
        result = await _validate_feed(url)
        if result:
            valid_count += 1
            logger.info(f"  VALID FEED: {result['name']} — {url}")
        else:
            logger.info(f"  NOT A FEED: {url}")

    logger.info(f"\nConversion rate: {valid_count}/{len(urls)} URLs were valid RSS feeds")
    # This is the key metric — if this is 0, web discovery is broken


@pytest.mark.integration
async def test_step3b_targeted_rss_search():
    """Try more RSS-specific search queries to see if we can find feeds."""
    queries = [
        "site:feeds.feedburner.com iran",
        "iran news atom feed xml",
        "middle east policy RSS feed URL",
        "iran sanctions news feed.xml",
    ]
    total_urls = 0
    valid_feeds = 0

    for query in queries:
        urls = await _find_rss_feeds(query)
        total_urls += len(urls)
        for url in urls[:5]:
            result = await _validate_feed(url)
            if result:
                valid_feeds += 1
                logger.info(f"  VALID: {result['name']} — {url}")

    logger.info(f"\nTargeted search: {valid_feeds} valid feeds from {total_urls} URLs across {len(queries)} queries")


@pytest.mark.integration
async def test_full_discovery_pipeline(llm):
    """Run the complete discovery pipeline and report results."""
    logger.info("=" * 60)
    logger.info("FULL DISCOVERY PIPELINE TEST: Iran-US Relations")
    logger.info("=" * 60)

    result = await discover_sources(
        llm=llm,
        topic_name="Iran-US Relations",
        subtopics=["sanctions", "nuclear negotiations", "IRGC", "Middle East geopolitics"],
        existing_urls=set(),
        max_feeds=25,
        data_dir=DATA_DIR,
        max_rounds=2,
        max_llm_calls=8,
    )

    logger.info(f"\n{'=' * 60}")
    logger.info(f"RESULTS:")
    logger.info(f"  Total feeds: {len(result.feeds)}")
    logger.info(f"  From registry: {result.sources_from_registry}")
    logger.info(f"  From Google News: {result.sources_from_google_news}")
    logger.info(f"  From web search: {result.sources_from_web}")
    logger.info(f"\nDiversity:")
    logger.info(f"  Geographic: {result.diversity.geographic_score:.2f}")
    logger.info(f"  Affiliation: {result.diversity.affiliation_score:.2f}")
    logger.info(f"  Language: {result.diversity.language_score:.2f}")
    logger.info(f"  Overall: {result.diversity.overall:.2f}")

    logger.info(f"\nFeeds found:")
    for f in result.feeds:
        logger.info(
            f"  [{f.get('tier', '?')}] {f.get('name', '?')} "
            f"({f.get('affiliation', '?')}/{f.get('country', '?')}) "
            f"— {f['url'][:70]}"
        )

    if result.diversity.warnings:
        logger.info(f"\nDiversity warnings:")
        for w in result.diversity.warnings:
            logger.info(f"  ⚠ {w}")

    # Basic assertions
    assert len(result.feeds) > 0, "Should discover at least some feeds"
    assert result.sources_from_registry > 0 or result.sources_from_google_news > 0, \
        "Should get feeds from registry or Google News at minimum"

    # Report web search effectiveness
    if result.sources_from_web == 0:
        logger.warning(
            "\n⚠ WEB SEARCH FOUND 0 FEEDS — this confirms the DDG→RSS conversion problem"
        )
