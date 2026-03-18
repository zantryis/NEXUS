"""Multi-topic discovery quality test — hits real Gemini API + DuckDuckGo.

Run with: .venv/bin/pytest tests/integration/test_discovery_topics.py -v -s --log-cli-level=INFO -m integration
"""

import logging
import os
import pytest
from pathlib import Path

from dotenv import load_dotenv

_env_path = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(_env_path)

from nexus.config.models import ModelsConfig
from nexus.llm.client import LLMClient
from nexus.engine.sources.discovery import discover_sources

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
_gemini_key = os.getenv("GEMINI_API_KEY", "")
pytestmark = pytest.mark.integration


@pytest.fixture
def llm():
    if not _gemini_key:
        pytest.skip("GEMINI_API_KEY not set")
    models = ModelsConfig(discovery="gemini-3-flash-preview")
    return LLMClient(models_config=models, api_key=_gemini_key)


def _report(topic, result):
    logger.info(f"\n{'=' * 70}")
    logger.info(f"TOPIC: {topic}")
    logger.info(f"{'=' * 70}")
    logger.info(f"Total: {len(result.feeds)}  |  Registry: {result.sources_from_registry}  |  Google News: {result.sources_from_google_news}  |  Web: {result.sources_from_web}")
    logger.info(f"Diversity — geo: {result.diversity.geographic_score:.2f}  affil: {result.diversity.affiliation_score:.2f}  lang: {result.diversity.language_score:.2f}  overall: {result.diversity.overall:.2f}")
    if result.diversity.warnings:
        for w in result.diversity.warnings:
            logger.info(f"  ⚠ {w}")
    logger.info(f"\nFeeds:")
    for f in result.feeds:
        src = "REG" if f.get("tags") else ("GOOG" if "google.com" in f["url"] else "WEB")
        logger.info(
            f"  [{f.get('tier','?')}] [{src:4}] {f.get('name','?')[:45]:45} "
            f"({f.get('affiliation','?'):10}/{f.get('country','?'):2}) "
            f"{f['url'][:65]}"
        )


@pytest.mark.integration
async def test_discover_beauty(llm):
    result = await discover_sources(
        llm=llm, topic_name="Beauty Industry",
        subtopics=["skincare trends", "cosmetics regulation", "K-beauty", "clean beauty"],
        existing_urls=set(), max_feeds=25, data_dir=DATA_DIR,
    )
    _report("Beauty Industry", result)
    assert len(result.feeds) >= 5


@pytest.mark.integration
async def test_discover_fashion(llm):
    result = await discover_sources(
        llm=llm, topic_name="Fashion Industry",
        subtopics=["luxury brands", "sustainable fashion", "fashion week", "fast fashion labor"],
        existing_urls=set(), max_feeds=25, data_dir=DATA_DIR,
    )
    _report("Fashion Industry", result)
    assert len(result.feeds) >= 5


@pytest.mark.integration
async def test_discover_rugby(llm):
    result = await discover_sources(
        llm=llm, topic_name="International Rugby Union",
        subtopics=["Six Nations", "Rugby World Cup", "Super Rugby", "World Rugby rankings"],
        existing_urls=set(), max_feeds=25, data_dir=DATA_DIR,
    )
    _report("International Rugby Union", result)
    assert len(result.feeds) >= 5


@pytest.mark.integration
async def test_discover_rwanda(llm):
    result = await discover_sources(
        llm=llm, topic_name="Rwanda News & Politics",
        subtopics=["Kagame government", "DRC conflict", "East African Community", "Rwandan economy"],
        existing_urls=set(), max_feeds=25, data_dir=DATA_DIR,
    )
    _report("Rwanda News & Politics", result)
    assert len(result.feeds) >= 3
