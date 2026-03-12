"""E2E tests for source discovery — hits real APIs.

Run with: .venv/bin/pytest tests/e2e/test_discovery.py -m e2e -v
"""

import os
import pytest
from pathlib import Path

from nexus.config.models import ModelsConfig
from nexus.engine.sources.discovery import discover_sources
from nexus.llm.client import LLMClient


pytestmark = [pytest.mark.e2e, pytest.mark.asyncio]


@pytest.fixture
def llm():
    """Real LLM client from environment keys."""
    return LLMClient(
        ModelsConfig(),
        api_key=os.getenv("GEMINI_API_KEY"),
        deepseek_api_key=os.getenv("DEEPSEEK_API_KEY") or os.getenv("deepseek"),
    )


async def test_global_registry_matching(llm, tmp_path):
    """Discovery finds feeds from global registry for known topics."""
    # Copy global registry to temp data dir
    import shutil
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    src_registry = Path("data/sources/global_registry.yaml")
    if src_registry.exists():
        dest = data_dir / "sources" / "global_registry.yaml"
        dest.parent.mkdir(parents=True)
        shutil.copy2(src_registry, dest)

    result = await discover_sources(
        llm, "AI/ML Research",
        subtopics=["machine learning", "LLM"],
        max_feeds=10,
        data_dir=data_dir,
    )

    assert len(result.feeds) > 0, "No feeds discovered"
    assert result.sources_from_registry > 0, "No registry matches found"


async def test_diversity_scoring(llm, tmp_path):
    """Discovered feed set gets meaningful diversity metrics."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    # Copy global registry
    import shutil
    src_registry = Path("data/sources/global_registry.yaml")
    if src_registry.exists():
        dest = data_dir / "sources" / "global_registry.yaml"
        dest.parent.mkdir(parents=True)
        shutil.copy2(src_registry, dest)

    result = await discover_sources(
        llm, "Global Energy",
        subtopics=["renewable energy", "oil", "climate"],
        max_feeds=15,
        data_dir=data_dir,
    )

    assert result.diversity.overall >= 0, "Diversity not computed"
    if len(result.feeds) >= 5:
        assert result.diversity.geographic_score > 0, "No geographic diversity"
