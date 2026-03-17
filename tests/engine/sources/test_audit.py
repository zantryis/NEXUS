"""Tests for source quality audit tool."""

from unittest.mock import AsyncMock, patch

import pytest

from nexus.config.models import TopicConfig
from nexus.engine.sources.polling import ContentItem


def _make_articles(n: int = 5) -> list[ContentItem]:
    return [
        ContentItem(
            title=f"Article {i}",
            url=f"https://example.com/{i}",
            source_id="test-source",
            text=f"Body text for article {i}.",
        )
        for i in range(n)
    ]


class TestAuditSource:
    @pytest.mark.asyncio
    async def test_audit_returns_score_and_verdict(self):
        """audit_source returns mean score and keep/review/drop verdict."""
        from nexus.engine.sources.audit import audit_source

        topic = TopicConfig(name="AI ML Research", subtopics=["llm"])
        articles = _make_articles(5)

        # Mock score_batch to return high scores
        with patch("nexus.engine.sources.audit.score_batch",
                    return_value=[(8, "relevant"), (7, "relevant"),
                                  (9, "very relevant"), (6, "somewhat"),
                                  (8, "relevant")]):
            result = await audit_source(
                llm=AsyncMock(),
                source_id="test-source",
                articles=articles,
                topic=topic,
            )

        assert result["source_id"] == "test-source"
        assert result["mean_score"] == pytest.approx(7.6, abs=0.1)
        assert result["verdict"] == "keep"
        assert result["n_articles"] == 5

    @pytest.mark.asyncio
    async def test_low_score_returns_drop(self):
        """Sources scoring below 0.2 * 10 = 2.0 are marked drop."""
        from nexus.engine.sources.audit import audit_source

        topic = TopicConfig(name="AI ML Research", subtopics=["llm"])
        articles = _make_articles(3)

        with patch("nexus.engine.sources.audit.score_batch",
                    return_value=[(1, "irrelevant"), (2, "barely"), (1, "no")]):
            result = await audit_source(
                llm=AsyncMock(),
                source_id="low-source",
                articles=articles,
                topic=topic,
            )

        assert result["verdict"] == "drop"

    @pytest.mark.asyncio
    async def test_medium_score_returns_review(self):
        """Sources scoring between drop and keep thresholds are marked review."""
        from nexus.engine.sources.audit import audit_source

        topic = TopicConfig(name="AI ML Research", subtopics=["llm"])
        articles = _make_articles(3)

        with patch("nexus.engine.sources.audit.score_batch",
                    return_value=[(4, "some"), (3, "maybe"), (4, "ok")]):
            result = await audit_source(
                llm=AsyncMock(),
                source_id="mid-source",
                articles=articles,
                topic=topic,
            )

        assert result["verdict"] == "review"


class TestAuditRegistry:
    @pytest.mark.asyncio
    async def test_audit_all_sources(self):
        """audit_registry audits each source and returns summary."""
        from nexus.engine.sources.audit import audit_registry

        topic = TopicConfig(name="AI ML Research", subtopics=["llm"])
        sources = [
            {"id": "src-a", "url": "https://a.com/rss", "type": "rss"},
            {"id": "src-b", "url": "https://b.com/rss", "type": "rss"},
        ]

        articles_a = _make_articles(3)
        for a in articles_a:
            a.source_id = "src-a"
        articles_b = _make_articles(3)
        for a in articles_b:
            a.source_id = "src-b"

        with patch("nexus.engine.sources.audit.poll_all_feeds",
                    return_value=articles_a + articles_b), \
             patch("nexus.engine.sources.audit.score_batch",
                    return_value=[(7, "good"), (8, "good"), (6, "ok")]):

            results = await audit_registry(
                llm=AsyncMock(),
                sources=sources,
                topic=topic,
            )

        assert len(results) == 2
        assert all(r["verdict"] in ("keep", "review", "drop") for r in results)

    @pytest.mark.asyncio
    async def test_empty_source_marked_dead(self):
        """Sources with no articles are marked as dead."""
        from nexus.engine.sources.audit import audit_registry

        topic = TopicConfig(name="AI ML Research", subtopics=["llm"])
        sources = [{"id": "dead-src", "url": "https://dead.com/rss", "type": "rss"}]

        with patch("nexus.engine.sources.audit.poll_all_feeds",
                    return_value=[]):
            results = await audit_registry(
                llm=AsyncMock(),
                sources=sources,
                topic=topic,
            )

        assert len(results) == 1
        assert results[0]["verdict"] == "dead"
