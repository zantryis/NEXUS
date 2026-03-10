"""Tests for briefing synthesis."""

import pytest
from datetime import date
from unittest.mock import AsyncMock
from nexus.config.models import NexusConfig, UserConfig, TopicConfig, BriefingConfig
from nexus.engine.knowledge.events import Event
from nexus.engine.knowledge.compression import Summary
from nexus.engine.sources.polling import ContentItem
from nexus.engine.synthesis.briefing import (
    build_context,
    generate_briefing,
    TopicContext,
)


@pytest.fixture
def config():
    return NexusConfig(
        user=UserConfig(name="Tristan", timezone="America/Denver", output_language="en"),
        briefing=BriefingConfig(style="analytical", depth="detailed"),
        topics=[
            TopicConfig(name="AI Research", priority="high", subtopics=["agents"]),
        ],
    )


@pytest.fixture
def topic_context():
    return TopicContext(
        topic=TopicConfig(name="AI Research", priority="high", subtopics=["agents"]),
        monthly_summaries=[],
        weekly_summaries=[
            Summary(
                period_start=date(2026, 3, 2),
                period_end=date(2026, 3, 6),
                text="Several new agent frameworks were announced.",
                event_count=3,
            ),
        ],
        recent_events=[
            Event(
                date=date(2026, 3, 9),
                summary="OpenAI releases new agent toolkit",
                sources=[{"url": "https://example.com", "language": "en", "outlet": "Reuters"}],
                entities=["OpenAI"],
                significance=8,
            ),
        ],
        top_articles=[
            ContentItem(
                title="OpenAI Agent Toolkit",
                url="https://example.com",
                source_id="reuters",
                full_text="OpenAI announced a new toolkit for building agents...",
            ),
        ],
    )


def test_build_context(topic_context):
    context = build_context([topic_context])
    assert "AI Research" in context
    assert "OpenAI" in context
    assert "agent frameworks" in context


@pytest.mark.asyncio
async def test_generate_briefing(config, topic_context):
    mock_llm = AsyncMock()
    mock_llm.complete.return_value = (
        "# Daily Briefing — 2026-03-09\n\n"
        "## AI Research\n\n"
        "OpenAI released a new agent toolkit (Reuters)..."
    )

    briefing = await generate_briefing(mock_llm, config, [topic_context])
    assert "AI Research" in briefing
    assert "OpenAI" in briefing
    mock_llm.complete.assert_called_once()

    # Verify the prompt includes key requirements
    call_args = mock_llm.complete.call_args
    assert call_args.kwargs["config_key"] == "synthesis"
    assert "en" in call_args.kwargs["system_prompt"]  # output language
