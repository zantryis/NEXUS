"""Tests for LLM-based emoji assignment."""

import json
import pytest
from unittest.mock import AsyncMock

from nexus.config.emoji import assign_topic_emojis
from nexus.config.models import TopicConfig


@pytest.fixture
def mock_llm():
    return AsyncMock()


async def test_assign_topic_emojis_calls_llm(mock_llm):
    """Topics without emojis trigger an LLM call."""
    mock_llm.complete.return_value = json.dumps({"AI Research": "\U0001f916"})
    topics = [TopicConfig(name="AI Research")]

    result = await assign_topic_emojis(mock_llm, topics)

    assert result == {"AI Research": "\U0001f916"}
    mock_llm.complete.assert_called_once()


async def test_assign_topic_emojis_skips_existing(mock_llm):
    """Topics with emojis already set don't trigger an LLM call."""
    topics = [TopicConfig(name="AI Research", emoji="\U0001f916")]

    result = await assign_topic_emojis(mock_llm, topics)

    assert result == {}
    mock_llm.complete.assert_not_called()


async def test_assign_topic_emojis_partial(mock_llm):
    """Only topics without emojis are sent to LLM."""
    mock_llm.complete.return_value = json.dumps({"Geopolitics": "\U0001f30d"})
    topics = [
        TopicConfig(name="AI Research", emoji="\U0001f916"),
        TopicConfig(name="Geopolitics"),
    ]

    result = await assign_topic_emojis(mock_llm, topics)

    assert result == {"Geopolitics": "\U0001f30d"}
    # Prompt should only contain Geopolitics
    call_args = mock_llm.complete.call_args
    assert "Geopolitics" in call_args[0][2]
    assert "AI Research" not in call_args[0][2]


async def test_assign_topic_emojis_llm_failure_graceful(mock_llm):
    """LLM failure returns empty dict, no exception raised."""
    mock_llm.complete.side_effect = RuntimeError("API down")
    topics = [TopicConfig(name="AI Research")]

    result = await assign_topic_emojis(mock_llm, topics)

    assert result == {}


async def test_assign_topic_emojis_bad_json_graceful(mock_llm):
    """Malformed LLM response returns empty dict."""
    mock_llm.complete.return_value = "not json"
    topics = [TopicConfig(name="AI Research")]

    result = await assign_topic_emojis(mock_llm, topics)

    assert result == {}


async def test_topic_config_emoji_field():
    """TopicConfig accepts and defaults emoji field."""
    t1 = TopicConfig(name="Test")
    assert t1.emoji is None

    t2 = TopicConfig(name="Test", emoji="\U0001f680")
    assert t2.emoji == "\U0001f680"
