"""Tests for editorial style injection in briefing and dialogue renderers."""

import pytest
from unittest.mock import AsyncMock, patch

from nexus.config.models import NexusConfig, BriefingConfig, UserConfig
from nexus.engine.synthesis.knowledge import TopicSynthesis
from nexus.engine.synthesis.renderers import (
    EDITORIAL_STANCE, BRIEFING_SYSTEM_PROMPT, render_text_briefing,
)
from nexus.engine.audio.script import (
    EDITORIAL_DIALOGUE_ADDENDUM, DIALOGUE_SYSTEM_PROMPT, generate_dialogue_script,
)


def _make_config(style: str = "analytical") -> NexusConfig:
    """Create a minimal NexusConfig with given briefing style."""
    return NexusConfig(
        preset="balanced",
        user=UserConfig(name="Test", timezone="UTC", output_language="en"),
        briefing=BriefingConfig(style=style),
        topics=[],
    )


def _make_synthesis() -> list[TopicSynthesis]:
    return [TopicSynthesis(
        topic_name="Test Topic",
        topic_slug="test-topic",
        threads=[],
        background=[],
        source_balance={"private": 3, "state": 2},
        languages_represented=["en"],
    )]


async def test_editorial_stance_injected():
    """Verify system prompt includes editorial stance when style='editorial'."""
    config = _make_config("editorial")
    syntheses = _make_synthesis()

    captured_prompt = {}

    async def mock_complete(config_key, system_prompt, user_prompt):
        captured_prompt["system"] = system_prompt
        return "# Briefing\n\nTest briefing text."

    mock_llm = AsyncMock()
    mock_llm.complete = mock_complete

    await render_text_briefing(mock_llm, config, syntheses)

    assert "international law" in captured_prompt["system"].lower()
    assert "EDITORIAL STANCE" in captured_prompt["system"]
    assert "Geneva Conventions" in captured_prompt["system"]


async def test_non_editorial_no_stance():
    """Verify 'analytical' style does NOT include editorial stance."""
    config = _make_config("analytical")
    syntheses = _make_synthesis()

    captured_prompt = {}

    async def mock_complete(config_key, system_prompt, user_prompt):
        captured_prompt["system"] = system_prompt
        return "# Briefing\n\nTest briefing text."

    mock_llm = AsyncMock()
    mock_llm.complete = mock_complete

    await render_text_briefing(mock_llm, config, syntheses)

    assert "EDITORIAL STANCE" not in captured_prompt["system"]
    assert "Geneva Conventions" not in captured_prompt["system"]


async def test_conversational_no_stance():
    """Verify 'conversational' style does NOT include editorial stance."""
    config = _make_config("conversational")
    syntheses = _make_synthesis()

    captured_prompt = {}

    async def mock_complete(config_key, system_prompt, user_prompt):
        captured_prompt["system"] = system_prompt
        return "# Briefing\n\nTest briefing text."

    mock_llm = AsyncMock()
    mock_llm.complete = mock_complete

    await render_text_briefing(mock_llm, config, syntheses)

    assert "EDITORIAL STANCE" not in captured_prompt["system"]


async def test_editorial_dialogue_addendum():
    """Verify audio script gets editorial addendum when style='editorial'."""
    config = _make_config("editorial")
    syntheses = _make_synthesis()

    captured_prompt = {}

    async def mock_complete(config_key, system_prompt, user_prompt, **kwargs):
        captured_prompt["system"] = system_prompt
        return '{"turns": [{"speaker": "A", "text": "Hello"}]}'

    mock_llm = AsyncMock()
    mock_llm.complete = mock_complete

    await generate_dialogue_script(mock_llm, config, syntheses)

    assert "EDITORIAL VOICE" in captured_prompt["system"]
    assert "international law" in captured_prompt["system"].lower()


async def test_non_editorial_dialogue_no_addendum():
    """Verify audio script does NOT get editorial addendum for other styles."""
    config = _make_config("analytical")
    syntheses = _make_synthesis()

    captured_prompt = {}

    async def mock_complete(config_key, system_prompt, user_prompt, **kwargs):
        captured_prompt["system"] = system_prompt
        return '{"turns": [{"speaker": "A", "text": "Hello"}]}'

    mock_llm = AsyncMock()
    mock_llm.complete = mock_complete

    await generate_dialogue_script(mock_llm, config, syntheses)

    assert "EDITORIAL VOICE" not in captured_prompt["system"]
