"""Tests for briefing delivery."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from pathlib import Path

from nexus.agent.delivery import (
    truncate_briefing, deliver_briefing, deliver_breaking_alert,
    MAX_MESSAGE_LENGTH,
)


def test_truncate_short():
    text = "Short briefing."
    assert truncate_briefing(text) == text


def test_truncate_long():
    text = "x" * 5000
    result = truncate_briefing(text)
    assert len(result) <= MAX_MESSAGE_LENGTH
    assert "truncated" in result


async def test_deliver_briefing_text_only():
    bot = AsyncMock()
    result = await deliver_briefing(bot, 123, "Today's briefing text.")
    assert result is True
    bot.send_message.assert_called_once()


async def test_deliver_briefing_with_audio(tmp_path):
    audio = tmp_path / "test.mp3"
    audio.write_bytes(b"fake-audio")

    bot = AsyncMock()
    result = await deliver_briefing(bot, 123, "Text briefing.", audio)
    assert result is True
    bot.send_message.assert_called_once()
    bot.send_audio.assert_called_once()


async def test_deliver_briefing_failure():
    bot = AsyncMock()
    bot.send_message = AsyncMock(side_effect=Exception("Network error"))
    result = await deliver_briefing(bot, 123, "Text")
    assert result is False


async def test_deliver_breaking_alert():
    bot = AsyncMock()
    alert = {
        "headline": "Crisis erupts",
        "source_url": "https://reuters.com/1",
        "significance_score": 9,
    }
    result = await deliver_breaking_alert(bot, 123, alert)
    assert result is True
    call_text = bot.send_message.call_args.kwargs["text"]
    assert "BREAKING" in call_text
    assert "Crisis erupts" in call_text
