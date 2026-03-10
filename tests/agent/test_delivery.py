"""Tests for briefing delivery."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from pathlib import Path

from nexus.agent.delivery import (
    split_message, _md_to_telegram_html,
    deliver_briefing, deliver_breaking_alert,
    MAX_MESSAGE_LENGTH,
)


def test_split_short():
    text = "Short briefing."
    chunks = split_message(text)
    assert chunks == ["Short briefing."]


def test_split_long():
    text = "x" * 5000
    chunks = split_message(text)
    assert all(len(c) <= MAX_MESSAGE_LENGTH for c in chunks)
    assert len(chunks) >= 2


def test_split_at_paragraph():
    text = ("x" * 2500) + "\n\n" + ("y" * 2500)
    chunks = split_message(text)
    assert len(chunks) == 2
    assert chunks[0] == "x" * 2500


def test_md_to_html_headers():
    text = "## Iran-US Relations\nSome content"
    result = _md_to_telegram_html(text)
    assert "<b>Iran-US Relations</b>" in result
    assert "Some content" in result


def test_md_to_html_bold():
    result = _md_to_telegram_html("**bold text**")
    assert "<b>bold text</b>" in result


def test_md_to_html_escapes_html():
    result = _md_to_telegram_html("1 < 2 & 3 > 1")
    assert "&lt;" in result
    assert "&amp;" in result


async def test_deliver_briefing_text_only():
    bot = AsyncMock()
    result = await deliver_briefing(bot, 123, "## Today\nBriefing text.")
    assert result is True
    bot.send_message.assert_called_once()
    call_kwargs = bot.send_message.call_args.kwargs
    assert call_kwargs["parse_mode"] == "HTML"


async def test_deliver_briefing_multi_message():
    bot = AsyncMock()
    long_text = ("## Topic\n\n" + "A" * 4000 + "\n\n") * 3
    result = await deliver_briefing(bot, 123, long_text)
    assert result is True
    assert bot.send_message.call_count > 1


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
    call_kwargs = bot.send_message.call_args.kwargs
    assert "BREAKING" in call_kwargs["text"]
    assert "Crisis erupts" in call_kwargs["text"]
    assert call_kwargs["parse_mode"] == "HTML"
