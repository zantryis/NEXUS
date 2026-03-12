"""Tests for briefing delivery."""

import pytest
from datetime import date
from unittest.mock import AsyncMock, MagicMock
from pathlib import Path

from nexus.agent.delivery import (
    split_message, _md_to_telegram_html, _get_topic_emoji,
    deliver_briefing, deliver_breaking_alert, deliver_breaking_digest,
    format_breaking_digest, md_to_telegram_html_light,
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
    result = _md_to_telegram_html(text, report_date=date(2026, 3, 10))
    assert "<b>IRAN-US RELATIONS</b>" in result
    assert "Some content" in result


def test_md_to_html_double_headers():
    """The ## ## bug from LLM output should be cleaned up."""
    text = "## ## Double Header\nContent"
    result = _md_to_telegram_html(text, report_date=date(2026, 3, 10))
    assert "<b>DOUBLE HEADER</b>" in result
    # Should not have leftover ## in output
    assert "## ##" not in result


def test_md_to_html_sub_headers():
    text = "### Operation Details\nContent"
    result = _md_to_telegram_html(text, report_date=date(2026, 3, 10))
    assert "<b>" in result
    assert "Operation Details" in result


def test_md_to_html_bold():
    result = _md_to_telegram_html("**bold text**", report_date=date(2026, 3, 10))
    assert "<b>bold text</b>" in result


def test_md_to_html_escapes_html():
    result = _md_to_telegram_html("1 < 2 & 3 > 1", report_date=date(2026, 3, 10))
    assert "&lt;" in result
    assert "&amp;" in result


def test_md_to_html_bullets():
    result = _md_to_telegram_html("*   Item one\n*   Item two", report_date=date(2026, 3, 10))
    assert "\u2022 Item one" in result
    assert "\u2022 Item two" in result


def test_md_to_html_newsletter_header():
    result = _md_to_telegram_html("Hello", report_date=date(2026, 3, 10))
    assert "NEXUS DAILY BRIEFING" in result
    assert "March 10, 2026" in result


def test_md_to_html_section_separators():
    text = "## AI Research\nContent here"
    result = _md_to_telegram_html(text, report_date=date(2026, 3, 10))
    assert "\u2501" in result  # horizontal bar separator


def test_get_topic_emoji():
    assert _get_topic_emoji("Iran-US Relations") == "\U0001f30d"
    assert _get_topic_emoji("AI/ML Research") == "\U0001f916"
    assert _get_topic_emoji("Formula 1") == "\U0001f3ce\ufe0f"
    assert _get_topic_emoji("Global Energy Transition") == "\u26a1"
    assert _get_topic_emoji("Unknown Topic") == "\U0001f4cc"


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
    # Check page indicators present
    last_call = bot.send_message.call_args_list[-1]
    assert ")" in last_call.kwargs["text"]  # page number indicator


async def test_deliver_briefing_with_audio(tmp_path):
    audio = tmp_path / "test.mp3"
    audio.write_bytes(b"fake-audio")

    bot = AsyncMock()
    result = await deliver_briefing(bot, 123, "Text briefing.", audio)
    assert result is True
    bot.send_message.assert_called_once()
    bot.send_audio.assert_called_once()
    # Check podcast branding
    audio_kwargs = bot.send_audio.call_args.kwargs
    assert "Nexus Report" in audio_kwargs["title"]
    assert "Nova" in audio_kwargs["performer"]


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


# ── Lightweight Q&A HTML formatter tests ──


def test_light_html_bold():
    result = md_to_telegram_html_light("**important** point")
    assert "<b>important</b>" in result


def test_light_html_italic():
    result = md_to_telegram_html_light("*emphasis* here")
    assert "<i>emphasis</i>" in result


def test_light_html_bullets():
    result = md_to_telegram_html_light("- Item one\n- Item two")
    assert "\u2022 Item one" in result
    assert "\u2022 Item two" in result


def test_light_html_star_bullets():
    result = md_to_telegram_html_light("* First\n* Second")
    assert "\u2022 First" in result


def test_light_html_headers():
    result = md_to_telegram_html_light("## Section\nContent")
    assert "<b>Section</b>" in result
    # Should NOT have newsletter treatment (no separator, no emoji, no uppercase)
    assert "\u2501" not in result
    assert "SECTION" not in result


def test_light_html_escapes():
    result = md_to_telegram_html_light("A < B & C > D")
    assert "&lt;" in result
    assert "&amp;" in result
    assert "&gt;" in result


def test_light_html_code():
    result = md_to_telegram_html_light("Use `config.yaml` here")
    assert "<code>config.yaml</code>" in result


def test_light_html_no_newsletter_header():
    """Light formatter should NOT add newsletter header."""
    result = md_to_telegram_html_light("Just a response")
    assert "NEXUS DAILY BRIEFING" not in result


# ── Breaking news digest tests ──


async def test_digest_empty():
    bot = AsyncMock()
    result = await deliver_breaking_digest(bot, 123, {})
    assert result is False
    bot.send_message.assert_not_called()


async def test_digest_topic_grouped():
    bot = AsyncMock()
    alerts_by_topic = {
        "iran-us-relations": [
            {"headline": "Iran alert", "source_url": "https://a.com", "significance_score": 9},
        ],
        "ai-ml-research": [
            {"headline": "AI alert", "source_url": "https://b.com", "significance_score": 7},
        ],
    }
    result = await deliver_breaking_digest(bot, 123, alerts_by_topic)
    assert result is True
    bot.send_message.assert_called()
    text = bot.send_message.call_args.kwargs["text"]
    assert "DIGEST" in text
    # Topics should appear — most significant first
    assert "IRAN" in text
    assert "AI" in text
    assert text.index("IRAN") < text.index("AI")


async def test_digest_single_topic():
    bot = AsyncMock()
    alerts_by_topic = {
        "test-topic": [
            {"headline": "Solo alert", "source_url": "https://x.com", "significance_score": 8},
        ],
    }
    result = await deliver_breaking_digest(bot, 123, alerts_by_topic)
    assert result is True
    text = bot.send_message.call_args.kwargs["text"]
    assert "Solo alert" in text
    assert "[8/10]" in text
