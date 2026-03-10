"""Briefing delivery — send briefing text + audio via Telegram."""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Telegram message limit
MAX_MESSAGE_LENGTH = 4096


def truncate_briefing(text: str, max_length: int = MAX_MESSAGE_LENGTH) -> str:
    """Truncate briefing text to fit Telegram message limits."""
    if len(text) <= max_length:
        return text
    return text[:max_length - 50] + "\n\n... [truncated, see full briefing on dashboard]"


async def deliver_briefing(
    bot,
    chat_id: int,
    briefing_text: str,
    audio_path: Path | None = None,
) -> bool:
    """Send briefing to a Telegram chat.

    Args:
        bot: telegram.Bot instance
        chat_id: target chat ID
        briefing_text: markdown briefing text
        audio_path: optional path to MP3 audio file

    Returns True if delivery succeeded.
    """
    try:
        # Send text briefing
        text = truncate_briefing(briefing_text)
        await bot.send_message(
            chat_id=chat_id,
            text=text,
            parse_mode="Markdown",
        )

        # Send audio if available
        if audio_path and audio_path.exists():
            with open(audio_path, "rb") as f:
                await bot.send_audio(
                    chat_id=chat_id,
                    audio=f,
                    title=f"Nexus Briefing — {audio_path.stem}",
                    performer="Nexus Intelligence",
                )

        logger.info(f"Briefing delivered to chat {chat_id}")
        return True

    except Exception as e:
        logger.error(f"Briefing delivery failed: {e}")
        return False


async def deliver_breaking_alert(
    bot,
    chat_id: int,
    alert: dict,
) -> bool:
    """Send a breaking news alert to a Telegram chat."""
    try:
        text = (
            f"*BREAKING NEWS* (significance: {alert['significance_score']}/10)\n\n"
            f"{alert['headline']}\n\n"
            f"[Source]({alert['source_url']})"
        )
        await bot.send_message(
            chat_id=chat_id,
            text=text,
            parse_mode="Markdown",
        )
        return True
    except Exception as e:
        logger.error(f"Breaking alert delivery failed: {e}")
        return False
