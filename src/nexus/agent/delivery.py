"""Briefing delivery — send briefing text + audio via Telegram."""

import html
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# Telegram message limit
MAX_MESSAGE_LENGTH = 4096


def _md_to_telegram_html(text: str) -> str:
    """Convert markdown briefing to Telegram-friendly HTML.

    Telegram supports: <b>, <i>, <code>, <pre>, <a href="">.
    """
    # Escape HTML entities in the raw text first
    text = html.escape(text)

    # Headers: ## Title → bold
    text = re.sub(r'^#{1,3}\s+(.+)$', r'<b>\1</b>', text, flags=re.MULTILINE)

    # Bold: **text** or __text__
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'__(.+?)__', r'<b>\1</b>', text)

    # Italic: *text* or _text_ (but not inside bold tags)
    text = re.sub(r'(?<!\w)\*([^*]+?)\*(?!\w)', r'<i>\1</i>', text)
    text = re.sub(r'(?<!\w)_([^_]+?)_(?!\w)', r'<i>\1</i>', text)

    # Inline code: `text`
    text = re.sub(r'`([^`]+?)`', r'<code>\1</code>', text)

    # Links: [text](url) — unescape the URL
    def _fix_link(m):
        link_text = m.group(1)
        url = html.unescape(m.group(2))
        return f'<a href="{url}">{link_text}</a>'
    text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', _fix_link, text)

    # Bullet points: keep as-is (Telegram renders them fine as plain text)
    # Horizontal rules: --- → newline
    text = re.sub(r'^-{3,}$', '', text, flags=re.MULTILINE)

    return text.strip()


def split_message(text: str, max_length: int = MAX_MESSAGE_LENGTH) -> list[str]:
    """Split text into Telegram-sized chunks at natural breakpoints."""
    if len(text) <= max_length:
        return [text]

    chunks = []
    remaining = text

    while remaining:
        if len(remaining) <= max_length:
            chunks.append(remaining)
            break

        # Find a good split point: prefer double newline (paragraph break)
        split_at = remaining.rfind('\n\n', 0, max_length)
        if split_at == -1 or split_at < max_length // 2:
            # Fall back to single newline
            split_at = remaining.rfind('\n', 0, max_length)
        if split_at == -1 or split_at < max_length // 2:
            # Last resort: split at max_length
            split_at = max_length

        chunks.append(remaining[:split_at].rstrip())
        remaining = remaining[split_at:].lstrip('\n')

    return chunks


async def deliver_briefing(
    bot,
    chat_id: int,
    briefing_text: str,
    audio_path: Path | None = None,
) -> bool:
    """Send briefing to a Telegram chat as multiple messages if needed.

    Args:
        bot: telegram.Bot instance
        chat_id: target chat ID
        briefing_text: markdown briefing text
        audio_path: optional path to MP3 audio file

    Returns True if delivery succeeded.
    """
    try:
        # Convert markdown to Telegram HTML
        html_text = _md_to_telegram_html(briefing_text)

        # Split into chunks
        chunks = split_message(html_text)
        logger.info(f"Sending briefing in {len(chunks)} message(s)")

        for i, chunk in enumerate(chunks):
            try:
                await bot.send_message(
                    chat_id=chat_id,
                    text=chunk,
                    parse_mode="HTML",
                    disable_web_page_preview=True,
                )
            except Exception as e:
                # If HTML parsing fails, send as plain text
                logger.warning(f"HTML send failed for chunk {i}, sending as plain text: {e}")
                await bot.send_message(
                    chat_id=chat_id,
                    text=chunk,
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
        sig = alert['significance_score']
        headline = html.escape(alert['headline'])
        url = alert['source_url']
        text = (
            f"<b>BREAKING NEWS</b> (significance: {sig}/10)\n\n"
            f"{headline}\n\n"
            f'<a href="{url}">Source</a>'
        )
        await bot.send_message(
            chat_id=chat_id,
            text=text,
            parse_mode="HTML",
        )
        return True
    except Exception as e:
        logger.error(f"Breaking alert delivery failed: {e}")
        return False
