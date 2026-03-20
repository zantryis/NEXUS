"""Briefing delivery — send briefing text + audio via Telegram."""

import html
import logging
import re
from datetime import date
from pathlib import Path

logger = logging.getLogger(__name__)

# Telegram message limit
MAX_MESSAGE_LENGTH = 4096

# Topic → emoji mapping for newsletter section headers
_TOPIC_EMOJIS = {
    "executive summary": "\U0001f4cb",
    "iran": "\U0001f30d",
    "energy": "\u26a1",
    "ai": "\U0001f916",
    "formula": "\U0001f3ce\ufe0f",
    "climate": "\U0001f331",
    "tech": "\U0001f4bb",
    "economy": "\U0001f4c8",
    "defense": "\U0001f6e1\ufe0f",
    "politics": "\U0001f3db\ufe0f",
    "china": "\U0001f1e8\U0001f1f3",
    "russia": "\U0001f1f7\U0001f1fa",
    "space": "\U0001f680",
}


def _get_topic_emoji(header: str, config_emoji: str | None = None) -> str:
    """Match a topic header to an emoji."""
    if config_emoji:
        return config_emoji
    header_lower = header.lower()
    for keyword, emoji in _TOPIC_EMOJIS.items():
        if keyword in header_lower:
            return emoji
    return "\U0001f4cc"


def _md_to_telegram_html(text: str, report_date: date | None = None) -> str:
    """Convert markdown briefing to Telegram newsletter HTML.

    Produces a clean, readable newsletter format with visual section
    separators, topic emojis, proper bullet points, and bold/italic/links.
    """
    # Escape HTML entities first
    text = html.escape(text)

    # Fix double headers: ## ## Topic → ## Topic (require space between hash groups)
    text = re.sub(r'^(#{1,3})\s+#{1,3}\s+', r'\1 ', text, flags=re.MULTILINE)

    # Convert markdown bullet points (* item) to bullet character
    text = re.sub(r'^\*\s{1,4}', '\u2022 ', text, flags=re.MULTILINE)

    # Sub-headers (### ...) → bold with arrow, no separator
    def _format_sub(m):
        return f"\n<b>\u25b8 {m.group(1)}</b>"
    text = re.sub(r'^###\s+(.+)$', _format_sub, text, flags=re.MULTILINE)

    # Main section headers (## ...) → emoji + separator + UPPERCASE
    def _format_section(m):
        header_text = m.group(1).strip()
        emoji = _get_topic_emoji(header_text)
        return f"\n\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\n{emoji}  <b>{header_text.upper()}</b>\n"
    text = re.sub(r'^##\s+(.+)$', _format_section, text, flags=re.MULTILINE)

    # Bold: **text** or __text__
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'__(.+?)__', r'<b>\1</b>', text)

    # Italic: *text* or _text_ (not inside bold tags)
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

    # Remove horizontal rules (---)
    text = re.sub(r'^-{3,}$', '', text, flags=re.MULTILINE)

    # Clean up excessive blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Add newsletter header
    if report_date is None:
        report_date = date.today()
    date_str = report_date.strftime("%B %d, %Y")
    header = f"\U0001f4f0  <b>NEXUS DAILY BRIEFING</b>\n<i>{date_str}</i>\n"

    text = header + "\n" + text.strip()
    return text


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
        # Convert markdown to Telegram HTML newsletter
        html_text = _md_to_telegram_html(briefing_text)

        # Split into chunks
        chunks = split_message(html_text)
        total = len(chunks)
        logger.info(f"Sending briefing in {total} message(s)")

        for i, chunk in enumerate(chunks):
            # Add page indicator for multi-part briefings
            if total > 1:
                chunk = f"{chunk}\n\n<i>({i + 1}/{total})</i>"
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
                    title=f"The Nexus Report \u2014 {audio_path.stem}",
                    performer="Nova & Atlas",
                )

        logger.info(f"Briefing delivered to chat {chat_id}")
        return True

    except Exception as e:
        logger.error(f"Briefing delivery failed: {e}")
        return False


def md_to_telegram_html_light(text: str) -> str:
    """Lightweight markdown → Telegram HTML for Q&A responses.

    Handles bold, italic, bullets, inline code, links. No newsletter
    headers, section separators, or topic emojis.
    """
    text = html.escape(text)

    # Markdown bullet points → bullet character (- item or * item)
    text = re.sub(r'^[-*]\s+', '\u2022 ', text, flags=re.MULTILINE)

    # Headers → bold (any level)
    text = re.sub(r'^#{1,3}\s+(.+)$', r'<b>\1</b>', text, flags=re.MULTILINE)

    # Bold: **text** or __text__
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'__(.+?)__', r'<b>\1</b>', text)

    # Italic: *text* or _text_
    text = re.sub(r'(?<!\w)\*([^*]+?)\*(?!\w)', r'<i>\1</i>', text)
    text = re.sub(r'(?<!\w)_([^_]+?)_(?!\w)', r'<i>\1</i>', text)

    # Inline code: `text`
    text = re.sub(r'`([^`]+?)`', r'<code>\1</code>', text)

    # Links: [text](url)
    def _fix_link(m):
        link_text = m.group(1)
        url = html.unescape(m.group(2))
        return f'<a href="{url}">{link_text}</a>'
    text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', _fix_link, text)

    # Horizontal rules
    text = re.sub(r'^-{3,}$', '', text, flags=re.MULTILINE)

    # Clean up excessive blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


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


def format_breaking_digest(alerts_by_topic: dict[str, list[dict]]) -> str:
    """Format topic-grouped breaking news alerts into HTML text.

    Topics sorted by max significance (most urgent first).
    Returns empty string if no alerts.
    """
    if not alerts_by_topic:
        return ""

    topic_order = sorted(
        alerts_by_topic.keys(),
        key=lambda slug: max(
            a.get("significance_score", 0) for a in alerts_by_topic[slug]
        ),
        reverse=True,
    )

    lines = ["\U0001f6a8 <b>BREAKING NEWS DIGEST</b>\n"]

    for slug in topic_order:
        alerts = sorted(
            alerts_by_topic[slug],
            key=lambda a: a.get("significance_score", 0),
            reverse=True,
        )
        emoji = _get_topic_emoji(slug)
        topic_name = slug.replace("-", " ").upper()
        lines.append(f"\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501")
        lines.append(f"{emoji}  <b>{topic_name}</b>\n")

        for i, alert in enumerate(alerts, 1):
            sig = alert.get("significance_score", "?")
            headline = html.escape(alert.get("headline", ""))
            url = alert.get("source_url", "")
            lines.append(f"<b>{i}.</b> [{sig}/10] {headline}")
            if url:
                lines.append(f'   \U0001f517 <a href="{url}">Source</a>')
        lines.append("")

    return "\n".join(lines).strip()


async def deliver_breaking_digest(
    bot,
    chat_id: int,
    alerts_by_topic: dict[str, list[dict]],
) -> bool:
    """Send topic-grouped breaking news digest as a single consolidated message."""
    text = format_breaking_digest(alerts_by_topic)
    if not text:
        return False

    try:
        chunks = split_message(text)

        for chunk in chunks:
            try:
                await bot.send_message(
                    chat_id=chat_id,
                    text=chunk,
                    parse_mode="HTML",
                    disable_web_page_preview=True,
                )
            except Exception:
                await bot.send_message(chat_id=chat_id, text=chunk)

        total = sum(len(a) for a in alerts_by_topic.values())
        logger.info(f"Breaking digest delivered: {total} alerts across {len(alerts_by_topic)} topics to chat {chat_id}")
        return True

    except Exception as e:
        logger.error(f"Breaking digest delivery failed: {e}")
        return False
