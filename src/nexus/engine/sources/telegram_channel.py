"""Telegram public channel source adapter."""

import logging
import re

import httpx

from nexus.engine.sources.base import SourceAdapter
from nexus.engine.sources.polling import ContentItem

logger = logging.getLogger(__name__)


class TelegramChannelAdapter(SourceAdapter):
    """Scrape public Telegram channel preview pages for messages."""

    source_type = "telegram_channel"

    def __init__(self, timeout: float = 10.0):
        self._timeout = timeout

    async def poll(self, source_config: dict) -> list[ContentItem]:
        channel = source_config.get("channel", "").lstrip("@")
        if not channel:
            return []

        source_id = source_config.get("id", f"tg-{channel}")
        url = f"https://t.me/s/{channel}"

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(url)
                resp.raise_for_status()
        except (httpx.HTTPError, httpx.ConnectError) as e:
            logger.warning(f"Telegram channel fetch failed for @{channel}: {e}")
            return []

        return self._parse_messages(resp.text, source_id, source_config)

    def _parse_messages(
        self, html: str, source_id: str, config: dict
    ) -> list[ContentItem]:
        """Extract messages from Telegram's public channel preview HTML."""
        items = []
        # Telegram wraps messages in div.tgme_widget_message_wrap
        # Each message has class tgme_widget_message_text for the text content
        # and data-post="channel/msgid" attribute
        message_texts = re.findall(
            r'<div class="tgme_widget_message_text[^"]*"[^>]*>(.*?)</div>',
            html,
            re.DOTALL,
        )
        message_links = re.findall(
            r'data-post="([^"]+)"',
            html,
        )

        for i, text in enumerate(message_texts[:20]):  # Cap at 20 messages
            # Strip HTML tags for plain text
            clean = re.sub(r"<[^>]+>", "", text).strip()
            if not clean or len(clean) < 20:
                continue

            post_id = message_links[i] if i < len(message_links) else ""
            msg_url = f"https://t.me/{post_id}" if post_id else ""

            items.append(
                ContentItem(
                    title=clean[:120] + ("..." if len(clean) > 120 else ""),
                    url=msg_url,
                    source_id=source_id,
                    snippet=clean[:500],
                    source_language=config.get("language"),
                    source_affiliation=config.get("affiliation", "social"),
                    source_country=config.get("country"),
                    source_tier=config.get("tier", "C"),
                )
            )
        return items
