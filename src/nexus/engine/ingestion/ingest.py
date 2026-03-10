"""Content ingestion — fetch full article text from URLs."""

import asyncio
import json
import logging
from collections import defaultdict
from urllib.parse import urlparse

import trafilatura
from nexus.engine.sources.polling import ContentItem

logger = logging.getLogger(__name__)

# Concurrency limits
_GLOBAL_SEMAPHORE = asyncio.Semaphore(10)
_domain_semaphores: dict[str, asyncio.Semaphore] = defaultdict(lambda: asyncio.Semaphore(2))

# Strings that suggest paywall/subscription gate
_PAYWALL_MARKERS = [
    "subscribe to continue",
    "subscription required",
    "sign in to read",
    "premium content",
    "paywall",
    "already a subscriber",
    "create a free account",
    "this article is for subscribers",
    "to read the full story",
    "register to continue",
]


def _get_domain(url: str) -> str:
    return urlparse(url).netloc


def _detect_paywall(html: str) -> bool:
    """Heuristic paywall detection from raw HTML."""
    html_lower = html.lower()
    return any(marker in html_lower for marker in _PAYWALL_MARKERS)


def ingest_item(item: ContentItem) -> ContentItem:
    """Fetch and extract full text for a single content item (sync)."""
    html = trafilatura.fetch_url(item.url)
    if not html:
        logger.warning(f"Failed to fetch: {item.url}")
        item.extraction_status = "fetch_failed"
        item.extraction_error = "No HTML returned"
        return item

    # Check for paywall before extraction
    if _detect_paywall(html):
        item.extraction_status = "paywall"
        logger.info(f"Paywall detected: [{item.source_id}] {item.url}")
        # Still try to extract — RSS snippet + partial text may be useful

    # Extract with JSON output for language detection
    result_json = trafilatura.extract(html, output_format="json")
    if result_json:
        try:
            data = json.loads(result_json)
            item.full_text = data.get("text", "")
            detected_lang = data.get("language")
            if detected_lang:
                item.detected_language = detected_lang
        except (json.JSONDecodeError, TypeError):
            # Fallback to plain extraction
            item.full_text = trafilatura.extract(html) or ""
    else:
        item.full_text = ""

    if item.full_text:
        if item.extraction_status != "paywall":
            item.extraction_status = "ok"
    else:
        if item.extraction_status != "paywall":
            item.extraction_status = "extract_failed"
            item.extraction_error = "No text extracted"
        logger.warning(f"Failed to extract text: {item.url}")

    # Fall back to source language if no detection
    if not item.detected_language and item.source_language:
        item.detected_language = item.source_language

    return item


async def async_ingest_item(item: ContentItem) -> ContentItem:
    """Async wrapper: runs sync ingest_item in a thread with rate limiting."""
    domain = _get_domain(item.url)
    async with _GLOBAL_SEMAPHORE:
        async with _domain_semaphores[domain]:
            return await asyncio.to_thread(ingest_item, item)


async def async_ingest_items(items: list[ContentItem]) -> list[ContentItem]:
    """Ingest all items concurrently. Returns items with extracted text."""
    if not items:
        return []
    tasks = [async_ingest_item(item) for item in items]
    ingested = await asyncio.gather(*tasks)
    # Return items that have text OR are paywalled (RSS snippet may still be useful)
    return [item for item in ingested if item.full_text or item.extraction_status == "paywall"]


def ingest_items(items: list[ContentItem]) -> list[ContentItem]:
    """Sync ingestion fallback. Returns only items with extracted text."""
    results = []
    for item in items:
        ingested = ingest_item(item)
        if ingested.full_text:
            results.append(ingested)
    return results
