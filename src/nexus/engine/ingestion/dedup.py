"""URL normalization and deduplication for content items."""

import logging
from urllib.parse import urlparse, urlencode, parse_qs, urlunparse

from nexus.engine.sources.polling import ContentItem

logger = logging.getLogger(__name__)

TRACKING_PARAMS = {
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "traffic_source", "at_medium", "at_campaign", "ref", "source",
    "fbclid", "gclid", "mc_cid", "mc_eid",
}


def normalize_url(url: str) -> str:
    """Strip tracking params, fragments, and trailing slashes for dedup."""
    parsed = urlparse(url)
    params = parse_qs(parsed.query)
    cleaned = {k: v for k, v in params.items() if k not in TRACKING_PARAMS}
    clean_query = urlencode(cleaned, doseq=True)
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path.rstrip("/"), "", clean_query, ""))


def dedup_items(items: list[ContentItem]) -> list[ContentItem]:
    """Remove duplicate items by normalized URL. Keeps first occurrence."""
    seen: set[str] = set()
    result = []
    for item in items:
        key = normalize_url(item.url)
        if key not in seen:
            seen.add(key)
            result.append(item)
    if len(items) != len(result):
        logger.info(f"Dedup: {len(items)} → {len(result)} ({len(items) - len(result)} duplicates removed)")
    return result
