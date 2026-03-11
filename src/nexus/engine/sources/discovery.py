"""Source auto-discovery — find RSS feeds for new topics via LLM + web search."""

import asyncio
import json
import logging
from hashlib import sha256

import feedparser

from nexus.llm.client import LLMClient
from nexus.agent.websearch import web_search

logger = logging.getLogger(__name__)

QUERY_SYSTEM_PROMPT = (
    "Generate search queries to find RSS news feeds for a given topic. "
    "Output a JSON array of 3-5 search queries that would help find "
    "RSS feeds, news sources, and blogs covering this topic.\n\n"
    "Include queries like:\n"
    '- "{topic} news RSS feed"\n'
    '- "{topic} blog RSS"\n'
    '- "best {topic} news sources RSS"\n'
    "Make queries specific and varied to maximize feed discovery.\n"
    "Output JSON array only, no explanation."
)


async def _generate_search_queries(
    llm: LLMClient,
    topic_name: str,
    subtopics: list[str] | None = None,
) -> list[str]:
    """Use LLM to generate search queries for finding RSS feeds."""
    context = f"Topic: {topic_name}"
    if subtopics:
        context += f"\nSubtopics: {', '.join(subtopics)}"

    try:
        raw = await llm.complete(
            config_key="discovery",
            system_prompt=QUERY_SYSTEM_PROMPT,
            user_prompt=context,
            json_response=True,
        )
        queries = json.loads(raw)
        if isinstance(queries, list):
            return [q for q in queries if isinstance(q, str)]
    except Exception as e:
        logger.warning(f"Query generation failed: {e}")

    # Fallback: simple queries
    return [
        f"{topic_name} news RSS feed",
        f"{topic_name} blog RSS",
    ]


async def _find_rss_feeds(query: str) -> list[str]:
    """Search the web for RSS feed URLs matching a query."""
    results = await web_search(query, max_results=8)
    urls: list[str] = []
    for r in results:
        url = r.get("url", "")
        # Collect all URLs — we'll validate them as feeds later
        if url:
            urls.append(url)
        # Also look for RSS URLs mentioned in snippets
        snippet = r.get("snippet", "")
        for word in snippet.split():
            if word.startswith("http") and ("rss" in word.lower() or "feed" in word.lower() or "atom" in word.lower()):
                clean = word.rstrip(".,;)")
                urls.append(clean)
    return list(set(urls))


async def _validate_feed(url: str) -> dict | None:
    """Check if a URL is a valid RSS/Atom feed with entries."""
    try:
        loop = asyncio.get_event_loop()
        parsed = await loop.run_in_executor(None, feedparser.parse, url)

        if parsed.bozo and not parsed.entries:
            return None
        if len(parsed.entries) < 1:
            return None

        title = getattr(parsed.feed, "title", "") or ""
        language = getattr(parsed.feed, "language", "en") or "en"
        # Normalize language code
        language = language.split("-")[0].lower()

        # Generate a stable ID from URL
        url_hash = sha256(url.encode()).hexdigest()[:8]
        slug = title.lower().replace(" ", "-")[:20] if title else url_hash
        feed_id = f"discovered-{slug}-{url_hash}"

        return {
            "id": feed_id,
            "url": url,
            "name": title,
            "type": "rss",
            "language": language,
            "affiliation": "unknown",
            "country": "unknown",
            "tier": "B",
        }
    except Exception as e:
        logger.debug(f"Feed validation failed for {url}: {e}")
        return None


def _deduplicate(feeds: list[dict]) -> list[dict]:
    """Remove duplicate feeds by URL."""
    seen: set[str] = set()
    unique: list[dict] = []
    for f in feeds:
        if f["url"] not in seen:
            seen.add(f["url"])
            unique.append(f)
    return unique


async def discover_sources(
    llm: LLMClient,
    topic_name: str,
    subtopics: list[str] | None = None,
    existing_urls: set[str] | None = None,
    max_feeds: int = 15,
) -> list[dict]:
    """Discover RSS feeds for a topic using LLM + web search + validation.

    Returns list of validated source dicts ready to add to a registry.
    """
    existing_urls = existing_urls or set()

    # Step 1: generate search queries
    queries = await _generate_search_queries(llm, topic_name, subtopics)
    logger.info(f"Discovery queries for '{topic_name}': {queries}")

    # Step 2: search for feeds
    all_urls: list[str] = []
    for query in queries:
        urls = await _find_rss_feeds(query)
        all_urls.extend(urls)

    # Deduplicate and filter existing
    unique_urls = list(set(all_urls) - existing_urls)
    logger.info(f"Found {len(unique_urls)} candidate URLs for '{topic_name}'")

    # Step 3: validate feeds in parallel (with concurrency limit)
    sem = asyncio.Semaphore(5)

    async def _validate_with_limit(url: str):
        async with sem:
            return await _validate_feed(url)

    tasks = [_validate_with_limit(url) for url in unique_urls[:30]]  # cap candidates
    results = await asyncio.gather(*tasks)

    # Filter valid feeds
    valid = [r for r in results if r is not None]
    valid = _deduplicate(valid)

    logger.info(f"Validated {len(valid)} feeds for '{topic_name}'")
    return valid[:max_feeds]
