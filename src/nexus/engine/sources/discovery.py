"""Source auto-discovery — find RSS feeds for new topics via LLM + web search.

Enhanced flow:
  1. Match from global registry (curated, rich metadata) — primary path
  2. Google News RSS (free, always valid)
  3. Web search discovery (existing DuckDuckGo path)
  4. LLM metadata classification for unknown feeds
  5. Diversity scoring
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from urllib.parse import quote_plus

import yaml
import feedparser

from nexus.llm.client import LLMClient
from nexus.agent.websearch import web_search
from nexus.engine.sources.diversity import DiversityMetrics, compute_diversity

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

REGISTRY_MATCH_PROMPT = (
    "You are a source relevance scorer. Given a list of RSS news sources and a topic, "
    "score how relevant each source is to the topic on a scale of 1-10.\n\n"
    "Consider:\n"
    "- Source name and tags\n"
    "- Whether the source's coverage area overlaps with the topic\n"
    "- Quality tier (A=best, B=good, C=niche)\n\n"
    "Respond with a JSON array: [{\"id\": \"<source_id>\", \"score\": <int>, \"reason\": \"<brief>\"}]\n"
    "Only include sources scoring 5 or above."
)

CLASSIFY_PROMPT = (
    "You classify news source metadata. Given a list of RSS feeds with their names and "
    "sample entry titles, determine for each:\n"
    "- affiliation: one of 'public', 'private', 'state', 'nonprofit', 'academic'\n"
    "- country: ISO 3166-1 alpha-2 code (e.g., 'US', 'GB', 'DE')\n"
    "- tier: 'A' (major outlet), 'B' (established), 'C' (niche/blog)\n\n"
    "Respond with JSON array: [{\"id\": \"<feed_id>\", \"affiliation\": \"...\", "
    "\"country\": \"...\", \"tier\": \"...\"}]\n"
    "If uncertain about a field, use your best guess based on the name and content."
)


@dataclass
class DiscoveryResult:
    """Result of source discovery for a topic."""
    feeds: list[dict] = field(default_factory=list)
    diversity: DiversityMetrics = field(default_factory=DiversityMetrics)
    sources_from_registry: int = 0
    sources_from_web: int = 0
    sources_from_google_news: int = 0


# ── Step 1: Global registry matching ──────────────────────────────────


def _load_global_registry(data_dir: Path) -> list[dict]:
    """Load curated sources from global_registry.yaml."""
    path = data_dir / "sources" / "global_registry.yaml"
    if not path.exists():
        return []
    raw = yaml.safe_load(path.read_text())
    if not raw or "sources" not in raw:
        return []
    return raw["sources"]


async def _match_from_global_registry(
    llm: LLMClient,
    topic_name: str,
    subtopics: list[str] | None,
    global_sources: list[dict],
    existing_urls: set[str],
    max_matches: int = 20,
) -> list[dict]:
    """Use LLM to score global registry sources against the topic."""
    candidates = [s for s in global_sources if s.get("url") not in existing_urls]
    if not candidates:
        return []

    # Format sources for LLM
    source_lines = []
    for s in candidates:
        tags = ", ".join(s.get("tags", []))
        source_lines.append(
            f"- id: {s['id']}, name: {s.get('name', '?')}, "
            f"tags: [{tags}], tier: {s.get('tier', '?')}, "
            f"country: {s.get('country', '?')}"
        )

    context = f"Topic: {topic_name}"
    if subtopics:
        context += f"\nSubtopics: {', '.join(subtopics)}"
    context += f"\n\nSources:\n" + "\n".join(source_lines)

    try:
        raw = await llm.complete(
            config_key="discovery",
            system_prompt=REGISTRY_MATCH_PROMPT,
            user_prompt=context,
            json_response=True,
        )
        scores = json.loads(raw)
        if not isinstance(scores, list):
            scores = [scores]

        # Map scores back to sources
        score_map = {entry["id"]: entry["score"] for entry in scores if entry.get("score", 0) >= 5}
        matched = []
        for s in candidates:
            if s["id"] in score_map:
                s_copy = dict(s)
                s_copy["_relevance"] = score_map[s["id"]]
                matched.append(s_copy)

        matched.sort(key=lambda x: x.get("_relevance", 0), reverse=True)
        # Clean up internal field
        for m in matched:
            m.pop("_relevance", None)
        return matched[:max_matches]

    except Exception as e:
        logger.warning(f"Registry matching failed: {e}")
        # Fallback: tag-based matching
        topic_lower = topic_name.lower()
        sub_lower = {s.lower() for s in (subtopics or [])}
        matched = []
        for s in candidates:
            tags = {t.lower() for t in s.get("tags", [])}
            name_lower = s.get("name", "").lower()
            if (any(t in topic_lower or topic_lower in t for t in tags)
                    or any(t in sub_lower for t in tags)
                    or topic_lower in name_lower):
                matched.append(s)
        return matched[:max_matches]


# ── Step 2: Google News RSS ───────────────────────────────────────────


def _google_news_rss_urls(topic_name: str, subtopics: list[str] | None) -> list[str]:
    """Generate Google News RSS URLs for topic and subtopics."""
    queries = [topic_name]
    if subtopics:
        queries.extend(subtopics[:3])  # Limit to avoid too many feeds

    urls = []
    for q in queries:
        encoded = quote_plus(q)
        urls.append(f"https://news.google.com/rss/search?q={encoded}&hl=en-US&gl=US&ceid=US:en")
    return urls


async def _discover_from_google_news(
    topic_name: str,
    subtopics: list[str] | None,
    existing_urls: set[str],
) -> list[dict]:
    """Create Google News RSS feed sources for the topic."""
    urls = _google_news_rss_urls(topic_name, subtopics)
    feeds = []

    for url in urls:
        if url in existing_urls:
            continue
        validated = await _validate_feed(url)
        if validated:
            # Override metadata — Google News aggregates
            validated["affiliation"] = "private"
            validated["country"] = "US"
            validated["tier"] = "A"
            validated["name"] = f"Google News: {topic_name}"
            feeds.append(validated)

    return feeds


# ── Step 3: Web search discovery (existing) ───────────────────────────


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


# ── Step 4: Metadata classification ──────────────────────────────────


async def classify_feed_metadata(
    llm: LLMClient, feeds: list[dict],
) -> list[dict]:
    """Use LLM to classify affiliation, country, and tier for unknown feeds.

    Only classifies feeds with affiliation='unknown'. Returns the same list
    with metadata updated in-place.
    """
    unknown = [f for f in feeds if f.get("affiliation") == "unknown"]
    if not unknown:
        return feeds

    # Format feeds for LLM
    feed_lines = []
    for f in unknown:
        feed_lines.append(
            f"- id: {f['id']}, name: {f.get('name', '?')}, "
            f"url: {f['url']}, language: {f.get('language', '?')}"
        )

    try:
        raw = await llm.complete(
            config_key="discovery",
            system_prompt=CLASSIFY_PROMPT,
            user_prompt="Classify these feeds:\n" + "\n".join(feed_lines),
            json_response=True,
        )
        classifications = json.loads(raw)
        if not isinstance(classifications, list):
            classifications = [classifications]

        class_map = {c["id"]: c for c in classifications if "id" in c}

        for f in unknown:
            if f["id"] in class_map:
                c = class_map[f["id"]]
                f["affiliation"] = c.get("affiliation", f["affiliation"])
                f["country"] = c.get("country", f["country"])
                f["tier"] = c.get("tier", f["tier"])

    except Exception as e:
        logger.warning(f"Feed classification failed: {e}")

    return feeds


# ── Step 5: Deduplication ─────────────────────────────────────────────


def _deduplicate(feeds: list[dict]) -> list[dict]:
    """Remove duplicate feeds by URL."""
    seen: set[str] = set()
    unique: list[dict] = []
    for f in feeds:
        if f["url"] not in seen:
            seen.add(f["url"])
            unique.append(f)
    return unique


# ── Main orchestrator ─────────────────────────────────────────────────


async def discover_sources(
    llm: LLMClient,
    topic_name: str,
    subtopics: list[str] | None = None,
    existing_urls: set[str] | None = None,
    max_feeds: int = 25,
    data_dir: Path | None = None,
) -> DiscoveryResult:
    """Discover RSS feeds for a topic using multiple strategies.

    Enhanced flow:
      1. Match from global registry (curated sources with rich metadata)
      2. Add Google News RSS feeds (free, always-valid)
      3. Web search discovery (DuckDuckGo)
      4. Classify unknown metadata via LLM
      5. Score diversity

    Returns DiscoveryResult with feeds, diversity metrics, and breakdown.
    """
    existing_urls = existing_urls or set()
    all_feeds: list[dict] = []

    # Step 1: Global registry matching
    registry_matches = []
    if data_dir:
        global_sources = _load_global_registry(data_dir)
        if global_sources:
            registry_matches = await _match_from_global_registry(
                llm, topic_name, subtopics, global_sources, existing_urls,
            )
            all_feeds.extend(registry_matches)
            logger.info(f"Registry matching: {len(registry_matches)} sources for '{topic_name}'")

    # Step 2: Google News RSS
    google_feeds = await _discover_from_google_news(
        topic_name, subtopics, existing_urls | {f["url"] for f in all_feeds},
    )
    all_feeds.extend(google_feeds)

    # Step 3: Web search discovery
    queries = await _generate_search_queries(llm, topic_name, subtopics)
    logger.info(f"Discovery queries for '{topic_name}': {queries}")

    web_urls: list[str] = []
    for query in queries:
        urls = await _find_rss_feeds(query)
        web_urls.extend(urls)

    # Deduplicate and filter existing + already-found
    known_urls = existing_urls | {f["url"] for f in all_feeds}
    unique_web_urls = list(set(web_urls) - known_urls)
    logger.info(f"Found {len(unique_web_urls)} candidate web URLs for '{topic_name}'")

    # Validate web-discovered feeds in parallel
    sem = asyncio.Semaphore(5)

    async def _validate_with_limit(url: str):
        async with sem:
            return await _validate_feed(url)

    tasks = [_validate_with_limit(url) for url in unique_web_urls[:30]]
    results = await asyncio.gather(*tasks)
    web_feeds = [r for r in results if r is not None]
    all_feeds.extend(web_feeds)

    # Deduplicate all feeds
    all_feeds = _deduplicate(all_feeds)

    # Step 4: Classify unknown metadata
    all_feeds = await classify_feed_metadata(llm, all_feeds)

    # Cap to max_feeds (prioritize registry > google > web)
    if len(all_feeds) > max_feeds:
        all_feeds = all_feeds[:max_feeds]

    # Step 5: Diversity scoring
    diversity = compute_diversity(all_feeds)

    return DiscoveryResult(
        feeds=all_feeds,
        diversity=diversity,
        sources_from_registry=len(registry_matches),
        sources_from_web=len(web_feeds),
        sources_from_google_news=len(google_feeds),
    )
