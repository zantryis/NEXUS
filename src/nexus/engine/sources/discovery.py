"""Source auto-discovery — find RSS feeds for new topics via LLM + web search.

Agentic flow with budget-aware refinement:
  1. Match from global registry (curated, rich metadata) — primary path
  2. Google News RSS (free, always valid)
  3. Web search discovery (DuckDuckGo)
  3b. Evaluate discovered feeds (score sample article titles against topic)
  3c. Refine queries if too few good feeds found (budget-aware loop)
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
    "IMPORTANT: Prioritize specialized, topic-specific publications and trade journals "
    "over general news outlets. Search for industry associations, research organizations, "
    "and dedicated news services covering this topic.\n\n"
    "Include queries like:\n"
    '- "{topic} industry news RSS feed"\n'
    '- "{topic} trade journal RSS"\n'
    '- "{topic} research organization news feed"\n'
    '- "{topic} association blog RSS"\n'
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
    "Score conservatively for niche topics: a general tech outlet should score 3-4 for "
    "a specific topic like 'Semiconductor Supply Chain'. Only score 7+ if the source "
    "has dedicated, regular coverage of the topic.\n\n"
    "Respond with a JSON array: [{\"id\": \"<source_id>\", \"score\": <int>, \"reason\": \"<brief>\"}]\n"
    "Only include sources scoring 5 or above."
)

CLASSIFY_PROMPT = (
    "You classify news source metadata. Given a list of RSS feeds with their names and "
    "sample entry titles, determine for each:\n"
    "- affiliation: one of 'public', 'private', 'state', 'nonprofit', 'academic'\n"
    "- country: ISO 3166-1 alpha-2 code (e.g., 'US', 'GB', 'DE')\n"
    "- tier: 'A' (major outlet with original reporting), 'B' (established secondary), "
    "'C' (niche blog/social media)\n\n"
    "Default to tier B. Only assign A to well-known outlets with original reporting. "
    "Assign C to personal blogs, social media aggregators, and unverified sources.\n\n"
    "Respond with JSON array: [{\"id\": \"<feed_id>\", \"affiliation\": \"...\", "
    "\"country\": \"...\", \"tier\": \"...\"}]\n"
    "If uncertain about a field, use your best guess based on the name and content."
)

EVALUATE_PROMPT = (
    "You evaluate RSS feed quality for a specific topic. "
    "Given sample article titles from a feed and a target topic, "
    "score how relevant this feed is on a scale of 1-10.\n\n"
    "Score 8-10: Dedicated coverage, most articles are on-topic\n"
    "Score 5-7: Regular coverage, some articles are relevant\n"
    "Score 1-4: Generic outlet, rarely covers this topic\n\n"
    "For each feed, respond with JSON: {\"score\": <int>, \"reason\": \"<brief>\"}\n"
    "Respond with a JSON array of objects, one per feed."
)

REFINE_QUERIES_PROMPT = (
    "The initial search queries found generic news outlets instead of "
    "topic-specific sources. Generate 3 more targeted search queries "
    "to find specialized RSS feeds.\n\n"
    "Focus on:\n"
    "- Industry-specific publications and trade journals\n"
    "- Research institutions and think tanks\n"
    "- Government agencies and regulatory bodies with RSS feeds\n"
    "- Regional outlets known for covering this topic\n\n"
    "Do NOT repeat generic queries like '{topic} news RSS'.\n"
    "Output JSON array of query strings only."
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


# ── Step 3b: Feed evaluation (agentic) ───────────────────────────────


async def _sample_feed_titles(url: str, max_titles: int = 5) -> list[str]:
    """Fetch a few article titles from an RSS feed for relevance evaluation."""
    try:
        loop = asyncio.get_event_loop()
        parsed = await loop.run_in_executor(None, feedparser.parse, url)
        return [
            entry.get("title", "")
            for entry in parsed.entries[:max_titles]
            if entry.get("title")
        ]
    except Exception:
        return []


async def _evaluate_feed_relevance(
    llm: LLMClient,
    feeds: list[dict],
    topic_name: str,
) -> list[dict]:
    """Score discovered feeds by checking sample article titles against topic.

    Returns only feeds scoring >= 5 (relevant enough to keep).
    """
    if not feeds:
        return []

    # Fetch sample titles for each feed in parallel
    sem = asyncio.Semaphore(5)

    async def _fetch(feed):
        async with sem:
            titles = await _sample_feed_titles(feed["url"])
            return feed, titles

    results = await asyncio.gather(*[_fetch(f) for f in feeds])

    # Build evaluation prompt
    feed_blocks = []
    evaluable = []
    for feed, titles in results:
        if not titles:
            continue
        evaluable.append(feed)
        title_list = "\n".join(f"  - {t}" for t in titles)
        feed_blocks.append(
            f"Feed: {feed.get('name', feed['id'])} ({feed['url']})\n"
            f"Sample titles:\n{title_list}"
        )

    if not feed_blocks:
        return feeds  # Can't evaluate, return all

    user_prompt = (
        f"Topic: {topic_name}\n\n"
        + "\n\n".join(feed_blocks)
    )

    try:
        raw = await llm.complete(
            config_key="discovery",
            system_prompt=EVALUATE_PROMPT,
            user_prompt=user_prompt,
            json_response=True,
        )
        scores = json.loads(raw)
        if not isinstance(scores, list):
            scores = [scores]

        good_feeds = []
        for feed, score_entry in zip(evaluable, scores):
            score = score_entry.get("score", 0) if isinstance(score_entry, dict) else 0
            if score >= 5:
                good_feeds.append(feed)
            else:
                reason = score_entry.get("reason", "") if isinstance(score_entry, dict) else ""
                logger.info(
                    f"Dropping feed '{feed.get('name', feed['id'])}' "
                    f"(score={score}, reason={reason})"
                )

        # Include feeds we couldn't evaluate (no titles fetched)
        evaluated_urls = {f["url"] for f in evaluable}
        for feed in feeds:
            if feed["url"] not in evaluated_urls:
                good_feeds.append(feed)

        return good_feeds

    except Exception as e:
        logger.warning(f"Feed evaluation failed: {e}")
        return feeds  # On failure, keep all


async def _generate_refined_queries(
    llm: LLMClient,
    topic_name: str,
    subtopics: list[str] | None,
    previous_queries: list[str],
) -> list[str]:
    """Generate more specific search queries after initial results were too generic."""
    context = f"Topic: {topic_name}"
    if subtopics:
        context += f"\nSubtopics: {', '.join(subtopics)}"
    context += f"\n\nPrevious queries that found generic results:\n"
    context += "\n".join(f"- {q}" for q in previous_queries)

    try:
        raw = await llm.complete(
            config_key="discovery",
            system_prompt=REFINE_QUERIES_PROMPT,
            user_prompt=context,
            json_response=True,
        )
        queries = json.loads(raw)
        if isinstance(queries, list):
            return [q for q in queries if isinstance(q, str)]
    except Exception as e:
        logger.warning(f"Refined query generation failed: {e}")

    return [
        f"{topic_name} trade journal RSS feed",
        f"{topic_name} industry association news",
    ]


# ── Main orchestrator ─────────────────────────────────────────────────


async def discover_sources(
    llm: LLMClient,
    topic_name: str,
    subtopics: list[str] | None = None,
    existing_urls: set[str] | None = None,
    max_feeds: int = 25,
    data_dir: Path | None = None,
    max_rounds: int = 2,
    max_llm_calls: int = 8,
) -> DiscoveryResult:
    """Discover RSS feeds for a topic using multiple strategies.

    Enhanced agentic flow:
      1. Match from global registry (curated sources with rich metadata)
      2. Add Google News RSS feeds (free, always-valid)
      3. Web search discovery (DuckDuckGo)
      3b. Evaluate discovered feeds against topic (score sample titles)
      3c. Refine queries if too few good feeds found (budget-aware loop)
      4. Classify unknown metadata via LLM
      5. Score diversity

    Args:
        max_rounds: Maximum discovery rounds (1=initial only, 2+=with refinement).
        max_llm_calls: Budget cap for total LLM calls during discovery.

    Returns DiscoveryResult with feeds, diversity metrics, and breakdown.
    """
    existing_urls = existing_urls or set()
    all_feeds: list[dict] = []
    llm_calls = 0

    # Step 1: Global registry matching
    registry_matches = []
    if data_dir:
        global_sources = _load_global_registry(data_dir)
        if global_sources:
            registry_matches = await _match_from_global_registry(
                llm, topic_name, subtopics, global_sources, existing_urls,
            )
            llm_calls += 1
            all_feeds.extend(registry_matches)
            logger.info(f"Registry matching: {len(registry_matches)} sources for '{topic_name}'")

    # Step 2: Google News RSS
    google_feeds = await _discover_from_google_news(
        topic_name, subtopics, existing_urls | {f["url"] for f in all_feeds},
    )
    all_feeds.extend(google_feeds)

    # Step 3: Web search discovery with agentic refinement loop
    all_queries: list[str] = []
    web_feeds: list[dict] = []

    for round_num in range(1, max_rounds + 1):
        if llm_calls >= max_llm_calls:
            logger.info(f"Discovery budget exhausted ({llm_calls}/{max_llm_calls} calls)")
            break

        # Generate queries (initial or refined)
        if round_num == 1:
            queries = await _generate_search_queries(llm, topic_name, subtopics)
        else:
            queries = await _generate_refined_queries(
                llm, topic_name, subtopics, all_queries,
            )
        llm_calls += 1
        all_queries.extend(queries)
        logger.info(
            f"Discovery round {round_num}/{max_rounds} for '{topic_name}': "
            f"queries={queries} (budget: {llm_calls}/{max_llm_calls} calls)"
        )

        # Search and validate
        round_urls: list[str] = []
        for query in queries:
            urls = await _find_rss_feeds(query)
            round_urls.extend(urls)

        known_urls = existing_urls | {f["url"] for f in all_feeds}
        unique_urls = list(set(round_urls) - known_urls)
        logger.info(f"Round {round_num}: {len(unique_urls)} candidate URLs")

        # Validate feeds in parallel
        val_sem = asyncio.Semaphore(5)

        async def _validate_with_limit(url: str):
            async with val_sem:
                return await _validate_feed(url)

        tasks = [_validate_with_limit(url) for url in unique_urls[:30]]
        results = await asyncio.gather(*tasks)
        round_feeds = [r for r in results if r is not None]

        # Step 3b: Evaluate feed relevance (score sample titles)
        if round_feeds and llm_calls < max_llm_calls:
            round_feeds = await _evaluate_feed_relevance(
                llm, round_feeds, topic_name,
            )
            llm_calls += 1
            logger.info(
                f"Round {round_num}: {len(round_feeds)} feeds passed evaluation "
                f"(budget: {llm_calls}/{max_llm_calls} calls)"
            )

        web_feeds.extend(round_feeds)
        all_feeds.extend(round_feeds)

        # Step 3c: Decide whether to refine
        # Count "good" web feeds (not from registry/google)
        if len(web_feeds) >= 3:
            logger.info(
                f"Round {round_num}: {len(web_feeds)} good web feeds found, "
                f"stopping refinement"
            )
            break
        elif round_num < max_rounds:
            logger.info(
                f"Round {round_num}: only {len(web_feeds)} good web feeds, "
                f"will refine queries"
            )

    # Deduplicate all feeds
    all_feeds = _deduplicate(all_feeds)

    # Step 4: Classify unknown metadata
    if llm_calls < max_llm_calls:
        all_feeds = await classify_feed_metadata(llm, all_feeds)
        llm_calls += 1

    # Cap to max_feeds (prioritize registry > google > web)
    if len(all_feeds) > max_feeds:
        all_feeds = all_feeds[:max_feeds]

    # Step 5: Diversity scoring
    diversity = compute_diversity(all_feeds)

    logger.info(
        f"Discovery complete for '{topic_name}': "
        f"{len(all_feeds)} feeds ({len(registry_matches)} registry, "
        f"{len(google_feeds)} google, {len(web_feeds)} web), "
        f"{llm_calls} LLM calls used"
    )

    return DiscoveryResult(
        feeds=all_feeds,
        diversity=diversity,
        sources_from_registry=len(registry_matches),
        sources_from_web=len(web_feeds),
        sources_from_google_news=len(google_feeds),
    )
