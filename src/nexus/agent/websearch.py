"""Web search integration for Q&A — supplements knowledge store gaps."""

import asyncio
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Minimum lines of store context to consider "sufficient"
THIN_CONTEXT_THRESHOLD = 10

try:
    from ddgs import DDGS
except ImportError:
    try:
        from duckduckgo_search import DDGS  # legacy name
    except ImportError:
        DDGS = None  # type: ignore


def _is_context_thin(context: str) -> bool:
    """Check if store context is insufficient to answer well."""
    if "No data available" in context:
        return True
    lines = [l for l in context.strip().splitlines() if l.strip() and not l.startswith("##")]
    return len(lines) < THIN_CONTEXT_THRESHOLD


async def web_search(query: str, max_results: int = 5) -> list[dict]:
    """Search the web using DuckDuckGo. Returns list of {title, snippet, url}."""
    if DDGS is None:
        logger.debug("duckduckgo-search not installed, skipping web search")
        return []

    try:
        # Run in thread pool since DDGS is synchronous
        def _search():
            with DDGS() as ddgs:
                return list(ddgs.text(query, max_results=max_results))

        raw = await asyncio.get_event_loop().run_in_executor(None, _search)
        return [
            {
                "title": r.get("title", ""),
                "snippet": r.get("body", ""),
                "url": r.get("href", ""),
            }
            for r in raw
        ]
    except Exception as e:
        logger.warning(f"Web search failed: {e}")
        return []


def format_web_results(results: list[dict]) -> str:
    """Format web search results into context for the LLM."""
    if not results:
        return ""
    parts = ["\n## Web search results (supplement)"]
    for r in results:
        parts.append(f"- **{r['title']}**: {r['snippet']} ({r['url']})")
    return "\n".join(parts)
