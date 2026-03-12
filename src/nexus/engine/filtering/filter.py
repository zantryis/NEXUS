"""Relevance filtering — score content items against topic definitions via LLM.

Two-pass filtering:
  Pass 1 (relevance): Cheap batch scoring — does this article match the topic? (1-10)
  Pass 2 (significance + novelty): With full text + knowledge context — is this
    article significant and does it report something new? (1-10 + boolean)
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

from nexus.config.models import TopicConfig
from nexus.engine.sources.polling import ContentItem
from nexus.llm.client import LLMClient

logger = logging.getLogger(__name__)

# --- Pass 1: Relevance batch scoring ---

BATCH_SYSTEM_PROMPT = (
    "You are a relevance scorer. Given a list of articles and a topic definition, "
    "score how relevant each article is to the topic on a scale of 1-10. "
    "Include source metadata (affiliation, country) in your assessment — "
    "state media, public broadcasters, and private outlets all provide valid perspectives. "
    "Respond with a JSON array of objects: [{\"id\": <int>, \"score\": <int>, \"reason\": \"<brief>\"}] "
    "for each article. Use the article's id number as provided."
)

SINGLE_SYSTEM_PROMPT = (
    "You are a relevance scorer. Given an article and a topic definition, "
    "score how relevant the article is to the topic on a scale of 1-10. "
    'Respond with JSON: {"score": <int>, "reason": "<brief explanation>"}'
)

BATCH_SIZE = 10

# --- Pass 2: Significance + novelty scoring ---

PASS2_SYSTEM_PROMPT = (
    "You assess article significance and novelty. Given articles, a topic definition, "
    "and recent known events, score each article on:\n"
    "- significance (1-10): How important is this development for the topic?\n"
    "- is_novel (boolean): Does this report NEW information not already captured in the known events?\n\n"
    "Respond with a JSON array: [{\"id\": <int>, \"significance\": <int>, \"is_novel\": <bool>, \"reason\": \"<brief>\"}]"
)

PASS2_BATCH_SIZE = 5  # Smaller batches — more text per article


@dataclass
class FilterResult:
    """Result of filter_items() — accepted items + full decision log."""
    accepted: list[ContentItem] = field(default_factory=list)
    log_entries: list[dict] = field(default_factory=list)


async def score_relevance(
    llm: LLMClient, item: ContentItem, topic: TopicConfig
) -> tuple[int, str]:
    """Score a single item's relevance to a topic. Returns (score, reason)."""
    user_prompt = (
        f"Topic: {topic.name}\n"
        f"Subtopics: {', '.join(topic.subtopics)}\n\n"
        f"Article title: {item.title}\n"
        f"Article text: {(item.full_text or item.snippet)[:2000]}"
    )
    try:
        response = await llm.complete(
            config_key="filtering",
            system_prompt=SINGLE_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            json_response=True,
        )
        data = json.loads(response)
        return int(data["score"]), data["reason"]
    except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
        logger.warning(f"Failed to parse relevance score for {item.url}: {e}")
        return 0, f"Failed to parse response: {e}"


async def score_batch(
    llm: LLMClient, items: list[ContentItem], topic: TopicConfig
) -> list[tuple[int, str]]:
    """Score a batch of items in a single LLM call. Returns list of (score, reason)."""
    articles_text = []
    for i, item in enumerate(items):
        text = (item.full_text or item.snippet or "")[:1000]
        source_meta = ""
        if item.source_affiliation or item.source_country:
            source_meta = f" [{item.source_affiliation or 'unknown'}/{item.source_country or '?'}]"
        articles_text.append(
            f"[Article {i}] Source: {item.source_id}{source_meta}\n"
            f"Title: {item.title}\nText: {text}\n"
        )

    user_prompt = (
        f"Topic: {topic.name}\n"
        f"Subtopics: {', '.join(topic.subtopics)}\n\n"
        f"Score each article:\n\n" + "\n".join(articles_text)
    )
    try:
        response = await llm.complete(
            config_key="filtering",
            system_prompt=BATCH_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            json_response=True,
        )
        data = json.loads(response)
        if not isinstance(data, list):
            data = [data]

        # Build a map of id -> (score, reason)
        score_map = {}
        for entry in data:
            idx = int(entry["id"])
            score_map[idx] = (int(entry["score"]), entry.get("reason", ""))

        # Return in order, defaulting to 0 for missing
        results = []
        for i in range(len(items)):
            results.append(score_map.get(i, (0, "Missing from batch response")))
        return results

    except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
        logger.warning(f"Batch scoring failed, falling back to individual: {e}")
        results = []
        for item in items:
            results.append(await score_relevance(llm, item, topic))
        return results


async def score_significance_batch(
    llm: LLMClient,
    items: list[ContentItem],
    topic: TopicConfig,
    event_context: str,
) -> list[dict]:
    """Pass 2: Score significance + novelty with full context.

    Returns list of {"significance": int, "is_novel": bool, "reason": str}.
    """
    articles_text = []
    for i, item in enumerate(items):
        text = (item.full_text or item.snippet or "")[:2000]
        source_meta = ""
        if item.source_affiliation or item.source_country:
            source_meta = f" [{item.source_affiliation or 'unknown'}/{item.source_country or '?'}]"
        lang = item.detected_language or item.source_language or "unknown"
        articles_text.append(
            f"[Article {i}] Source: {item.source_id}{source_meta} Language: {lang}\n"
            f"Title: {item.title}\nText: {text}\n"
        )

    user_prompt = (
        f"Topic: {topic.name}\n"
        f"Subtopics: {', '.join(topic.subtopics)}\n\n"
        f"Known events (last 7 days):\n{event_context or 'None yet'}\n\n"
        f"Assess each article:\n\n" + "\n".join(articles_text)
    )

    try:
        response = await llm.complete(
            config_key="filtering",
            system_prompt=PASS2_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            json_response=True,
        )
        data = json.loads(response)
        if not isinstance(data, list):
            data = [data]

        score_map = {}
        for entry in data:
            idx = int(entry["id"])
            score_map[idx] = {
                "significance": int(entry.get("significance", 5)),
                "is_novel": bool(entry.get("is_novel", True)),
                "reason": entry.get("reason", ""),
            }

        results = []
        for i in range(len(items)):
            results.append(score_map.get(i, {"significance": 5, "is_novel": True, "reason": "Missing"}))
        return results

    except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
        logger.warning(f"Pass 2 batch scoring failed: {e}")
        # Default: assume novel and moderately significant
        return [{"significance": 5, "is_novel": True, "reason": "Parse error fallback"}] * len(items)


def apply_perspective_diversity(
    items: list[ContentItem],
    topic: TopicConfig,
    max_items: int = 30,
) -> list[ContentItem]:
    """Apply perspective diversity constraints based on topic config.

    For perspective_diversity="high": ensure >=20% from each affiliation type present.
    For perspective_diversity="medium": ensure >=10% from each affiliation type.
    For perspective_diversity="low": pure score ranking, no diversity constraint.
    """
    if topic.perspective_diversity == "low" or not items:
        return sorted(items, key=lambda x: x.relevance_score or 0, reverse=True)[:max_items]

    min_pct = 0.2 if topic.perspective_diversity == "high" else 0.1

    # Group by affiliation
    by_affiliation: dict[str, list[ContentItem]] = {}
    for item in items:
        affil = item.source_affiliation or "unknown"
        by_affiliation.setdefault(affil, []).append(item)

    # Sort each group by score
    for affil in by_affiliation:
        by_affiliation[affil].sort(key=lambda x: x.relevance_score or 0, reverse=True)

    num_affiliations = len(by_affiliation)
    if num_affiliations <= 1:
        return sorted(items, key=lambda x: x.relevance_score or 0, reverse=True)[:max_items]

    # Guarantee minimum slots per affiliation
    min_per_type = max(1, int(max_items * min_pct))
    selected: list[ContentItem] = []
    remaining: list[ContentItem] = []

    for affil, group in by_affiliation.items():
        guaranteed = group[:min_per_type]
        selected.extend(guaranteed)
        remaining.extend(group[min_per_type:])

    # Fill remaining slots by score
    remaining.sort(key=lambda x: x.relevance_score or 0, reverse=True)
    slots_left = max_items - len(selected)
    if slots_left > 0:
        selected.extend(remaining[:slots_left])

    # Log affiliation distribution
    dist = {}
    for item in selected:
        affil = item.source_affiliation or "unknown"
        dist[affil] = dist.get(affil, 0) + 1
    logger.info(f"Perspective diversity ({topic.perspective_diversity}): {dist}")

    return selected


def _format_event_context(events: list) -> str:
    """Format recent events as context string for pass 2."""
    if not events:
        return ""
    lines = []
    for e in events:
        summary = e.summary if hasattr(e, "summary") else str(e)
        event_date = e.date if hasattr(e, "date") else "?"
        lines.append(f"- [{event_date}] {summary}")
    return "\n".join(lines)


async def filter_items(
    llm: LLMClient,
    items: list[ContentItem],
    topic: TopicConfig,
    threshold: float | None = None,
    recent_events: Optional[list] = None,
) -> FilterResult:
    """Two-pass filter: relevance batch → significance+novelty with context.

    Pass 1: Batch relevance scoring against topic (cheap, uses topic.filter_threshold).
    Pass 2: Significance + novelty for survivors (uses full text + event context).

    Returns FilterResult with accepted items and full decision log for all items.
    """
    effective_threshold = threshold if threshold is not None else topic.filter_threshold
    slug = topic.name.lower().replace(" ", "-").replace("/", "-")
    today = date.today().isoformat()

    # Track all items through the pipeline
    log_entries: list[dict] = []

    # Initialize log entry for every item
    item_logs: dict[str, dict] = {}
    for item in items:
        entry = {
            "run_date": today,
            "topic_slug": slug,
            "url": item.url,
            "title": item.title or "",
            "source_id": item.source_id or "",
            "source_affiliation": item.source_affiliation or "",
            "source_country": item.source_country or "",
            "relevance_score": None,
            "relevance_reason": "",
            "passed_pass1": False,
            "significance_score": None,
            "is_novel": None,
            "significance_reason": "",
            "passed_pass2": None,
            "final_score": None,
            "outcome": "rejected_relevance",
        }
        item_logs[item.url] = entry

    # --- Pass 1: Relevance ---
    pass1_results = []
    for batch_start in range(0, len(items), BATCH_SIZE):
        batch = items[batch_start:batch_start + BATCH_SIZE]
        scores = await score_batch(llm, batch, topic)

        for item, (score, reason) in zip(batch, scores):
            item.relevance_score = score
            log = item_logs[item.url]
            log["relevance_score"] = score
            log["relevance_reason"] = reason

            if score >= effective_threshold:
                log["passed_pass1"] = True
                pass1_results.append(item)
            else:
                log["outcome"] = "rejected_relevance"
                logger.debug(f"Pass 1 filtered out (score={score}): {item.title}")

    logger.info(f"Pass 1: {len(pass1_results)}/{len(items)} passed relevance filter (threshold={effective_threshold})")

    if not pass1_results:
        log_entries = list(item_logs.values())
        return FilterResult(accepted=[], log_entries=log_entries)

    # --- Pass 2: Significance + Novelty (only if we have events for context) ---
    if recent_events is None:
        # No event context available — skip pass 2, mark pass1 survivors as accepted
        for item in pass1_results:
            log = item_logs[item.url]
            log["final_score"] = item.relevance_score
            log["outcome"] = "accepted"
        log_entries = list(item_logs.values())
        return FilterResult(accepted=pass1_results, log_entries=log_entries)

    event_context = _format_event_context(recent_events)
    pass2_results = []

    for batch_start in range(0, len(pass1_results), PASS2_BATCH_SIZE):
        batch = pass1_results[batch_start:batch_start + PASS2_BATCH_SIZE]
        assessments = await score_significance_batch(llm, batch, topic, event_context)

        for item, assessment in zip(batch, assessments):
            sig = assessment["significance"]
            is_novel = assessment["is_novel"]
            reason = assessment.get("reason", "")

            log = item_logs[item.url]
            log["significance_score"] = sig
            log["is_novel"] = is_novel
            log["significance_reason"] = reason

            # Composite score: relevance (from pass 1) weighted with significance
            # Novel articles get a boost
            relevance = item.relevance_score or 0
            novelty_bonus = 1.0 if is_novel else 0.7
            composite = (relevance * 0.4 + sig * 0.6) * novelty_bonus
            item.relevance_score = round(composite, 1)

            # Keep if significance >= 4 or novel
            if sig >= 4 or is_novel:
                log["passed_pass2"] = True
                log["final_score"] = item.relevance_score
                pass2_results.append(item)
            else:
                log["passed_pass2"] = False
                log["outcome"] = "rejected_significance"
                logger.debug(f"Pass 2 filtered out (sig={sig}, novel={is_novel}): {item.title}")

    logger.info(f"Pass 2: {len(pass2_results)}/{len(pass1_results)} passed significance+novelty filter")

    # --- Perspective diversity selection ---
    selected = apply_perspective_diversity(pass2_results, topic)

    # Mark diversity-rejected items
    selected_urls = {item.url for item in selected}
    for item in pass2_results:
        log = item_logs[item.url]
        if item.url in selected_urls:
            log["outcome"] = "accepted"
        else:
            log["outcome"] = "rejected_diversity"

    log_entries = list(item_logs.values())
    return FilterResult(accepted=selected, log_entries=log_entries)
