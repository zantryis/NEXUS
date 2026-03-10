"""Persistent thread matching and lifecycle management.

Matches new events to existing narrative threads using entity overlap
(fast, no LLM) with LLM confirmation for ambiguous cases.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import date

from nexus.engine.knowledge.events import Event
from nexus.llm.client import LLMClient

logger = logging.getLogger(__name__)

# Overlap thresholds for thread matching
HIGH_OVERLAP = 0.5   # Auto-match, no LLM needed
LOW_OVERLAP = 0.3    # Below this = no match candidate


@dataclass
class ThreadMatch:
    """Result of matching an event to a thread."""
    event_index: int
    thread_slug: str | None = None
    is_new_thread: bool = False
    new_headline: str | None = None


def compute_entity_overlap(entities_a: list[str], entities_b: list[str]) -> float:
    """Compute Jaccard similarity between two entity lists (case-insensitive)."""
    if not entities_a or not entities_b:
        return 0.0
    set_a = {e.lower() for e in entities_a}
    set_b = {e.lower() for e in entities_b}
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union) if union else 0.0


def create_thread_slug(headline: str) -> str:
    """Create a URL-safe slug from a thread headline."""
    slug = headline.lower()
    slug = re.sub(r"[^a-z0-9\s-]", "", slug)
    slug = re.sub(r"[\s]+", "-", slug.strip())
    slug = re.sub(r"-+", "-", slug)
    return slug[:80]


def promote_thread_status(current_status: str, event_dates: list[date]) -> str:
    """Determine thread status based on lifecycle rules.

    - emerging → active: events from 2+ distinct days
    - active stays active
    - resolved stays resolved
    """
    if current_status == "resolved":
        return "resolved"
    if current_status == "active":
        return "active"
    # emerging → check if multi-day
    unique_days = len(set(event_dates))
    if unique_days >= 2:
        return "active"
    return "emerging"


MATCH_SYSTEM_PROMPT = (
    "You are a thread matching engine. Given new events and existing narrative threads, "
    "determine which thread each event belongs to, or if it starts a new thread.\n\n"
    "For each event, either:\n"
    "1. Match it to an existing thread by slug\n"
    "2. Group unmatched events into new threads with headlines\n\n"
    "Respond with JSON:\n"
    '{"matched": [{"event_index": 0, "thread_slug": "slug", "confidence": "high|medium"}],\n'
    ' "new_threads": [{"headline": "New Thread Title", "event_indices": [1, 3]}]}'
)


async def match_events_to_threads(
    llm: LLMClient,
    events: list[Event],
    active_threads: list[dict],
) -> list[ThreadMatch]:
    """Match events to existing threads or create new ones.

    Two-stage matching:
    1. Entity overlap (no LLM): >= 0.5 = auto-match
    2. LLM confirmation: for ambiguous cases (0.3-0.5) or unmatched events
    """
    if not events:
        return []

    matches: list[ThreadMatch] = []
    needs_llm: list[int] = []  # Event indices needing LLM resolution

    # Stage 1: Entity overlap matching
    for i, event in enumerate(events):
        best_thread = None
        best_overlap = 0.0

        for thread in active_threads:
            thread_entities = thread.get("key_entities", [])
            overlap = compute_entity_overlap(event.entities, thread_entities)
            if overlap > best_overlap:
                best_overlap = overlap
                best_thread = thread

        if best_overlap >= HIGH_OVERLAP and best_thread:
            matches.append(ThreadMatch(
                event_index=i,
                thread_slug=best_thread["slug"],
                is_new_thread=False,
            ))
        elif best_overlap >= LOW_OVERLAP and best_thread:
            # Ambiguous — needs LLM
            needs_llm.append(i)
        else:
            # No match candidate
            needs_llm.append(i)

    # Stage 2: LLM for ambiguous/unmatched events
    if needs_llm:
        llm_matches = await _llm_match(llm, events, needs_llm, active_threads)
        matches.extend(llm_matches)

    return matches


async def _llm_match(
    llm: LLMClient,
    events: list[Event],
    event_indices: list[int],
    active_threads: list[dict],
) -> list[ThreadMatch]:
    """Use LLM to match ambiguous events or group into new threads."""
    # Format events for LLM
    event_lines = []
    for idx in event_indices:
        e = events[idx]
        event_lines.append(
            f"[Event {idx}] {e.summary} (entities: {', '.join(e.entities)})"
        )

    thread_lines = []
    for t in active_threads:
        entities = ", ".join(t.get("key_entities", []))
        thread_lines.append(
            f"- {t['slug']}: {t['headline']} (entities: {entities})"
        )

    user_prompt = (
        f"Existing threads:\n{chr(10).join(thread_lines) or 'None'}\n\n"
        f"Events to match:\n" + "\n".join(event_lines)
    )

    try:
        response = await llm.complete(
            config_key="knowledge_summary",
            system_prompt=MATCH_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            json_response=True,
        )
        data = json.loads(response)

        results: list[ThreadMatch] = []
        matched_indices = set()

        # Process matched events
        for m in data.get("matched", []):
            idx = m["event_index"]
            matched_indices.add(idx)
            results.append(ThreadMatch(
                event_index=idx,
                thread_slug=m["thread_slug"],
                is_new_thread=False,
            ))

        # Process new threads
        for nt in data.get("new_threads", []):
            headline = nt["headline"]
            slug = create_thread_slug(headline)
            for idx in nt.get("event_indices", []):
                matched_indices.add(idx)
                results.append(ThreadMatch(
                    event_index=idx,
                    thread_slug=slug,
                    is_new_thread=True,
                    new_headline=headline,
                ))

        # Any remaining unmatched events get their own thread
        for idx in event_indices:
            if idx not in matched_indices:
                headline = events[idx].summary[:80]
                results.append(ThreadMatch(
                    event_index=idx,
                    thread_slug=create_thread_slug(headline),
                    is_new_thread=True,
                    new_headline=headline,
                ))

        return results

    except Exception as e:
        logger.warning(f"Thread matching LLM failed: {e}. Creating new threads.")
        # Fallback: each unmatched event becomes its own thread
        return [
            ThreadMatch(
                event_index=idx,
                thread_slug=create_thread_slug(events[idx].summary[:80]),
                is_new_thread=True,
                new_headline=events[idx].summary[:80],
            )
            for idx in event_indices
        ]
