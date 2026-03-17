"""Persistent thread matching and lifecycle management.

Matches new events to existing narrative threads using entity overlap
(fast, no LLM) with LLM confirmation for ambiguous cases.
"""

import json
import logging
import re
from dataclasses import dataclass
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


STALE_AFTER_DAYS = 14


def check_staleness(current_status: str, last_event_date: date, reference_date: date | None = None) -> str:
    """Demote active/emerging threads to 'stale' if no events in STALE_AFTER_DAYS days."""
    ref = reference_date or date.today()
    if current_status in ("resolved", "stale"):
        return current_status
    if (ref - last_event_date).days >= STALE_AFTER_DAYS:
        return "stale"
    return current_status


_MERGE_SYSTEM_PROMPT = (
    "You are a thread consolidation engine for a news intelligence system.\n\n"
    "Given pairs of narrative threads, determine whether each pair tracks the SAME "
    "specific story — meaning they follow the same chain of cause and effect, the same "
    "actors taking the same actions, just described with different headlines.\n\n"
    "IMPORTANT distinctions:\n"
    "- same_arc = TRUE: 'Iran Nuclear Talks Resume' and 'IAEA-Iran Negotiations Continue' "
    "(same story, different wording)\n"
    "- same_arc = FALSE: 'Iran Nuclear Program' and 'Strait of Hormuz Blockade' "
    "(related topics but different causal chains)\n"
    "- same_arc = FALSE: 'US Climate Policy Reversal' and 'UK Grid Modernization' "
    "(same sector but different countries/actors)\n\n"
    "When in doubt, answer FALSE. Merging incorrectly destroys analytical granularity.\n\n"
    "Respond with JSON:\n"
    '{"pairs": [{"thread_a": <id>, "thread_b": <id>, "same_arc": true|false}]}'
)


async def find_merge_candidates(
    threads: list[dict],
    llm: "LLMClient | None" = None,
    high_threshold: float = HIGH_OVERLAP,
    low_threshold: float = LOW_OVERLAP,
) -> list[tuple[int, int]]:
    """Find thread pairs that should be merged using hybrid Jaccard+LLM.

    Returns (keep_id, absorb_id) tuples. Higher significance thread is kept.
    Stage 1: Jaccard >= high_threshold → auto-merge.
    Stage 2: low_threshold <= Jaccard < high_threshold → LLM decides.
    """
    auto_merges: list[tuple[int, int, float]] = []  # (id_a, id_b, overlap)
    ambiguous: list[tuple[int, int, float]] = []

    for i in range(len(threads)):
        ent_i = threads[i].get("key_entities", [])
        for j in range(i + 1, len(threads)):
            ent_j = threads[j].get("key_entities", [])
            overlap = compute_entity_overlap(ent_i, ent_j)
            if overlap >= high_threshold:
                auto_merges.append((threads[i]["id"], threads[j]["id"], overlap))
            elif overlap >= low_threshold:
                ambiguous.append((threads[i]["id"], threads[j]["id"], overlap))

    # Resolve auto-merges: keep higher significance
    sig_map = {t["id"]: t.get("significance", 5) for t in threads}
    created_map = {t["id"]: t.get("created_at", "") for t in threads}

    def _pick_keep(id_a: int, id_b: int) -> tuple[int, int]:
        sig_a, sig_b = sig_map.get(id_a, 5), sig_map.get(id_b, 5)
        if sig_a > sig_b:
            return (id_a, id_b)
        elif sig_b > sig_a:
            return (id_b, id_a)
        # Tie-break: older thread wins
        if created_map.get(id_a, "") <= created_map.get(id_b, ""):
            return (id_a, id_b)
        return (id_b, id_a)

    pairs: list[tuple[int, int]] = []
    for id_a, id_b, _ in auto_merges:
        pairs.append(_pick_keep(id_a, id_b))

    # Stage 2: LLM for ambiguous pairs
    if ambiguous and llm is not None:
        llm_confirmed = await _llm_merge_check(llm, threads, ambiguous)
        for id_a, id_b in llm_confirmed:
            pairs.append(_pick_keep(id_a, id_b))

    # Chain-safe transitive resolution: if A absorbs B and B absorbs C, A absorbs C
    if pairs:
        pairs = _resolve_transitive_merges(pairs)

    return pairs


async def _llm_merge_check(
    llm: "LLMClient",
    threads: list[dict],
    ambiguous: list[tuple[int, int, float]],
) -> list[tuple[int, int]]:
    """Ask LLM to confirm whether ambiguous thread pairs cover the same arc."""
    thread_map = {t["id"]: t for t in threads}
    pair_lines = []
    for id_a, id_b, overlap in ambiguous:
        ta, tb = thread_map[id_a], thread_map[id_b]
        pair_lines.append(
            f"- Thread {id_a}: \"{ta['headline']}\" (entities: {', '.join(ta.get('key_entities', []))})\n"
            f"  Thread {id_b}: \"{tb['headline']}\" (entities: {', '.join(tb.get('key_entities', []))})\n"
            f"  Entity overlap: {overlap:.2f}"
        )

    user_prompt = "Determine if these thread pairs cover the same narrative arc:\n\n" + "\n\n".join(pair_lines)

    try:
        response = await llm.complete(
            config_key="filtering",
            system_prompt=_MERGE_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            json_response=True,
        )
        data = json.loads(response)
        confirmed = []
        for p in data.get("pairs", []):
            if p.get("same_arc"):
                confirmed.append((p["thread_a"], p["thread_b"]))
        return confirmed
    except Exception as e:
        logger.warning(f"Thread merge LLM check failed: {e}. Skipping ambiguous pairs.")
        return []


def _resolve_transitive_merges(pairs: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Resolve transitive chains: if A←B and B←C, return A←B and A←C."""
    # Build a union-find to collapse chains
    parent: dict[int, int] = {}

    def find(x: int) -> int:
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent[x], parent[x])
            x = parent[x]
        return x

    # For each pair, the keep_id is the root
    keep_sig: dict[int, int] = {}
    for keep_id, absorb_id in pairs:
        keep_sig[keep_id] = keep_sig.get(keep_id, 0)
        keep_sig[absorb_id] = keep_sig.get(absorb_id, 0)
        root_keep = find(keep_id)
        root_absorb = find(absorb_id)
        if root_keep != root_absorb:
            # Always make the keep_id the root
            parent[root_absorb] = root_keep

    # Reconstruct pairs: each non-root points to its root
    all_ids = set()
    for k, a in pairs:
        all_ids.add(k)
        all_ids.add(a)

    result = []
    seen_absorb = set()
    for tid in all_ids:
        root = find(tid)
        if tid != root and tid not in seen_absorb:
            result.append((root, tid))
            seen_absorb.add(tid)
    return result


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
