"""Pairwise comparison engine for hybrid funnel filtering.

Resolves ambiguous MAYBE items (score 3-6) by comparing them against
known-good KEEP items (score 7+) using relative LLM judgments.

Stages:
  triage_split()        — split scored items into KEEP / MAYBE / DROP pools
  detect_degenerate()   — detect collapsed score distributions
  select_references()   — pick diverse reference items from KEEP pool
  compare_batch()       — LLM pairwise comparison, returns (promoted, reason)
  resolve_maybe_items() — orchestrate MAYBE resolution via pairwise
"""

import json
import logging
import random
from collections import defaultdict

from nexus.config.models import TopicConfig
from nexus.engine.sources.polling import ContentItem

logger = logging.getLogger(__name__)

# Triage thresholds
KEEP_THRESHOLD = 7    # score >= 7 → KEEP
DROP_THRESHOLD = 3    # score < 3  → DROP
# scores in [DROP_THRESHOLD, KEEP_THRESHOLD) → MAYBE

PAIRWISE_COMPARISONS = 3  # each MAYBE compared against N references


MIN_DEGENERATE_SAMPLES = 10  # need enough items for meaningful distribution


def detect_degenerate(scores: list[float], threshold: float = 0.7) -> bool:
    """Detect collapsed score distributions.

    Returns True if >threshold fraction of scores fall in any 2-point band,
    indicating the model isn't differentiating between articles.
    Requires at least MIN_DEGENERATE_SAMPLES items for meaningful detection.
    """
    if len(scores) < MIN_DEGENERATE_SAMPLES:
        return False

    n = len(scores)
    for base in range(1, 10):
        band = sum(1 for s in scores if base <= s <= base + 1)
        if band / n > threshold:
            return True
    return False


def triage_split(
    items: list[ContentItem],
) -> tuple[list[ContentItem], list[ContentItem], list[ContentItem]]:
    """Split scored items into KEEP / MAYBE / DROP pools.

    - score >= 7  → KEEP
    - score < 3   → DROP
    - 3 <= score < 7 or None → MAYBE
    """
    keep, maybe, drop = [], [], []
    for item in items:
        score = item.relevance_score
        if score is None:
            maybe.append(item)
        elif score >= KEEP_THRESHOLD:
            keep.append(item)
        elif score < DROP_THRESHOLD:
            drop.append(item)
        else:
            maybe.append(item)
    return keep, maybe, drop


def select_references(
    keep_items: list[ContentItem],
    n: int = PAIRWISE_COMPARISONS,
) -> list[ContentItem]:
    """Select diverse reference items from the KEEP pool.

    Prefers source diversity (different affiliations) when possible.
    """
    if not keep_items or n <= 0:
        return []

    if len(keep_items) <= n:
        return list(keep_items)

    # Group by affiliation for diversity
    by_affiliation: dict[str, list[ContentItem]] = defaultdict(list)
    for item in keep_items:
        affil = item.source_affiliation or "unknown"
        by_affiliation[affil].append(item)

    selected: list[ContentItem] = []

    # Round-robin from each affiliation group
    affiliations = list(by_affiliation.keys())
    random.shuffle(affiliations)
    idx_per_affil = {a: 0 for a in affiliations}

    while len(selected) < n:
        added_any = False
        for affil in affiliations:
            if len(selected) >= n:
                break
            group = by_affiliation[affil]
            idx = idx_per_affil[affil]
            if idx < len(group):
                selected.append(group[idx])
                idx_per_affil[affil] = idx + 1
                added_any = True
        if not added_any:
            break

    return selected


COMPARE_SYSTEM_PROMPT = (
    "You compare article relevance. For each pair, decide which article is "
    "MORE RELEVANT to the given topic. Articles are shown in random order.\n\n"
    'Respond with JSON array: [{"pair": <int>, "winner": "A"|"B"|"tie", '
    '"reason": "<brief>"}]'
)


async def compare_batch(
    llm,
    maybe_items: list[ContentItem],
    references: list[ContentItem],
    topic: TopicConfig,
) -> list[tuple[bool, str]]:
    """Compare MAYBE items against references via LLM pairwise judgments.

    Each MAYBE item is compared against each reference. Majority wins → promoted.
    Ties count in favor of the MAYBE item (benefit of the doubt).

    Returns list of (promoted: bool, reason: str) per MAYBE item.
    """
    results: list[tuple[bool, str]] = []

    for maybe_item in maybe_items:
        wins = 0
        reasons: list[str] = []

        for ref in references:
            # MAYBE always in position A; majority voting across multiple
            # references washes out positional bias (see plan notes).
            a_item, b_item = maybe_item, ref
            maybe_pos = "A"

            user_prompt = (
                f"Topic: {topic.name}\n"
                f"Subtopics: {', '.join(topic.subtopics)}\n\n"
                f"Pair 1:\n"
                f'  Article A: "{a_item.title}" — {(a_item.snippet or "")[:200]}\n'
                f'  Article B: "{b_item.title}" — {(b_item.snippet or "")[:200]}\n'
            )

            try:
                response = await llm.complete(
                    config_key="filtering",
                    system_prompt=COMPARE_SYSTEM_PROMPT,
                    user_prompt=user_prompt,
                    json_response=True,
                )
                data = json.loads(response)
                if isinstance(data, list) and data:
                    winner = data[0].get("winner", "").upper()
                    reason = data[0].get("reason", "")
                    reasons.append(reason)

                    if winner == "TIE":
                        wins += 1  # ties favor MAYBE
                    elif winner == maybe_pos:
                        wins += 1
                else:
                    reasons.append("Invalid response format")
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
                logger.warning(f"Pairwise compare parse error: {e}")
                reasons.append(f"Parse error: {e}")

        # Majority vote: need > half of comparisons
        promoted = wins >= (len(references) + 1) // 2  # majority threshold
        reason_summary = "; ".join(reasons[:3])
        results.append((promoted, f"{wins}/{len(references)} wins — {reason_summary}"))

    return results


FALLBACK_ROUNDS = 3  # Swiss-style rounds for degenerate fallback
FALLBACK_BATCH_SIZE = 8  # pairs per LLM call in fallback


async def _compare_pairs(
    llm,
    pairs: list[tuple[ContentItem, ContentItem]],
    topic: TopicConfig,
) -> list[str]:
    """Compare a batch of pairs, return list of winners ("A", "B", or "tie")."""
    if not pairs:
        return []

    pair_texts = []
    for i, (a, b) in enumerate(pairs, 1):
        pair_texts.append(
            f"Pair {i}:\n"
            f'  Article A: "{a.title}" — {(a.snippet or "")[:200]}\n'
            f'  Article B: "{b.title}" — {(b.snippet or "")[:200]}\n'
        )

    user_prompt = (
        f"Topic: {topic.name}\n"
        f"Subtopics: {', '.join(topic.subtopics)}\n\n"
        + "\n".join(pair_texts)
    )

    try:
        response = await llm.complete(
            config_key="filtering",
            system_prompt=COMPARE_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            json_response=True,
        )
        data = json.loads(response)
        if not isinstance(data, list):
            data = [data]

        # Build pair_num -> winner map
        winner_map = {}
        for entry in data:
            pair_num = int(entry.get("pair", 0))
            winner_map[pair_num] = entry.get("winner", "tie").upper()

        return [winner_map.get(i + 1, "TIE") for i in range(len(pairs))]

    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        logger.warning(f"Fallback pair compare error: {e}")
        return ["TIE"] * len(pairs)


def _swiss_pairings(
    items: list[ContentItem],
    wins: dict[str, int],
) -> list[tuple[ContentItem, ContentItem]]:
    """Generate Swiss-style pairings: match items with similar win counts."""
    # Sort by wins (descending), then pair adjacent items
    sorted_items = sorted(items, key=lambda x: wins.get(x.url, 0), reverse=True)
    pairs = []
    for i in range(0, len(sorted_items) - 1, 2):
        pairs.append((sorted_items[i], sorted_items[i + 1]))
    return pairs


async def pairwise_fallback(
    llm,
    items: list[ContentItem],
    topic: TopicConfig,
    max_keep: int = 30,
) -> list[ContentItem]:
    """Full pairwise ranking for degenerate distributions.

    Swiss-style pairing over FALLBACK_ROUNDS rounds. Each round:
    1. Pair items by similar win count
    2. Compare pairs via LLM (batched)
    3. Accumulate wins

    Returns top max_keep items by win count.
    """
    if not items:
        return []

    if len(items) <= max_keep:
        return list(items)

    wins: dict[str, int] = {item.url: 0 for item in items}

    for round_num in range(FALLBACK_ROUNDS):
        pairs = _swiss_pairings(items, wins)
        if not pairs:
            break

        # Batch pairs into LLM calls
        for batch_start in range(0, len(pairs), FALLBACK_BATCH_SIZE):
            batch = pairs[batch_start:batch_start + FALLBACK_BATCH_SIZE]
            results = await _compare_pairs(llm, batch, topic)

            for (a, b), winner in zip(batch, results):
                if winner == "A":
                    wins[a.url] = wins.get(a.url, 0) + 1
                elif winner == "B":
                    wins[b.url] = wins.get(b.url, 0) + 1
                else:  # tie
                    wins[a.url] = wins.get(a.url, 0) + 1
                    wins[b.url] = wins.get(b.url, 0) + 1

        logger.debug(f"Fallback round {round_num + 1}: top wins = {sorted(wins.values(), reverse=True)[:5]}")

    # Rank by wins, return top max_keep
    ranked = sorted(items, key=lambda x: wins.get(x.url, 0), reverse=True)
    result = ranked[:max_keep]
    logger.info(f"Pairwise fallback: {len(result)}/{len(items)} items selected over {FALLBACK_ROUNDS} rounds")
    return result


async def resolve_maybe_items(
    llm,
    maybe_items: list[ContentItem],
    keep_items: list[ContentItem],
    topic: TopicConfig,
) -> list[ContentItem]:
    """Orchestrate pairwise resolution of MAYBE items.

    If no KEEP items exist, return all MAYBE items (can't compare without refs).
    Otherwise, compare each MAYBE against selected references, return promoted items.
    """
    if not maybe_items:
        return []

    if not keep_items:
        return list(maybe_items)

    refs = select_references(keep_items, n=PAIRWISE_COMPARISONS)
    results = await compare_batch(llm, maybe_items, refs, topic)

    promoted = []
    for item, (is_promoted, reason) in zip(maybe_items, results):
        if is_promoted:
            promoted.append(item)
            logger.debug(f"Pairwise promoted: {item.title} ({reason})")
        else:
            logger.debug(f"Pairwise dropped: {item.title} ({reason})")

    logger.info(f"Pairwise resolution: {len(promoted)}/{len(maybe_items)} MAYBE items promoted")
    return promoted
