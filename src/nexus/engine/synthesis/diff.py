"""Compute diffs between TopicSynthesis snapshots — "what changed since last briefing"."""

from nexus.engine.synthesis.knowledge import TopicSynthesis


def diff_syntheses(current: TopicSynthesis, previous: TopicSynthesis) -> dict:
    """Compare two TopicSynthesis objects and return a structured diff.

    Returns:
        {
            "new_threads": [NarrativeThread, ...],
            "updated_threads": [{"thread": NarrativeThread, "new_events": [...], "prev_event_count": int}, ...],
            "resolved_threads": [headline, ...],
            "new_convergence": [{"thread_headline": str, "facts": [...]}, ...],
            "new_divergence": [{"thread_headline": str, "items": [...]}, ...],
            "new_entities": [str, ...],
            "source_balance_shift": {affiliation: delta, ...},
        }
    """
    prev_headlines = {_norm(t.headline) for t in previous.threads}
    prev_thread_map = {_norm(t.headline): t for t in previous.threads}
    curr_headlines = {_norm(t.headline) for t in current.threads}

    new_threads = []
    updated_threads = []
    new_convergence = []
    new_divergence = []

    for thread in current.threads:
        key = _norm(thread.headline)
        if key not in prev_headlines:
            new_threads.append(thread)
            # All convergence/divergence in new threads is new
            if thread.convergence:
                new_convergence.append({
                    "thread_headline": thread.headline,
                    "facts": thread.convergence,
                })
            if thread.divergence:
                new_divergence.append({
                    "thread_headline": thread.headline,
                    "items": thread.divergence,
                })
        else:
            prev_thread = prev_thread_map[key]
            prev_summaries = {e.summary for e in prev_thread.events}
            new_events = [e for e in thread.events if e.summary not in prev_summaries]

            if new_events:
                updated_threads.append({
                    "thread": thread,
                    "new_events": new_events,
                    "prev_event_count": len(prev_thread.events),
                })

            # Diff convergence
            prev_conv_facts = _convergence_keys(prev_thread.convergence)
            new_conv = [c for c in thread.convergence
                        if _convergence_key(c) not in prev_conv_facts]
            if new_conv:
                new_convergence.append({
                    "thread_headline": thread.headline,
                    "facts": new_conv,
                })

            # Diff divergence
            prev_div_keys = _divergence_keys(prev_thread.divergence)
            new_div = [d for d in thread.divergence
                       if _divergence_key(d) not in prev_div_keys]
            if new_div:
                new_divergence.append({
                    "thread_headline": thread.headline,
                    "items": new_div,
                })

    # Threads in previous but not current
    resolved_threads = [
        prev_thread_map[key].headline
        for key in prev_headlines - curr_headlines
    ]

    # Entity diff
    prev_entities = set()
    for t in previous.threads:
        prev_entities.update(t.key_entities)
    curr_entities = set()
    for t in current.threads:
        curr_entities.update(t.key_entities)
    new_entities = sorted(curr_entities - prev_entities)

    # Source balance shift
    source_balance_shift = {}
    for affil, count in current.source_balance.items():
        prev_count = previous.source_balance.get(affil, 0)
        delta = count - prev_count
        if delta != 0:
            source_balance_shift[affil] = delta
    for affil, prev_count in previous.source_balance.items():
        if affil not in current.source_balance:
            source_balance_shift[affil] = -prev_count

    return {
        "new_threads": new_threads,
        "updated_threads": updated_threads,
        "resolved_threads": resolved_threads,
        "new_convergence": new_convergence,
        "new_divergence": new_divergence,
        "new_entities": new_entities,
        "source_balance_shift": source_balance_shift,
    }


def is_empty_diff(diff: dict) -> bool:
    """Check if a diff has no meaningful changes."""
    return (
        not diff["new_threads"]
        and not diff["updated_threads"]
        and not diff["resolved_threads"]
        and not diff["new_convergence"]
        and not diff["new_divergence"]
        and not diff["new_entities"]
    )


def _norm(s: str) -> str:
    """Normalize a headline for comparison."""
    return s.strip().lower()


def _convergence_key(c) -> str:
    if isinstance(c, dict):
        return c.get("fact", "").strip().lower()
    return str(c).strip().lower()


def _convergence_keys(convergence_list: list) -> set[str]:
    return {_convergence_key(c) for c in convergence_list}


def _divergence_key(d: dict) -> str:
    shared = d.get("shared_event", d.get("claim", ""))
    return shared.strip().lower()


def _divergence_keys(divergence_list: list) -> set[str]:
    return {_divergence_key(d) for d in divergence_list}
