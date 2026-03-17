"""Evidence assembly layer — gathers structural KG data for prediction questions.

Zero LLM calls. Pure store queries with leakage-safe date filtering.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import date

logger = logging.getLogger(__name__)

# ── Caps to prevent prompt overflow ──────────────────────────────────
MAX_THREADS = 5
MAX_CONVERGENCE = 10
MAX_DIVERGENCE = 5
MAX_CAUSAL_LINKS = 10
MAX_RELATIONSHIPS = 15
MAX_CROSS_TOPIC = 5
MAX_EVENTS = 15


@dataclass
class EvidencePackage:
    """All structural evidence assembled for a prediction question."""

    question: str
    as_of: date
    entities: list[dict] = field(default_factory=list)
    threads: list[dict] = field(default_factory=list)
    convergence: list[dict] = field(default_factory=list)
    divergence: list[dict] = field(default_factory=list)
    causal_chains: list[dict] = field(default_factory=list)
    relationships: list[dict] = field(default_factory=list)
    relationship_changes: list[dict] = field(default_factory=list)
    cross_topic_signals: list[dict] = field(default_factory=list)
    recent_events: list[dict] = field(default_factory=list)
    coverage: dict = field(default_factory=dict)


# ── Word-boundary matching ───────────────────────────────────────────


def _word_match(term: str, text: str) -> bool:
    """Check if *term* appears as a whole word/phrase in *text* (case-insensitive).

    Uses regex word boundaries to avoid false positives like "US" matching "discuss".
    """
    return bool(re.search(r"\b" + re.escape(term) + r"\b", text, re.IGNORECASE))


# ── Entity resolution (adapted from actor_engine.identify_actors) ────


async def _resolve_entities(store, question: str) -> list[dict]:
    """Find entities whose canonical names appear in the question text.

    Zero LLM calls — pure text matching against the entity store.
    """
    all_entities = await store.get_all_entities()
    if not all_entities:
        return []

    scored: list[tuple[int, dict]] = []

    for entity in all_entities:
        name = entity.get("canonical_name") or entity.get("name", "")
        if not name or len(name) < 2:
            continue
        if _word_match(name, question):
            scored.append((2, {
                "name": name,
                "entity_id": entity.get("id"),
                "entity_type": entity.get("entity_type", ""),
            }))
            continue
        # Check aliases
        aliases = entity.get("aliases") or []
        if isinstance(aliases, str):
            import json
            try:
                aliases = json.loads(aliases)
            except (json.JSONDecodeError, TypeError):
                aliases = [aliases]
        for alias in aliases:
            if isinstance(alias, str) and _word_match(alias, question):
                scored.append((1, {
                    "name": name,
                    "entity_id": entity.get("id"),
                    "entity_type": entity.get("entity_type", ""),
                }))
                break

    # Sort by score descending, deduplicate by entity_id
    scored.sort(key=lambda x: x[0], reverse=True)
    seen_ids: set[int | None] = set()
    result: list[dict] = []
    for _, ent in scored:
        eid = ent.get("entity_id")
        if eid in seen_ids:
            continue
        seen_ids.add(eid)
        result.append(ent)

    return result


# ── Evidence ranking ─────────────────────────────────────────────────


def _rank_events(events: list, *, as_of: date, max_events: int = MAX_EVENTS) -> list[dict]:
    """Rank events by recency × significance, return as dicts capped at max_events."""
    ref = as_of
    scored: list[tuple[float, dict]] = []

    for event in events:
        # Handle both dict-like and object-like events
        if hasattr(event, "date"):
            event_date = event.date if isinstance(event.date, date) else date.fromisoformat(str(event.date))
            summary = getattr(event, "summary", "")
            significance = getattr(event, "significance", 5)
            event_id = getattr(event, "event_id", None)
            entities = getattr(event, "entities", [])
        else:
            try:
                event_date = date.fromisoformat(str(event.get("date", "")))
            except (ValueError, TypeError):
                continue
            summary = event.get("summary", "")
            significance = event.get("significance", 5)
            event_id = event.get("event_id") or event.get("id")
            entities = event.get("entities", [])

        days_ago = max(1, (ref - event_date).days)
        score = float(significance) / days_ago

        scored.append((score, {
            "event_id": event_id,
            "date": event_date.isoformat(),
            "summary": summary,
            "significance": significance,
            "entities": entities,
        }))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [e for _, e in scored[:max_events]]


# ── Main assembly function ───────────────────────────────────────────


async def assemble_evidence_package(
    store,
    question: str,
    *,
    as_of: date,
) -> EvidencePackage:
    """Gather all structural evidence for a prediction question.

    Zero LLM calls. All store queries respect the as_of cutoff to prevent leakage.
    """
    pkg = EvidencePackage(question=question, as_of=as_of)

    # Step 1: Resolve entities from question text
    entities = await _resolve_entities(store, question)
    pkg.entities = entities

    if not entities:
        pkg.coverage = {
            "entities_found": 0,
            "threads_found": 0,
            "events_found": 0,
            "convergence_found": 0,
            "divergence_found": 0,
            "causal_links_found": 0,
            "relationships_found": 0,
            "cross_topic_found": 0,
        }
        return pkg

    # Step 2: For each entity → events, threads, relationships
    all_events = []
    all_threads: list[dict] = []
    all_relationships: list[dict] = []
    all_relationship_changes: list[dict] = []
    seen_event_ids: set = set()
    seen_thread_ids: set = set()

    for ent in entities:
        entity_id = ent.get("entity_id")
        if not entity_id:
            continue

        # Events (NO date filter on store method — must post-filter)
        try:
            raw_events = await store.get_events_for_entity(entity_id)
            for ev in raw_events:
                ev_date = ev.date if isinstance(ev.date, date) else date.fromisoformat(str(ev.date))
                ev_id = getattr(ev, "event_id", None)
                if ev_date <= as_of and ev_id not in seen_event_ids:
                    seen_event_ids.add(ev_id)
                    all_events.append(ev)
        except Exception:
            logger.debug("Failed to get events for entity %s", entity_id, exc_info=True)

        # Threads
        try:
            threads = await store.get_threads_for_entity(entity_id)
            for t in threads:
                tid = t.get("id")
                if tid and tid not in seen_thread_ids:
                    seen_thread_ids.add(tid)
                    all_threads.append(t)
        except Exception:
            logger.debug("Failed to get threads for entity %s", entity_id, exc_info=True)

        # Relationships (leakage-safe: as_of param)
        try:
            rels = await store.get_active_relationships_for_entity(entity_id, as_of=as_of)
            all_relationships.extend(rels)
        except Exception:
            logger.debug("Failed to get relationships for entity %s", entity_id, exc_info=True)

        # Relationship timeline (leakage-safe: reference_date param)
        try:
            timeline = await store.get_relationship_timeline(entity_id, days=14, reference_date=as_of)
            all_relationship_changes.extend(timeline)
        except Exception:
            logger.debug("Failed to get relationship timeline for entity %s", entity_id, exc_info=True)

    # Step 3: For each thread → snapshot, convergence, divergence, causal links
    all_convergence: list[dict] = []
    all_divergence: list[dict] = []
    all_causal: list[dict] = []
    thread_details: list[dict] = []

    for thread in all_threads[:MAX_THREADS]:
        tid = thread.get("id")
        if not tid:
            continue

        # Thread snapshot (leakage-safe: cutoff param)
        try:
            snapshot = await store.get_thread_snapshot_as_of(tid, as_of)
            if snapshot:
                thread_details.append({
                    "headline": thread.get("headline", ""),
                    "status": snapshot.status,
                    "trajectory_label": snapshot.trajectory_label,
                    "momentum_score": snapshot.momentum_score,
                    "velocity_7d": snapshot.velocity_7d,
                    "acceleration_7d": snapshot.acceleration_7d,
                    "significance": snapshot.significance,
                })
            else:
                thread_details.append({
                    "headline": thread.get("headline", ""),
                    "status": thread.get("status", "unknown"),
                    "trajectory_label": None,
                    "momentum_score": None,
                    "velocity_7d": None,
                })
        except Exception:
            logger.debug("Failed to get snapshot for thread %s", tid, exc_info=True)

        # Convergence (no date filter — safe since linked to thread)
        try:
            conv = await store.get_convergence_for_thread(tid)
            all_convergence.extend(conv)
        except Exception:
            logger.debug("Failed to get convergence for thread %s", tid, exc_info=True)

        # Divergence (no date filter — safe since linked to thread)
        try:
            div = await store.get_divergence_for_thread(tid)
            all_divergence.extend(div)
        except Exception:
            logger.debug("Failed to get divergence for thread %s", tid, exc_info=True)

        # Causal links within thread
        try:
            links = await store.get_causal_links_for_thread(tid)
            all_causal.extend(links)
        except Exception:
            logger.debug("Failed to get causal links for thread %s", tid, exc_info=True)

    # Step 4: Cross-topic signals (leakage-safe: cutoff param)
    # We need a topic_slug — derive from first thread or skip
    cross_topic: list = []
    topic_slugs_seen: set[str] = set()
    for ev in all_events[:5]:
        slug = getattr(ev, "topic_slug", None)
        if slug and slug not in topic_slugs_seen:
            topic_slugs_seen.add(slug)
            try:
                signals = await store.get_cross_topic_signals_as_of(slug, as_of)
                cross_topic.extend(signals)
            except Exception:
                logger.debug("Failed to get cross-topic signals for %s", slug, exc_info=True)

    # Step 5: Rank events and apply caps
    pkg.recent_events = _rank_events(all_events, as_of=as_of, max_events=MAX_EVENTS)
    pkg.threads = thread_details[:MAX_THREADS]
    pkg.convergence = all_convergence[:MAX_CONVERGENCE]
    pkg.divergence = all_divergence[:MAX_DIVERGENCE]
    pkg.causal_chains = all_causal[:MAX_CAUSAL_LINKS]
    pkg.relationships = all_relationships[:MAX_RELATIONSHIPS]
    pkg.relationship_changes = all_relationship_changes[:MAX_RELATIONSHIPS]
    pkg.cross_topic_signals = [
        s.model_dump() if hasattr(s, "model_dump") else s
        for s in cross_topic[:MAX_CROSS_TOPIC]
    ]

    # Step 6: Coverage stats
    pkg.coverage = {
        "entities_found": len(entities),
        "threads_found": len(thread_details),
        "events_found": len(pkg.recent_events),
        "convergence_found": len(pkg.convergence),
        "divergence_found": len(pkg.divergence),
        "causal_links_found": len(pkg.causal_chains),
        "relationships_found": len(pkg.relationships),
        "cross_topic_found": len(pkg.cross_topic_signals),
    }

    return pkg


# ── Formatting for LLM prompts ──────────────────────────────────────


def format_evidence_section(pkg: EvidencePackage) -> str:
    """Format an EvidencePackage into structured text for LLM prompts."""
    sections: list[str] = []

    if not pkg.entities and not pkg.threads and not pkg.recent_events:
        sections.append("## Intelligence Evidence\n\nNo monitored intelligence available for this question. "
                        "Assessment relies on world knowledge only.\n")
        return "\n".join(sections)

    sections.append("## Intelligence Evidence")
    sections.append(f"*As of {pkg.as_of.isoformat()}*\n")

    # Thread trajectories
    if pkg.threads:
        sections.append("### Thread Trajectories")
        for t in pkg.threads:
            trajectory = t.get("trajectory_label") or "unknown"
            momentum = t.get("momentum_score")
            velocity = t.get("velocity_7d")
            line = f"- **{t.get('headline', 'Unknown')}**: {trajectory}"
            if momentum is not None:
                line += f" (momentum={momentum:.1f}"
                if velocity is not None:
                    line += f", velocity={velocity:.1f}"
                line += ")"
            sections.append(line)
        sections.append("")

    # Recent events (with temporal markers so LLM can weight recency)
    if pkg.recent_events:
        sections.append("### Recent Events")
        for ev in pkg.recent_events:
            sig = ev.get("significance", "?")
            ev_date_str = ev.get("date", "?")
            age_tag = ""
            try:
                ev_date = date.fromisoformat(str(ev_date_str))
                days = (pkg.as_of - ev_date).days
                age_tag = f", {days}d ago" if days > 0 else ", today"
            except (ValueError, TypeError):
                pass
            sections.append(f"- [{ev_date_str}{age_tag}] (sig={sig}) {ev.get('summary', '')}")
        sections.append("")

    # Convergence (multi-source confirmed facts)
    if pkg.convergence:
        sections.append("### Convergence (Multi-Source Confirmed)")
        for c in pkg.convergence:
            confirmed_by = c.get("confirmed_by", [])
            sources = ", ".join(confirmed_by) if confirmed_by else "unknown"
            sections.append(f"- {c.get('fact_text', '')} [sources: {sources}]")
        sections.append("")

    # Divergence (conflicting framings)
    if pkg.divergence:
        sections.append("### Divergence (Conflicting Framings)")
        for d in pkg.divergence:
            sections.append(
                f"- {d.get('shared_event', '')}: "
                f"{d.get('source_a', '')} says \"{d.get('framing_a', '')}\" vs "
                f"{d.get('source_b', '')} says \"{d.get('framing_b', '')}\""
            )
        sections.append("")

    # Causal chains
    if pkg.causal_chains:
        sections.append("### Causal Links")
        for link in pkg.causal_chains:
            sections.append(
                f"- Event {link.get('source_event_id')} → Event {link.get('target_event_id')} "
                f"({link.get('relation_type', 'follow_on')}, strength={link.get('strength', '?')})"
            )
            if link.get("evidence_text"):
                sections.append(f"  {link['evidence_text']}")
        sections.append("")

    # Relationships
    if pkg.relationships:
        sections.append("### Active Relationships")
        for r in pkg.relationships:
            src = r.get("source_entity_name", "?")
            tgt = r.get("target_entity_name", "?")
            rel = r.get("relation_type", "related")
            sections.append(f"- {src} ↔ {tgt}: {rel}")
        sections.append("")

    # Cross-topic signals
    if pkg.cross_topic_signals:
        sections.append("### Cross-Topic Signals")
        for s in pkg.cross_topic_signals:
            sections.append(
                f"- {s.get('shared_entity', '?')} bridges "
                f"{s.get('topic_slug', '?')} ↔ {s.get('related_topic_slug', '?')} "
                f"({s.get('signal_type', 'entity_bridge')})"
            )
        sections.append("")

    return "\n".join(sections)
