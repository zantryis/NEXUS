"""Knowledge synthesis — build the intermediate TopicSynthesis object (X).

This is the core intellectual product of the pipeline. All artifacts (briefing,
podcast, dashboard) render from TopicSynthesis, not from raw events.
"""

import json
import logging
from typing import Optional, Union

from pydantic import BaseModel, Field

from nexus.config.models import TopicConfig
from nexus.engine.knowledge.compression import Summary
from nexus.engine.knowledge.events import Event, are_independent
from nexus.engine.projection.models import CrossTopicSignal, TopicProjection
from nexus.engine.knowledge.store import KnowledgeStore
from nexus.engine.sources.polling import ContentItem
from nexus.engine.synthesis.threads import (
    match_events_to_threads, create_thread_slug, promote_thread_status,
    find_merge_candidates,
)
from nexus.llm.client import LLMClient

logger = logging.getLogger(__name__)


class SourceClaim(BaseModel):
    """A factual claim with its source attribution."""
    text: str
    source_id: str
    source_affiliation: str = ""
    source_country: str = ""
    source_language: str = ""
    confidence: str = "reported"  # confirmed | reported | alleged


class NarrativeThread(BaseModel):
    """A coherent story arc across multiple events."""
    headline: str
    events: list[Event] = Field(default_factory=list)
    convergence: list[Union[str, dict]] = Field(default_factory=list)  # Facts multiple sources agree on
    divergence: list[dict] = Field(default_factory=list)   # Conflicting framings
    key_entities: list[str] = Field(default_factory=list)
    significance: int = 5
    # Persistence fields (optional — None when store not used)
    thread_id: Optional[int] = None
    slug: Optional[str] = None
    status: Optional[str] = None
    velocity_7d: Optional[float] = None
    acceleration_7d: Optional[float] = None
    significance_trend_7d: Optional[float] = None
    momentum_score: Optional[float] = None
    trajectory_label: Optional[str] = None
    snapshot_count: Optional[int] = None


class TopicSynthesis(BaseModel):
    """The structured knowledge product for one topic."""
    topic_name: str
    threads: list[NarrativeThread] = Field(default_factory=list)
    background: list[Summary] = Field(default_factory=list)
    source_balance: dict = Field(default_factory=dict)  # {affiliation: count}
    languages_represented: list[str] = Field(default_factory=list)
    cross_topic_signals: list[CrossTopicSignal] = Field(default_factory=list)
    projection: TopicProjection | None = None
    metadata: dict = Field(default_factory=dict)


_SYNTHESIS_PRE_DIVERGENCE = (
    "You are a knowledge synthesis engine. Given events, articles, and topic context, "
    "produce a structured analysis.\n\n"
    "## Source affiliations\n"
    "Each event lists its sources with affiliation and country metadata:\n"
    "- state: Government-controlled editorial (e.g., CGTN/CN, TASS/RU, Al Jazeera/QA). "
    "May reflect government positions.\n"
    "- public: Publicly funded, editorially independent (e.g., BBC/GB, DW/DE, NHK/JP). "
    "Generally fact-based but may carry cultural lens.\n"
    "- private: Corporate/private ownership (e.g., NYT/US, Guardian/GB, SCMP/HK). "
    "Editorial line varies by outlet.\n"
    "- nonprofit/academic: Mission-driven, specialized coverage.\n\n"
    "Use affiliation + country to assess editorial independence. "
    "Two state outlets from the same country are NOT independent sources.\n\n"
    "## Instructions\n"
    "For each narrative thread you identify:\n"
    "1. Group related events into coherent story arcs\n"
    "2. Identify convergence: facts confirmed by 2+ INDEPENDENT sources "
    "(different outlets, ideally different affiliations or countries). "
    "If all events in a thread come from the SAME outlet, convergence MUST be empty.\n"
)

_DIVERGENCE_INSTRUCTIONS_DEFAULT = (
    "3. Identify divergence: where two DIFFERENT outlets covering the SAME event "
    "frame it differently. Each source now has structured framing data: "
    "[editorial_tone] editorial_focus; actor_framing. "
    "Compare these across sources in each thread. Look for:\n"
    "  a) TONE CONTRAST: Different editorial tones (e.g., one is 'alarmist', another 'dismissive')\n"
    "  b) FOCUS DIVERGENCE: Different aspects emphasized (e.g., humanitarian vs military)\n"
    "  c) ACTOR FRAMING: Different characterization of the same actors (e.g., 'aggressors' vs 'defenders', "
    "'killed' vs 'dead', 'militants' vs 'fighters', active vs passive voice)\n"
    "  d) OMISSION: Key context one outlet includes but another leaves out\n"
    "For each divergence entry, note the category in the shared_event field as a prefix, "
    "e.g., '[TONE CONTRAST] Israeli ground incursion into Lebanon'.\n"
    "Do NOT flag coverage of entirely different stories as divergence.\n"
)

_SYNTHESIS_POST_DIVERGENCE = (
    "4. Note key entities involved\n"
    "5. Rate significance (1-10)\n\n"
    "## Thread consolidation\n"
    "Before creating a new thread, check if any existing thread covers substantially "
    "the same narrative. Threads sharing 3+ entities and describing the same causal "
    "chain MUST be merged into one. Prefer fewer, richer threads over many sparse ones.\n\n"
)

_SYNTHESIS_BASE_PROMPT = _SYNTHESIS_PRE_DIVERGENCE + _DIVERGENCE_INSTRUCTIONS_DEFAULT + _SYNTHESIS_POST_DIVERGENCE

_SYNTHESIS_OUTPUT_FORMAT_PRE = (
    "## Output JSON\n"
    "{\n"
    '  "threads": [\n'
    "    {\n"
    '      "headline": "Short thread title",\n'
    '      "event_indices": [0, 2, 5],\n'
    '      "convergence": [\n'
    '        {"fact": "Fact agreed by multiple sources", '
    '"confirmed_by": ["outlet1", "outlet2"]}\n'
    "      ],\n"
    '      "divergence": [\n'
    '        {"shared_event": "The event both outlets are reporting on", '
    '"source_a": "outlet1", "framing_a": "How outlet1 frames it", '
    '"source_b": "outlet2", "framing_b": "How outlet2 frames it"}\n'
    "      ],\n"
    '      "key_entities": ["Entity1", "Entity2"],\n'
    '      "significance": 8\n'
    "    }\n"
    "  ]\n"
    "}\n\n"
    "IMPORTANT: Return an empty list for convergence if sources are not independent. "
)

_DIVERGENCE_OUTPUT_QUALIFIER_DEFAULT = (
    "For divergence, systematically check each category (tone contrast, focus divergence, "
    "actor framing, omission) for every thread with 2+ independent sources. "
    "Compare the structured framing data ([tone] focus; actors) across sources. "
    "Most multi-source threads will have divergence — if you find none, re-examine. "
)

_SYNTHESIS_OUTPUT_FORMAT_POST = (
    "Quality over quantity — only include well-supported entries."
)

_SYNTHESIS_OUTPUT_FORMAT = (
    _SYNTHESIS_OUTPUT_FORMAT_PRE + _DIVERGENCE_OUTPUT_QUALIFIER_DEFAULT + _SYNTHESIS_OUTPUT_FORMAT_POST
)

# Keep backward-compatible reference for any external imports
SYNTHESIS_SYSTEM_PROMPT = _SYNTHESIS_BASE_PROMPT + _SYNTHESIS_OUTPUT_FORMAT


# ── Divergence Prompt Variants (for experiment Suite H) ──────────────────────

DIVERGENCE_VARIANTS = {
    "baseline": {
        "instructions": (
            "3. Identify divergence: where two DIFFERENT outlets report on the SAME event "
            "with genuinely conflicting framing, emphasis, or interpretation. "
            "Do NOT flag different coverage areas or different topics as divergence. "
            "Only flag when outlets disagree on the same underlying event or claim.\n"
        ),
        "output_qualifier": (
            "Return an empty list for divergence if no genuine framing conflicts exist. "
        ),
    },
    "broadened": {
        "instructions": (
            "3. Identify divergence: where two DIFFERENT outlets covering the SAME event or "
            "development use notably different framing, emphasis, tone, or interpretation. "
            "Divergence includes:\n"
            "  - Different causal explanations for the same event\n"
            "  - Different emphasis (e.g., one leads with casualties, another with geopolitics)\n"
            "  - Different characterization of actors (e.g., 'aggressor' vs 'defender')\n"
            "  - Selective omission of context that another outlet includes\n"
            "Do NOT flag coverage of entirely different stories as divergence.\n"
        ),
        "output_qualifier": (
            "For divergence, actively look for framing differences between outlets covering "
            "the same events. Most multi-source threads SHOULD have at least one divergence entry. "
        ),
    },
    "structured": {
        "instructions": (
            "3. Identify divergence: where two DIFFERENT outlets covering the SAME event "
            "frame it differently. Check each of these divergence categories:\n"
            "  a) FRAMING: Different narrative frames (e.g., 'security operation' vs 'invasion')\n"
            "  b) EMPHASIS: Different lead angles (e.g., military vs humanitarian vs economic)\n"
            "  c) OMISSION: Key context one outlet includes but another leaves out\n"
            "  d) TONE: Editorial tone difference (e.g., neutral reporting vs alarm/condemnation)\n"
            "  e) ATTRIBUTION: Different sources quoted or credited for the same claim\n"
            "For each divergence entry, note the category in the shared_event field as a prefix, "
            "e.g., '[FRAMING] Israeli ground incursion into Lebanon'.\n"
            "Do NOT flag coverage of entirely different stories as divergence.\n"
        ),
        "output_qualifier": (
            "For divergence, systematically check each category (framing, emphasis, omission, "
            "tone, attribution) for every thread with 2+ independent sources. "
            "Most multi-source threads will have divergence. "
        ),
    },
    "encouraged": {
        "instructions": (
            "3. Identify divergence: where two DIFFERENT outlets covering the SAME event "
            "or development present it differently. Sources from different countries and "
            "affiliations ALMOST ALWAYS frame events differently — your job is to find and "
            "articulate these differences. Look for:\n"
            "  - How each outlet characterizes the actors and their motivations\n"
            "  - What each outlet emphasizes vs downplays\n"
            "  - What context each outlet provides or omits\n"
            "  - Differences in tone (alarm, neutrality, approval, condemnation)\n"
            "If a thread has sources from different affiliations/countries covering the same "
            "event, there is ALMOST CERTAINLY divergence — look harder.\n"
            "Do NOT flag coverage of entirely different stories as divergence.\n"
        ),
        "output_qualifier": (
            "For divergence, every thread with sources from different affiliations or countries "
            "should have at least one divergence entry. If you find none, re-examine — framing "
            "differences between state/private/public outlets are nearly always present. "
        ),
    },
}


def _build_synthesis_prompt(
    topic: TopicConfig,
    divergence_instructions: str | None = None,
    divergence_output_qualifier: str | None = None,
) -> str:
    """Build a scope-aware synthesis system prompt."""
    scope = getattr(topic, "scope", "medium")
    scope_instruction = ""

    if scope == "broad":
        subtopics_str = ", ".join(topic.subtopics) if topic.subtopics else "various subfields"
        scope_instruction = (
            "\n## Topic scope: BROAD\n"
            f"This topic covers multiple distinct subfields: {subtopics_str}. "
            "Group events into subfield-specific threads. "
            "Do NOT merge unrelated subfields (e.g., 'AI agents' and 'protein folding') "
            "into a single thread. Each thread should correspond to a coherent subfield "
            "or cross-cutting development.\n\n"
        )
    elif scope == "narrow":
        scope_instruction = (
            "\n## Topic scope: FOCUSED\n"
            "This is a focused topic where events are likely interrelated. "
            "Look for causal chains and temporal progression across threads. "
            "Threads should reflect distinct story arcs within the same domain.\n\n"
        )

    div_instr = divergence_instructions or _DIVERGENCE_INSTRUCTIONS_DEFAULT
    div_qual = divergence_output_qualifier or _DIVERGENCE_OUTPUT_QUALIFIER_DEFAULT

    base = _SYNTHESIS_PRE_DIVERGENCE + div_instr + _SYNTHESIS_POST_DIVERGENCE
    output_fmt = _SYNTHESIS_OUTPUT_FORMAT_PRE + div_qual + _SYNTHESIS_OUTPUT_FORMAT_POST

    return base + scope_instruction + output_fmt


def _validate_convergence(threads: list[NarrativeThread]) -> None:
    """Strip convergence entries where confirmed_by sources are not independent. Modifies in-place."""
    for thread in threads:
        if not thread.convergence:
            continue
        # Build outlet → source metadata map from all events in thread
        outlet_meta: dict[str, dict] = {}
        for event in thread.events:
            for s in event.sources:
                outlet = s.get("outlet", "")
                if outlet and outlet not in outlet_meta:
                    outlet_meta[outlet] = s

        validated = []
        for c in thread.convergence:
            if not isinstance(c, dict):
                validated.append(c)  # Legacy string format — keep
                continue
            confirmed = c.get("confirmed_by", [])
            if len(confirmed) < 2:
                continue  # Single-source convergence invalid by definition
            # Check for at least one independent pair
            has_pair = any(
                are_independent(outlet_meta.get(confirmed[i], {}), outlet_meta.get(confirmed[j], {}))
                for i in range(len(confirmed)) for j in range(i + 1, len(confirmed))
            )
            if has_pair:
                validated.append(c)
            else:
                logger.debug(f"Stripped non-independent convergence: {c.get('fact', '')[:60]}")
        thread.convergence = validated


def _check_thread_overlaps(threads: list[NarrativeThread]) -> list[tuple[int, int, float]]:
    """Check all thread pairs for entity overlap. Returns pairs with Jaccard > 0.5."""
    overlaps = []
    for i in range(len(threads)):
        set_i = {e.lower() for e in threads[i].key_entities}
        if not set_i:
            continue
        for j in range(i + 1, len(threads)):
            set_j = {e.lower() for e in threads[j].key_entities}
            if not set_j:
                continue
            intersection = set_i & set_j
            union = set_i | set_j
            jaccard = len(intersection) / len(union)
            if jaccard > 0.5:
                overlaps.append((i, j, round(jaccard, 3)))
                logger.warning(
                    f"Thread overlap detected: [{i}] '{threads[i].headline[:40]}' "
                    f"↔ [{j}] '{threads[j].headline[:40]}' "
                    f"(Jaccard={jaccard:.2f}, shared: {intersection})"
                )
    return overlaps


def _format_events(events: list[Event]) -> str:
    """Format events for the synthesis LLM prompt, including per-source framing."""
    event_lines = []
    for i, e in enumerate(events):
        source_parts = []
        for s in e.sources:
            line = f"    - {s.get('outlet', '?')} ({s.get('affiliation', '?')}/{s.get('country', '?')})"
            framing = s.get("framing", "")
            if framing:
                line += f": {framing}"
            source_parts.append(line)
        sources_block = "\n".join(source_parts) if source_parts else "    - (none)"
        event_lines.append(
            f"[Event {i}] [{e.date}] (sig:{e.significance}) {e.summary}\n"
            f"  Entities: {', '.join(e.entities)}\n"
            f"  Sources:\n{sources_block}"
        )
    return "\n".join(event_lines)


def _build_article_snippets(events: list[Event], articles: list[ContentItem]) -> str:
    """Build supplementary article excerpts for multi-source events."""
    url_to_article = {a.url: a for a in articles}
    lines = []
    for i, e in enumerate(events):
        if len(e.sources) < 2:
            continue
        for s in e.sources:
            article = url_to_article.get(s.get("url"))
            if article and article.full_text:
                text = article.full_text[:200].replace("\n", " ")
                lines.append(f"[Event {i}] {s.get('outlet', '?')}: {article.title} -- {text}...")
    return "\n".join(lines) if lines else ""


async def synthesize_topic(
    llm: LLMClient,
    topic: TopicConfig,
    events: list[Event],
    articles: list[ContentItem],
    weekly_summaries: list[Summary],
    monthly_summaries: list[Summary],
    store: KnowledgeStore | None = None,
    topic_slug: str | None = None,
    divergence_instructions: str | None = None,
    divergence_output_qualifier: str | None = None,
) -> TopicSynthesis:
    """Build the TopicSynthesis knowledge object via LLM analysis."""
    # Build source balance from articles
    source_balance: dict[str, int] = {}
    languages: set[str] = set()
    for article in articles:
        affil = article.source_affiliation or "unknown"
        source_balance[affil] = source_balance.get(affil, 0) + 1
        lang = article.detected_language or article.source_language
        if lang:
            languages.add(lang)

    if not events:
        return TopicSynthesis(
            topic_name=topic.name,
            background=weekly_summaries + monthly_summaries,
            source_balance=source_balance,
            languages_represented=sorted(languages),
            metadata={"event_count": 0, "article_count": len(articles)},
        )

    # Format events for LLM
    event_lines_str = _format_events(events)

    # Background context
    bg_lines = []
    for s in (weekly_summaries or [])[-3:]:
        bg_lines.append(f"- Week {s.period_start}–{s.period_end}: {s.text}")
    for s in (monthly_summaries or [])[-1:]:
        bg_lines.append(f"- Month {s.period_start}–{s.period_end}: {s.text}")

    user_prompt = (
        f"Topic: {topic.name}\n"
        f"Subtopics: {', '.join(topic.subtopics)}\n\n"
        f"Background:\n{chr(10).join(bg_lines) or 'None'}\n\n"
        f"Events to analyze:\n{event_lines_str}"
    )

    # Append article snippets for multi-source divergence analysis
    snippets = _build_article_snippets(events, articles)
    if snippets:
        user_prompt += f"\n\n## Source article excerpts (for divergence analysis)\n{snippets}"

    try:
        response = await llm.complete(
            config_key="knowledge_summary",
            system_prompt=_build_synthesis_prompt(
                topic, divergence_instructions, divergence_output_qualifier,
            ),
            user_prompt=user_prompt,
            json_response=True,
        )
        data = json.loads(response)

        # Handle LLM returning a list of threads directly instead of {"threads": [...]}
        if isinstance(data, list):
            data = {"threads": data}

        threads = []
        for t in data.get("threads", []):
            # Map event indices to actual events
            indices = t.get("event_indices", [])
            thread_events = [events[i] for i in indices if i < len(events)]

            threads.append(NarrativeThread(
                headline=t["headline"],
                events=thread_events,
                convergence=t.get("convergence", []),
                divergence=t.get("divergence", []),
                key_entities=t.get("key_entities", []),
                significance=int(t.get("significance", 5)),
            ))

        # Post-synthesis validation
        _validate_convergence(threads)
        _check_thread_overlaps(threads)

        synthesis = TopicSynthesis(
            topic_name=topic.name,
            threads=threads,
            background=weekly_summaries + monthly_summaries,
            source_balance=source_balance,
            languages_represented=sorted(languages),
            metadata={
                "event_count": len(events),
                "article_count": len(articles),
                "thread_count": len(threads),
            },
        )

    except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
        logger.warning(f"Knowledge synthesis failed for {topic.name}: {e}")
        # Fallback: one thread per event
        threads = [
            NarrativeThread(
                headline=event.summary[:80],
                events=[event],
                key_entities=event.entities,
                significance=event.significance,
            )
            for event in events[:10]
        ]
        synthesis = TopicSynthesis(
            topic_name=topic.name,
            threads=threads,
            background=weekly_summaries + monthly_summaries,
            source_balance=source_balance,
            languages_represented=sorted(languages),
            metadata={"event_count": len(events), "article_count": len(articles), "fallback": True},
        )

    # Persist threads to store if available
    if store and topic_slug and synthesis.threads:
        await _persist_threads(store, llm, synthesis, events, topic_slug)

    return synthesis


async def _consolidate_threads(
    store: KnowledgeStore,
    llm: LLMClient,
    topic_slug: str,
) -> list[tuple[int, int]]:
    """Post-synthesis thread deduplication. Returns merge pairs executed."""
    active = await store.get_active_threads(topic_slug)
    if len(active) < 2:
        return []

    pairs = await find_merge_candidates(active, llm)
    for keep_id, absorb_id in pairs:
        logger.info(f"Auto-merging thread {absorb_id} → {keep_id}")
        await store.merge_threads(keep_id, absorb_id)

    return pairs


async def _persist_threads(
    store: KnowledgeStore,
    llm: LLMClient,
    synthesis: TopicSynthesis,
    events: list[Event],
    topic_slug: str,
) -> None:
    """Match synthesis threads to existing persistent threads and save."""
    try:
        active_threads = await store.get_active_threads(topic_slug)

        # Match new events to existing threads
        all_new_events = []
        for thread in synthesis.threads:
            all_new_events.extend(thread.events)

        if all_new_events:
            matches = await match_events_to_threads(llm, all_new_events, active_threads)
        else:
            matches = []

        # Build a map: thread headline → match info
        thread_slugs: dict[str, str] = {}  # headline → slug
        for match in matches:
            if match.is_new_thread and match.new_headline:
                thread_slugs[match.new_headline] = match.thread_slug
            elif match.thread_slug:
                # Find headline from active_threads
                for at in active_threads:
                    if at["slug"] == match.thread_slug:
                        thread_slugs[at["headline"]] = match.thread_slug
                        break

        # Persist each synthesis thread
        for thread in synthesis.threads:
            slug = thread_slugs.get(thread.headline) or create_thread_slug(thread.headline)
            event_dates = [e.date for e in thread.events]
            status = promote_thread_status("emerging", event_dates) if event_dates else "emerging"

            # Check if this matches an existing thread
            existing = next((t for t in active_threads if t["slug"] == slug), None)
            if existing:
                status = promote_thread_status(existing["status"], event_dates)

            tid = await store.upsert_thread(slug, thread.headline, thread.significance, status)
            await store.link_thread_topic(tid, topic_slug)

            # Link thread to its events by matching (summary, date, topic_slug)
            event_ids = []
            for ev in thread.events:
                ev_date = ev.date if isinstance(ev.date, str) else ev.date.isoformat()
                eid = await store.find_event_id(ev.summary, ev_date, topic_slug)
                if eid:
                    event_ids.append(eid)
            if event_ids:
                await store.link_thread_events(tid, event_ids)

            # Update thread with persistence info
            thread.thread_id = tid
            thread.slug = slug
            thread.status = status

            # Persist convergence/divergence
            await store.clear_thread_analysis(tid)
            for c in thread.convergence:
                if isinstance(c, dict):
                    await store.add_convergence(
                        tid, c.get("fact", str(c)),
                        c.get("confirmed_by", []),
                    )
                elif isinstance(c, str):
                    await store.add_convergence(tid, c, [])

            for d in thread.divergence:
                if isinstance(d, dict):
                    await store.add_divergence(
                        tid,
                        d.get("shared_event", ""),
                        d.get("source_a", ""),
                        d.get("framing_a", ""),
                        d.get("source_b", ""),
                        d.get("framing_b", ""),
                    )

        # Post-synthesis: consolidate overlapping threads
        merged = await _consolidate_threads(store, llm, topic_slug)
        if merged:
            logger.info(f"Consolidated {len(merged)} overlapping thread pairs for {topic_slug}")

    except Exception as e:
        logger.warning(f"Thread persistence failed (non-blocking): {e}")
