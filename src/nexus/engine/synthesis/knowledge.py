"""Knowledge synthesis — build the intermediate TopicSynthesis object (X).

This is the core intellectual product of the pipeline. All artifacts (briefing,
podcast, dashboard) render from TopicSynthesis, not from raw events.
"""

import json
import logging
from datetime import date
from typing import Any, Optional, Union

from pydantic import BaseModel, Field

from nexus.config.models import TopicConfig
from nexus.engine.knowledge.compression import Summary
from nexus.engine.knowledge.events import Event
from nexus.engine.knowledge.store import KnowledgeStore
from nexus.engine.sources.polling import ContentItem
from nexus.engine.synthesis.threads import (
    match_events_to_threads, create_thread_slug, promote_thread_status,
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


class TopicSynthesis(BaseModel):
    """The structured knowledge product for one topic."""
    topic_name: str
    threads: list[NarrativeThread] = Field(default_factory=list)
    background: list[Summary] = Field(default_factory=list)
    source_balance: dict = Field(default_factory=dict)  # {affiliation: count}
    languages_represented: list[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)


_SYNTHESIS_BASE_PROMPT = (
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
    "3. Identify divergence: where two DIFFERENT outlets report on the SAME event "
    "with genuinely conflicting framing, emphasis, or interpretation. "
    "Do NOT flag different coverage areas or different topics as divergence. "
    "Only flag when outlets disagree on the same underlying event or claim.\n"
    "4. Note key entities involved\n"
    "5. Rate significance (1-10)\n\n"
)

_SYNTHESIS_OUTPUT_FORMAT = (
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
    "Return an empty list for divergence if no genuine framing conflicts exist. "
    "Quality over quantity — only include well-supported entries."
)

# Keep backward-compatible reference for any external imports
SYNTHESIS_SYSTEM_PROMPT = _SYNTHESIS_BASE_PROMPT + _SYNTHESIS_OUTPUT_FORMAT


def _build_synthesis_prompt(topic: TopicConfig) -> str:
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

    return _SYNTHESIS_BASE_PROMPT + scope_instruction + _SYNTHESIS_OUTPUT_FORMAT


async def synthesize_topic(
    llm: LLMClient,
    topic: TopicConfig,
    events: list[Event],
    articles: list[ContentItem],
    weekly_summaries: list[Summary],
    monthly_summaries: list[Summary],
    store: KnowledgeStore | None = None,
    topic_slug: str | None = None,
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
    event_lines = []
    for i, e in enumerate(events):
        sources_str = ", ".join(
            f"{s.get('outlet', '?')} ({s.get('affiliation', '?')}/{s.get('country', '?')})"
            for s in e.sources
        )
        event_lines.append(
            f"[Event {i}] [{e.date}] (sig:{e.significance}) {e.summary}\n"
            f"  Entities: {', '.join(e.entities)}\n"
            f"  Sources: {sources_str}"
        )

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
        f"Events to analyze:\n" + "\n".join(event_lines)
    )

    try:
        response = await llm.complete(
            config_key="knowledge_summary",
            system_prompt=_build_synthesis_prompt(topic),
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

            # Update thread with persistence info
            thread.thread_id = tid
            thread.slug = slug
            thread.status = status

            # Persist convergence/divergence
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

    except Exception as e:
        logger.warning(f"Thread persistence failed (non-blocking): {e}")
