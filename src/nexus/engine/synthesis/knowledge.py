"""Knowledge synthesis — build the intermediate TopicSynthesis object (X).

This is the core intellectual product of the pipeline. All artifacts (briefing,
podcast, dashboard) render from TopicSynthesis, not from raw events.
"""

import json
import logging
from datetime import date
from typing import Optional

from pydantic import BaseModel, Field

from nexus.config.models import TopicConfig
from nexus.engine.knowledge.compression import Summary
from nexus.engine.knowledge.events import Event
from nexus.engine.sources.polling import ContentItem
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
    convergence: list[str] = Field(default_factory=list)  # Facts multiple sources agree on
    divergence: list[dict] = Field(default_factory=list)   # Conflicting framings
    key_entities: list[str] = Field(default_factory=list)
    significance: int = 5


class TopicSynthesis(BaseModel):
    """The structured knowledge product for one topic."""
    topic_name: str
    threads: list[NarrativeThread] = Field(default_factory=list)
    background: list[Summary] = Field(default_factory=list)
    source_balance: dict = Field(default_factory=dict)  # {affiliation: count}
    languages_represented: list[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)


SYNTHESIS_SYSTEM_PROMPT = (
    "You are a knowledge synthesis engine. Given events, articles, and topic context, "
    "produce a structured analysis.\n\n"
    "For each narrative thread you identify:\n"
    "1. Group related events into coherent story arcs\n"
    "2. Identify convergence: facts confirmed by 2+ independent sources\n"
    "3. Identify divergence: where sources disagree or frame events differently\n"
    "4. Note key entities involved\n"
    "5. Rate significance (1-10)\n\n"
    "Output JSON:\n"
    "{\n"
    '  "threads": [\n'
    "    {\n"
    '      "headline": "Short thread title",\n'
    '      "event_indices": [0, 2, 5],\n'
    '      "convergence": ["Fact agreed by multiple sources"],\n'
    '      "divergence": [\n'
    '        {"claim": "What is disputed", "source_a": "outlet1", "framing_a": "How outlet1 frames it", '
    '"source_b": "outlet2", "framing_b": "How outlet2 frames it"}\n'
    "      ],\n"
    '      "key_entities": ["Entity1", "Entity2"],\n'
    '      "significance": 8\n'
    "    }\n"
    "  ]\n"
    "}"
)


async def synthesize_topic(
    llm: LLMClient,
    topic: TopicConfig,
    events: list[Event],
    articles: list[ContentItem],
    weekly_summaries: list[Summary],
    monthly_summaries: list[Summary],
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
            system_prompt=SYNTHESIS_SYSTEM_PROMPT,
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

        return TopicSynthesis(
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
        return TopicSynthesis(
            topic_name=topic.name,
            threads=threads,
            background=weekly_summaries + monthly_summaries,
            source_balance=source_balance,
            languages_represented=sorted(languages),
            metadata={"event_count": len(events), "article_count": len(articles), "fallback": True},
        )
