"""Q&A agent — answer user questions using the knowledge store.

Two-stage approach:
1. Fast LLM call to extract entities + classify intent
2. Targeted store lookups, then answer with full context
"""

import json
import logging
import re

from nexus.config.models import NexusConfig
from nexus.engine.knowledge.store import KnowledgeStore
from nexus.llm.client import LLMClient
from nexus.agent.websearch import web_search, format_web_results, _is_context_thin

logger = logging.getLogger(__name__)

# Stage 1: extract entities + classify intent
ANALYSIS_SYSTEM_PROMPT = (
    "You analyze user questions to extract search terms and classify intent. "
    "Output JSON only.\n\n"
    "FIELDS:\n"
    "- entities: list of entity names / keywords to search (e.g., [\"Iran\", \"Red Bull\", \"Honda\"])\n"
    "- intent: one of \"recent\" (what happened today/this week), "
    "\"background\" (historical context, how did X start), "
    "\"reference\" (factual lookup, who/what/where)\n"
    "- language: ISO 639-1 code of the question's language (e.g., \"en\", \"zh\", \"fa\")\n\n"
    "Example: {\"entities\": [\"Iran\", \"sanctions\"], \"intent\": \"recent\", \"language\": \"en\"}\n"
    "Example: {\"entities\": [\"Red Bull\", \"engine\"], \"intent\": \"reference\", \"language\": \"en\"}\n"
    "Example: {\"entities\": [\"伊朗\", \"美国\"], \"intent\": \"background\", \"language\": \"zh\"}"
)

# Stage 2: answer with context
QA_SYSTEM_PROMPT = (
    "You are Nexus, a news intelligence assistant. You have access to a curated "
    "knowledge store of recent events, narrative threads, and background pages.\n\n"
    "FORMAT — this is a mobile chat app, keep it phone-friendly:\n"
    "- Keep answers SHORT: 2-4 short paragraphs max. No walls of text.\n"
    "- Use bullet points for lists, **bold** for key terms.\n"
    "- Use emojis to make responses scannable and engaging "
    "(e.g., \U0001f534 for alerts, \u2705 for confirmed, \u26a0\ufe0f for caution, "
    "\U0001f4ca for data, \U0001f30d for geopolitics, \U0001f3ce\ufe0f for F1, "
    "\u26a1 for energy, \U0001f916 for AI).\n"
    "- Start with a 1-sentence TL;DR, then brief details.\n\n"
    "KNOWLEDGE:\n"
    "- For SOURCED info (from context): attribute briefly (e.g., '(Reuters)').\n"
    "- For WEB SEARCH results: use them to provide up-to-date info, cite the source.\n"
    "- For GENERAL KNOWLEDGE beyond the context: freely use your training data "
    "to fill gaps — be helpful and complete. Note it naturally "
    "(e.g., 'Historically...' or 'As of the latest regulations...').\n"
    "- Only say 'I don't know' if you truly have no relevant information.\n\n"
    "Output language: {output_language}\n"
)

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "for", "from",
    "happen", "happened", "how", "i", "in", "is", "it", "latest", "me", "news",
    "of", "on", "recent", "tell", "that", "the", "this", "to", "today", "update",
    "updates", "was", "what", "when", "where", "who", "why", "with", "you",
}


def _fallback_analysis(question: str) -> dict:
    """Cheap local analysis when the LLM is unavailable."""
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9\-/+']+", question)
    entities = [
        token for token in tokens
        if len(token) > 2 and token.lower() not in _STOPWORDS
    ][:6]
    language = "en" if question.isascii() else "auto"
    return {"entities": entities, "intent": "recent", "language": language}


async def _fallback_answer(
    store: KnowledgeStore,
    config: NexusConfig,
    question: str,
    analysis: dict,
) -> str:
    """Return a deterministic answer from the knowledge store when LLM calls fail."""
    terms = {
        str(t).lower() for t in analysis.get("entities", [])
        if str(t).strip()
    }
    terms.update(
        token.lower() for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9\-/+']+", question)
        if len(token) > 2 and token.lower() not in _STOPWORDS
    )

    topic_stats = await store.get_topic_stats()
    days = 7 if analysis.get("intent") == "background" else 3
    candidates: list[tuple[int, object, str]] = []

    for ts in topic_stats:
        slug = ts["topic_slug"]
        events = await store.get_recent_events(slug, days=days, limit=8)
        for event in events:
            haystack = " ".join([
                slug.replace("-", " "),
                getattr(event, "summary", "") or "",
                " ".join(getattr(event, "entities", []) or []),
            ]).lower()
            score = sum(1 for term in terms if term in haystack)
            if not terms:
                score = 1
            if score > 0:
                candidates.append((score, event, slug))

    candidates.sort(
        key=lambda item: (item[0], getattr(item[1], "date", None)),
        reverse=True,
    )
    selected = candidates[:5]

    if not selected:
        threads = await store.get_active_threads()
        if threads:
            bullets = [
                f"• **{t['headline']}** — significance {t['significance']}"
                for t in threads[:4]
            ]
            return (
                "Here are the most active threads in the current knowledge base.\n\n"
                "**Most active threads**\n"
                + "\n".join(bullets)
            )
        return (
            "I don’t have a strong local match in the current knowledge base yet. "
            "Try /briefing for the latest full digest."
        )

    lines = [
        "Here’s the latest from the current knowledge base.",
        "",
    ]
    if analysis.get("entities"):
        lines.append(f"**Query focus:** {', '.join(analysis['entities'][:4])}")
        lines.append("")
    for _, event, slug in selected:
        outlets = ", ".join(s.get("outlet", "?") for s in getattr(event, "sources", [])[:3])
        date_str = getattr(event, "date", "")
        lines.append(f"• **{slug.replace('-', ' ').title()}** [{date_str}] {event.summary} ({outlets})")

    return "\n".join(lines)


async def _analyze_question(llm: LLMClient, question: str) -> dict:
    """Extract entities, intent, and language from a question."""
    try:
        raw = await llm.complete(
            config_key="filtering",  # fast model
            system_prompt=ANALYSIS_SYSTEM_PROMPT,
            user_prompt=question,
            json_response=True,
        )
        return json.loads(raw)
    except Exception as e:
        fallback = _fallback_analysis(question)
        logger.warning(f"Question analysis failed, using local fallback: {e}")
        return fallback


async def _gather_context(
    store: KnowledgeStore, analysis: dict,
) -> str:
    """Gather targeted context from the store based on question analysis."""
    parts: list[str] = []
    intent = analysis.get("intent", "recent")
    entity_names = analysis.get("entities", [])

    # Always include: recent events + active threads (baseline)
    topic_stats = await store.get_topic_stats()
    for ts in topic_stats:
        slug = ts["topic_slug"]
        days = 7 if intent == "background" else 3
        events = await store.get_recent_events(slug, days=days, limit=10)
        if events:
            parts.append(f"## {slug} (recent events)")
            for e in events:
                sources_str = ", ".join(s.get("outlet", "?") for s in e.sources)
                parts.append(f"- [{e.date}] {e.summary} (sources: {sources_str})")
        projection = await store.get_latest_projection(slug)
        if projection and projection.items:
            parts.append(f"## {slug} (forward look)")
            for item in projection.items[:3]:
                parts.append(
                    f"- {item.claim} [confidence={item.confidence}, horizon={item.horizon_days}d, signpost={item.signpost}]"
                )
        signals = await store.get_cross_topic_signals(slug, limit=3)
        if signals:
            parts.append(f"## {slug} (cross-topic)")
            for signal in signals:
                parts.append(
                    f"- {signal.shared_entity} linked to {signal.related_topic_slug} on {signal.observed_at}"
                )

    threads = await store.get_active_threads()
    if threads:
        parts.append("\n## Active narrative threads")
        for t in threads:
            entities_str = ", ".join(t.get("key_entities", [])[:5])
            parts.append(
                f"- {t['headline']} (significance: {t['significance']}, "
                f"entities: {entities_str})"
            )

    # Entity-targeted lookups
    matched_entity_ids: list[int] = []
    for name in entity_names:
        results = await store.search_entities(name, limit=3)
        for ent in results:
            if ent["id"] not in matched_entity_ids:
                matched_entity_ids.append(ent["id"])

    if matched_entity_ids:
        parts.append("\n## Entity-matched events")
        for eid in matched_entity_ids[:5]:  # cap to avoid context explosion
            entity_events = await store.get_events_for_entity(eid)
            if entity_events:
                # Show last 10 events for this entity
                for e in entity_events[-10:]:
                    sources_str = ", ".join(s.get("outlet", "?") for s in e.sources)
                    parts.append(f"- [{e.date}] {e.summary} (sources: {sources_str})")

        # Threads involving these entities
        for eid in matched_entity_ids[:5]:
            entity_threads = await store.get_threads_for_entity(eid)
            for t in entity_threads:
                thread_id = t["id"]
                convergence = await store.get_convergence_for_thread(thread_id)
                divergence = await store.get_divergence_for_thread(thread_id)
                if convergence or divergence:
                    parts.append(f"\n### Thread: {t['headline']}")
                    for c in convergence:
                        confirmed = ", ".join(c.get("confirmed_by", []))
                        parts.append(f"  Confirmed: {c['fact_text']} (by: {confirmed})")
                    for d in divergence:
                        parts.append(
                            f"  Disputed: {d['shared_event']}: "
                            f"{d['source_a']} vs {d['source_b']}"
                        )

    # Background pages (for background/reference intents)
    if intent in ("background", "reference"):
        for ts in topic_stats:
            slug = ts["topic_slug"]
            page = await store.get_page(f"backstory:{slug}")
            if page and page.get("content_md"):
                # Truncate to avoid context explosion
                content = page["content_md"][:2000]
                parts.append(f"\n## Background: {page['title']}\n{content}")

        # Weekly summaries for deeper context
        for ts in topic_stats:
            slug = ts["topic_slug"]
            summaries = await store.get_summaries(slug, "weekly")
            if summaries:
                parts.append(f"\n## Weekly summaries ({slug})")
                for s in summaries[-3:]:  # last 3 weeks
                    parts.append(f"- {s.period_start} to {s.period_end}: {s.text[:300]}")

    return "\n".join(parts) if parts else "No data available in the knowledge store."


async def answer_question(
    llm: LLMClient,
    store: KnowledgeStore,
    config: NexusConfig,
    question: str,
) -> str:
    """Answer a user question with entity-aware, intent-targeted context."""
    # Stage 1: analyze question
    analysis = await _analyze_question(llm, question)
    logger.info(f"Q&A analysis: {analysis}")

    # Detect response language: match user's input language
    response_lang = analysis.get("language", config.user.output_language)
    if response_lang == "auto":
        response_lang = config.user.output_language

    # Stage 2: gather targeted context from knowledge store
    context = await _gather_context(store, analysis)

    # Stage 2.5: supplement with web search if store context is thin
    if _is_context_thin(context):
        entities = analysis.get("entities", [])
        search_query = question if not entities else f"{question} {' '.join(entities)}"
        logger.info(f"Store context thin, running web search: {search_query}")
        web_results = await web_search(search_query, max_results=5)
        if web_results:
            context += "\n" + format_web_results(web_results)

    # Stage 3: answer
    system_prompt = QA_SYSTEM_PROMPT.format(output_language=response_lang)
    user_prompt = f"Knowledge context:\n{context}\n\nUser question: {question}"

    try:
        return await llm.complete(
            config_key="agent",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
    except Exception as e:
        logger.warning(f"Q&A generation failed, using store fallback: {e}")
        return await _fallback_answer(store, config, question, analysis)
