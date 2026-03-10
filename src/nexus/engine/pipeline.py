"""Engine pipeline orchestrator — runs the full daily pipeline."""

import logging
from datetime import date
from pathlib import Path

import yaml

from nexus.config.models import NexusConfig, TopicConfig
from nexus.engine.filtering.filter import filter_items
from nexus.engine.ingestion.ingest import ingest_items
from nexus.engine.knowledge.compression import load_summaries
from nexus.engine.knowledge.events import Event, append_events, extract_event, load_events
from nexus.engine.sources.polling import ContentItem, poll_all_feeds
from nexus.engine.synthesis.briefing import TopicContext, generate_briefing
from nexus.llm.client import LLMClient

logger = logging.getLogger(__name__)


def load_source_registry(data_dir: Path, topic: TopicConfig) -> list[dict]:
    """Load source registry for a topic. Returns list of {url, id} dicts."""
    slug = topic.name.lower().replace(" ", "-").replace("/", "-")
    registry_path = data_dir / "sources" / slug / "registry.yaml"
    if not registry_path.exists():
        return []
    raw = yaml.safe_load(registry_path.read_text())
    if not raw or "sources" not in raw:
        return []
    return [{"url": s["url"], "id": s["id"]} for s in raw["sources"]]


async def run_topic_pipeline(
    llm: LLMClient,
    topic: TopicConfig,
    data_dir: Path,
    sources: list[dict],
) -> TopicContext:
    """Run the pipeline for a single topic: poll → ingest → filter → events."""
    slug = topic.name.lower().replace(" ", "-").replace("/", "-")
    knowledge_dir = data_dir / "knowledge" / slug
    knowledge_dir.mkdir(parents=True, exist_ok=True)

    # Poll and ingest
    raw_items = poll_all_feeds(sources)
    logger.info(f"[{topic.name}] Polled {len(raw_items)} items")

    ingested = ingest_items(raw_items)
    logger.info(f"[{topic.name}] Ingested {len(ingested)} articles")

    # Filter
    relevant = await filter_items(llm, ingested, topic)
    logger.info(f"[{topic.name}] {len(relevant)} passed relevance filter")

    # Extract events — cap at top 20 by relevance to keep pipeline fast
    MAX_EVENTS_PER_TOPIC = 20
    top_relevant = sorted(
        relevant, key=lambda x: x.relevance_score or 0, reverse=True
    )[:MAX_EVENTS_PER_TOPIC]
    logger.info(f"[{topic.name}] Extracting events for top {len(top_relevant)} articles")

    existing_events = load_events(knowledge_dir / "events.yaml")
    new_events = []
    for item in top_relevant:
        event = await extract_event(llm, item, topic, existing_events + new_events)
        if event:
            new_events.append(event)

    if new_events:
        append_events(knowledge_dir / "events.yaml", new_events)
        logger.info(f"[{topic.name}] Logged {len(new_events)} new events")

    # Load summaries for context
    all_events = existing_events + new_events
    weekly = load_summaries(knowledge_dir / "weekly_summaries.yaml")
    monthly = load_summaries(knowledge_dir / "monthly_summaries.yaml")

    # Top articles: highest relevance for synthesis
    top_articles = sorted(
        relevant, key=lambda x: x.relevance_score or 0, reverse=True
    )[:3]

    return TopicContext(
        topic=topic,
        monthly_summaries=monthly,
        weekly_summaries=weekly,
        recent_events=new_events or all_events[-10:],
        top_articles=top_articles,
    )


async def run_pipeline(
    config: NexusConfig, llm: LLMClient, data_dir: Path
) -> Path:
    """Run the full daily engine pipeline. Returns path to generated briefing."""
    topic_contexts = []

    for topic in config.topics:
        sources = load_source_registry(data_dir, topic)
        tc = await run_topic_pipeline(llm, topic, data_dir, sources)
        topic_contexts.append(tc)

    # Generate briefing
    briefing_text = await generate_briefing(llm, config, topic_contexts)

    # Save artifact
    today = date.today().isoformat()
    briefing_dir = data_dir / "artifacts" / "briefings"
    briefing_dir.mkdir(parents=True, exist_ok=True)
    briefing_path = briefing_dir / f"{today}.md"
    briefing_path.write_text(briefing_text)

    logger.info(f"Briefing saved to {briefing_path}")
    return briefing_path
