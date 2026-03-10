"""Engine pipeline orchestrator — runs the full daily pipeline."""

import asyncio
import logging
import time
from datetime import date, timedelta
from pathlib import Path

import yaml

from nexus.config.models import NexusConfig, TopicConfig
from nexus.engine.filtering.filter import filter_items
from nexus.engine.ingestion.dedup import dedup_items
from nexus.engine.ingestion.ingest import async_ingest_items
from nexus.engine.knowledge.compression import (
    compress_to_weekly, load_summaries, save_summaries,
)
from nexus.engine.knowledge.events import (
    Event, append_events, extract_event, load_events,
    is_duplicate_event, merge_events,
)
from nexus.engine.sources.polling import ContentItem, poll_all_feeds
from nexus.engine.synthesis.knowledge import TopicSynthesis, synthesize_topic
from nexus.engine.synthesis.renderers import render_text_briefing
from nexus.llm.client import LLMClient

logger = logging.getLogger(__name__)


def load_source_registry(data_dir: Path, topic: TopicConfig) -> list[dict]:
    """Load source registry for a topic. Returns full source metadata dicts."""
    slug = topic.name.lower().replace(" ", "-").replace("/", "-")
    registry_path = data_dir / "sources" / slug / "registry.yaml"
    if not registry_path.exists():
        return []
    raw = yaml.safe_load(registry_path.read_text())
    if not raw or "sources" not in raw:
        return []
    return raw["sources"]


async def maybe_compress(
    llm: LLMClient, knowledge_dir: Path, topic_name: str, events: list[Event]
) -> None:
    """Compress old weeks into weekly summaries if not already done."""
    if not events:
        return

    weekly_path = knowledge_dir / "weekly_summaries.yaml"
    existing_summaries = load_summaries(weekly_path)
    summarized_dates = set()
    for s in existing_summaries:
        iso = s.period_start.isocalendar()
        summarized_dates.add((iso.year, iso.week))

    # Find weeks >7 days old that aren't summarized
    cutoff = date.today() - timedelta(days=7)
    old_events = [e for e in events if e.date < cutoff]
    if not old_events:
        return

    from nexus.engine.knowledge.compression import group_events_by_week
    weeks = group_events_by_week(old_events)
    unsummarized = {k: v for k, v in weeks.items() if k not in summarized_dates}

    if not unsummarized:
        return

    logger.info(f"[{topic_name}] Compressing {len(unsummarized)} old weeks")
    try:
        # Flatten unsummarized events for compression
        events_to_compress = []
        for week_events in unsummarized.values():
            events_to_compress.extend(week_events)

        new_summaries = await compress_to_weekly(llm, events_to_compress, topic_name)
        all_summaries = existing_summaries + new_summaries
        save_summaries(weekly_path, all_summaries)
        logger.info(f"[{topic_name}] Saved {len(new_summaries)} new weekly summaries")
    except Exception as e:
        logger.warning(f"[{topic_name}] Compression failed (non-blocking): {e}")


async def run_topic_pipeline(
    llm: LLMClient,
    topic: TopicConfig,
    data_dir: Path,
    sources: list[dict],
) -> TopicSynthesis:
    """Run the pipeline for a single topic: poll → ingest → filter → events → synthesize."""
    slug = topic.name.lower().replace(" ", "-").replace("/", "-")
    knowledge_dir = data_dir / "knowledge" / slug
    knowledge_dir.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}

    # Poll, dedup, and ingest
    t0 = time.monotonic()
    raw_items = poll_all_feeds(sources)
    timings["poll"] = time.monotonic() - t0
    logger.info(f"[{topic.name}] Polled {len(raw_items)} items")

    t0 = time.monotonic()
    unique_items = dedup_items(raw_items)
    timings["dedup"] = time.monotonic() - t0
    logger.info(f"[{topic.name}] {len(unique_items)} unique after dedup")

    t0 = time.monotonic()
    ingested = await async_ingest_items(unique_items)
    timings["ingest"] = time.monotonic() - t0
    logger.info(f"[{topic.name}] Ingested {len(ingested)} articles")

    # Load existing events for filter pass 2 context and event extraction
    existing_events = load_events(knowledge_dir / "events.yaml")

    # Recent events = last 7 days (up to 30) for novelty assessment
    cutoff = date.today() - timedelta(days=7)
    recent_events = [e for e in existing_events if e.date >= cutoff][-30:]

    # Filter (two-pass: relevance → significance+novelty)
    t0 = time.monotonic()
    relevant = await filter_items(llm, ingested, topic, recent_events=recent_events)
    timings["filter"] = time.monotonic() - t0
    logger.info(f"[{topic.name}] {len(relevant)} passed two-pass filter")

    # Extract events — cap at top 20 by relevance to keep pipeline fast
    MAX_EVENTS_PER_TOPIC = 20
    top_relevant = sorted(
        relevant, key=lambda x: x.relevance_score or 0, reverse=True
    )[:MAX_EVENTS_PER_TOPIC]
    logger.info(f"[{topic.name}] Extracting events for top {len(top_relevant)} articles")

    t0 = time.monotonic()
    extraction_sem = asyncio.Semaphore(5)

    async def _extract(item):
        async with extraction_sem:
            return await extract_event(llm, item, topic, existing_events)

    raw_events = await asyncio.gather(*[_extract(item) for item in top_relevant])
    extracted_events = [e for e in raw_events if e is not None]

    # Structural dedup: merge events with high entity overlap on same date
    new_events: list[Event] = []
    for event in extracted_events:
        merged = False
        for existing in new_events:
            if is_duplicate_event(event, existing):
                merge_events(existing, event)
                merged = True
                break
        if not merged:
            for existing in existing_events:
                if is_duplicate_event(event, existing):
                    merge_events(existing, event)
                    merged = True
                    break
        if not merged:
            new_events.append(event)

    timings["events"] = time.monotonic() - t0

    if new_events:
        append_events(knowledge_dir / "events.yaml", new_events)
        logger.info(
            f"[{topic.name}] Logged {len(new_events)} new events "
            f"({len(extracted_events)} extracted, {len(extracted_events) - len(new_events)} merged)"
        )

    # Compression (non-blocking): compress old weeks
    all_events = existing_events + new_events
    await maybe_compress(llm, knowledge_dir, topic.name, all_events)

    # Load summaries for synthesis context
    weekly = load_summaries(knowledge_dir / "weekly_summaries.yaml")
    monthly = load_summaries(knowledge_dir / "monthly_summaries.yaml")

    # Knowledge synthesis: build TopicSynthesis (X)
    t0 = time.monotonic()
    synthesis = await synthesize_topic(
        llm, topic,
        events=new_events or all_events[-10:],
        articles=relevant,
        weekly_summaries=weekly,
        monthly_summaries=monthly,
    )
    timings["synthesis"] = time.monotonic() - t0

    # Log timing summary
    total = sum(timings.values())
    parts = " | ".join(f"{k}={v:.1f}s" for k, v in timings.items())
    logger.info(f"[{topic.name}] Pipeline timing: {parts} | total={total:.1f}s")

    return synthesis


async def run_pipeline(
    config: NexusConfig, llm: LLMClient, data_dir: Path
) -> Path:
    """Run the full daily engine pipeline. Returns path to generated briefing."""
    pipeline_start = time.monotonic()
    syntheses: list[TopicSynthesis] = []

    for topic in config.topics:
        sources = load_source_registry(data_dir, topic)
        syn = await run_topic_pipeline(llm, topic, data_dir, sources)
        syntheses.append(syn)

    # Render briefing from TopicSynthesis objects
    t0 = time.monotonic()
    briefing_text = await render_text_briefing(llm, config, syntheses)
    render_time = time.monotonic() - t0
    logger.info(f"Briefing rendered in {render_time:.1f}s")

    # Save artifact
    today = date.today().isoformat()
    briefing_dir = data_dir / "artifacts" / "briefings"
    briefing_dir.mkdir(parents=True, exist_ok=True)
    briefing_path = briefing_dir / f"{today}.md"
    briefing_path.write_text(briefing_text)

    total_time = time.monotonic() - pipeline_start
    logger.info(f"Briefing saved to {briefing_path} (total pipeline: {total_time:.1f}s)")
    return briefing_path
