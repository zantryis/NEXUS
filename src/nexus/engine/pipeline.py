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
from nexus.engine.knowledge.compression import compress_to_weekly
from nexus.engine.knowledge.entities import resolve_entities
from nexus.web.thumbnails import enrich_new_entities
from nexus.engine.knowledge.events import (
    Event, extract_event, is_duplicate_event, merge_events,
)
from nexus.engine.knowledge.pages import refresh_stale_pages
from nexus.engine.projection.service import run_projection_pass
from nexus.engine.sources.polling import ContentItem, poll_all_feeds, filter_recent
from nexus.engine.synthesis.knowledge import TopicSynthesis, synthesize_topic
from nexus.engine.synthesis.renderers import render_text_briefing
from nexus.engine.audio.pipeline import run_audio_pipeline
from nexus.engine.evaluation.metrics import compute_run_metrics, save_metrics
from nexus.engine.knowledge.store import KnowledgeStore
from nexus.llm.client import LLMClient, _resolve_provider
from nexus.testing.fixtures import FixtureCapture, partition_by_date

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
    llm: LLMClient, store: KnowledgeStore, topic_slug: str,
    topic_name: str, events: list[Event],
) -> None:
    """Compress old weeks into weekly summaries if not already done."""
    if not events:
        return

    existing_summaries = await store.get_summaries(topic_slug, "weekly")
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
        events_to_compress = []
        for week_events in unsummarized.values():
            events_to_compress.extend(week_events)

        new_summaries = await compress_to_weekly(llm, events_to_compress, topic_name)
        for s in new_summaries:
            await store.add_summary(s, topic_slug, "weekly")

        logger.info(f"[{topic_name}] Saved {len(new_summaries)} new weekly summaries")
    except Exception as e:
        logger.warning(f"[{topic_name}] Compression failed (non-blocking): {e}")


def _event_cap_for_topic(topic: TopicConfig) -> int:
    """Calculate event extraction cap based on topic scope."""
    if topic.max_events is not None:
        return topic.max_events
    return {"narrow": 15, "medium": 20, "broad": 35}.get(topic.scope, 20)


async def run_topic_pipeline(
    llm: LLMClient,
    topic: TopicConfig,
    data_dir: Path,
    sources: list[dict],
    store: KnowledgeStore,
    run_date: date | None = None,
    capture: FixtureCapture | None = None,
    max_ingest: int | None = None,
    bg_tasks: list | None = None,
) -> TopicSynthesis:
    """Run the pipeline for a single topic: poll → ingest → filter → events → synthesize."""
    slug = topic.name.lower().replace(" ", "-").replace("/", "-")
    processing_date = run_date or date.today()
    timings: dict[str, float] = {}

    # Poll, dedup, and ingest
    t0 = time.monotonic()
    raw_items = poll_all_feeds(sources)
    timings["poll"] = time.monotonic() - t0
    logger.info(f"[{topic.name}] Polled {len(raw_items)} items")

    # Drop stale articles (>48h) to keep volume manageable
    recent_items = filter_recent(raw_items, max_age_hours=48)
    if len(recent_items) < len(raw_items):
        logger.info(
            f"[{topic.name}] Recency filter: {len(raw_items)} → {len(recent_items)} "
            f"({len(raw_items) - len(recent_items)} older than 48h dropped)"
        )
    if capture:
        capture.save_polled(recent_items)

    t0 = time.monotonic()
    unique_items = dedup_items(recent_items)
    timings["dedup"] = time.monotonic() - t0
    logger.info(f"[{topic.name}] {len(unique_items)} unique after dedup")

    # Cap ingestion to avoid excessive LLM filtering costs
    # Prioritize items with published dates (most recent first), then undated
    MAX_INGEST = max_ingest or 250
    if len(unique_items) > MAX_INGEST:
        dated = sorted(
            [i for i in unique_items if i.published],
            key=lambda x: x.published, reverse=True,
        )
        undated = [i for i in unique_items if not i.published]
        capped = (dated + undated)[:MAX_INGEST]
        logger.info(
            f"[{topic.name}] Capped ingestion: {len(unique_items)} → {MAX_INGEST} "
            f"(keeping {len([i for i in capped if i.published])} dated, "
            f"{len([i for i in capped if not i.published])} undated)"
        )
        unique_items = capped

    t0 = time.monotonic()
    ingested = await async_ingest_items(unique_items)
    timings["ingest"] = time.monotonic() - t0
    logger.info(f"[{topic.name}] Ingested {len(ingested)} articles")
    if capture:
        capture.save_ingested(ingested)

    # Load existing events from store (last 14 days only — older events live in summaries)
    existing_events = await store.get_events(slug, since=processing_date - timedelta(days=14))

    # Recent events = last 7 days (up to 30) for novelty assessment
    cutoff = processing_date - timedelta(days=7)
    recent_events = [e for e in existing_events if e.date >= cutoff][-30:]

    # Filter (two-pass: relevance → significance+novelty)
    t0 = time.monotonic()
    filter_result = await filter_items(llm, ingested, topic, recent_events=recent_events)
    relevant = filter_result.accepted
    timings["filter"] = time.monotonic() - t0
    logger.info(f"[{topic.name}] {len(relevant)} passed two-pass filter")

    # Persist filter decisions
    if filter_result.log_entries:
        await store.add_filter_log(filter_result.log_entries)

    if capture:
        capture.save_filtered(relevant)

    # Extract events — cap by topic scope
    event_cap = _event_cap_for_topic(topic)
    top_relevant = sorted(
        relevant, key=lambda x: x.relevance_score or 0, reverse=True
    )[:event_cap]
    logger.info(f"[{topic.name}] Extracting events for top {len(top_relevant)} articles")

    t0 = time.monotonic()
    extraction_sem = asyncio.Semaphore(5)

    async def _extract(item):
        async with extraction_sem:
            return await extract_event(llm, item, topic, existing_events, current_date=processing_date)

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

    if capture:
        capture.save_events(extracted_events)

    # Entity resolution: canonicalize entity strings into graph nodes
    resolve_map: dict[str, tuple[int, str]] = {}
    if new_events:
        t0 = time.monotonic()
        all_raw = list({e_name for event in new_events for e_name in event.entities})
        known = await store.get_all_entities(slug)
        resolutions = await resolve_entities(llm, all_raw, known)

        for r in resolutions:
            aliases = [r.raw] if r.raw != r.canonical else []
            eid = await store.upsert_entity(r.canonical, r.entity_type, aliases)
            resolve_map[r.raw] = (eid, r.canonical)

        timings["entity_resolution"] = time.monotonic() - t0

        for event in new_events:
            event.raw_entities = event.raw_entities or list(event.entities)
            canonical_entities = []
            seen_entities: set[str] = set()
            for entity_name in event.raw_entities:
                canonical_name = resolve_map.get(entity_name, (None, entity_name))[1]
                key = canonical_name.lower()
                if key in seen_entities:
                    continue
                seen_entities.add(key)
                canonical_entities.append(canonical_name)
            event.entities = canonical_entities

        new_entity_ids = [
            eid for r in resolutions if r.is_new
            for eid in [resolve_map.get(r.raw, (None,))[0]] if eid
        ]
        logger.info(
            f"[{topic.name}] Resolved {len(all_raw)} entities "
            f"({len(new_entity_ids)} new)"
        )

        # Auto-fetch thumbnails + Wikipedia URLs for new entities (background, non-blocking)
        if new_entity_ids:
            async def _enrich_bg(ids=new_entity_ids, topic_name=topic.name):
                try:
                    await enrich_new_entities(store, ids)
                    logger.info(f"[{topic_name}] Enriched {len(ids)} new entities with media")
                except Exception as e:
                    logger.warning(f"[{topic_name}] Entity enrichment failed (non-fatal): {e}")
            task = asyncio.create_task(_enrich_bg())
            if bg_tasks is not None:
                bg_tasks.append(task)

    if new_events:
        event_ids = await store.add_events(new_events, slug)
        # Link events to resolved entities
        if resolve_map:
            for event_id, event in zip(event_ids, new_events):
                event.event_id = event_id
                entity_ids = [
                    resolve_map[e_name][0]
                    for e_name in event.raw_entities or event.entities
                    if e_name in resolve_map
                ]
                if entity_ids:
                    await store.link_event_entities(event_id, entity_ids)

        # Extract entity-entity relationships from new events
        if resolve_map:
            from nexus.engine.knowledge.relationships import (
                extract_relationships_from_event,
                invalidate_contradicted_relationships,
            )
            t_rel = time.monotonic()
            rel_count = 0
            for event in new_events:
                if not event.event_id:
                    continue
                # Gather existing relationships for contradiction detection
                seen_entity_ids = set()
                existing_rels: list[dict] = []
                for e_name in event.raw_entities or event.entities:
                    if e_name in resolve_map:
                        eid = resolve_map[e_name][0]
                        if eid not in seen_entity_ids:
                            seen_entity_ids.add(eid)
                            existing_rels.extend(
                                await store.get_active_relationships_for_entity(eid)
                            )
                extracted_rels = await extract_relationships_from_event(
                    llm, event, existing_relationships=existing_rels,
                )
                if extracted_rels:
                    await invalidate_contradicted_relationships(
                        store, extracted_rels, event.date,
                    )
                    for rel in extracted_rels:
                        src_id = resolve_map.get(rel.source_entity, (None,))[0]
                        tgt_id = resolve_map.get(rel.target_entity, (None,))[0]
                        if src_id and tgt_id:
                            await store.save_entity_relationship({
                                "source_entity_id": src_id,
                                "target_entity_id": tgt_id,
                                "relation_type": rel.relation_type,
                                "evidence_text": rel.evidence_text,
                                "source_event_id": event.event_id,
                                "strength": rel.strength,
                                "valid_from": rel.valid_from.isoformat(),
                            })
                            rel_count += 1
            timings["relationship_extraction"] = time.monotonic() - t_rel
            if rel_count:
                logger.info(f"[{topic.name}] Extracted {rel_count} entity relationships")

        logger.info(
            f"[{topic.name}] Logged {len(new_events)} new events "
            f"({len(extracted_events)} extracted, {len(extracted_events) - len(new_events)} merged)"
        )

    # Compression (non-blocking): compress old weeks
    all_events = existing_events + new_events
    await maybe_compress(llm, store, slug, topic.name, all_events)

    # Load summaries for synthesis context
    weekly = await store.get_summaries(slug, "weekly")
    monthly = await store.get_summaries(slug, "monthly")

    # Knowledge synthesis: build TopicSynthesis (X)
    # Fallback to recent events only (last 3 days) — older context lives in summaries
    recent_fallback = [e for e in all_events if e.date >= processing_date - timedelta(days=3)][-10:]
    t0 = time.monotonic()
    synthesis = await synthesize_topic(
        llm, topic,
        events=new_events or recent_fallback,
        articles=relevant,
        weekly_summaries=weekly,
        monthly_summaries=monthly,
        store=store,
        topic_slug=slug,
    )
    timings["synthesis"] = time.monotonic() - t0
    if capture:
        capture.save_synthesis(synthesis)

    # Log timing summary
    total = sum(timings.values())
    parts = " | ".join(f"{k}={v:.1f}s" for k, v in timings.items())
    logger.info(f"[{topic.name}] Pipeline timing: {parts} | total={total:.1f}s")

    return synthesis


async def run_pipeline(
    config: NexusConfig, llm: LLMClient, data_dir: Path,
    capture: bool = False, fixture_dir: Path | None = None,
    gemini_api_key: str | None = None,
    openai_api_key: str | None = None,
    elevenlabs_api_key: str | None = None,
    max_ingest: int | None = None,
    trigger: str = "manual",
) -> Path:
    """Run the full daily engine pipeline. Returns path to generated briefing."""
    pipeline_start = time.monotonic()
    syntheses: list[TopicSynthesis] = []

    all_articles: list[ContentItem] = []
    all_events: list[Event] = []
    extracted_event_count = 0

    if capture and fixture_dir is None:
        fixture_dir = Path("tests/fixtures")

    # Initialize knowledge store
    store = KnowledgeStore(data_dir / "knowledge.db")
    await store.initialize()

    # Record pipeline run
    topic_slugs = [_topic_slug(t.name) for t in config.topics]
    run_id = await store.start_pipeline_run(topic_slugs, trigger=trigger)

    # Attach store for persistent cost tracking + budget sync
    await llm.set_store(store)

    bg_tasks: list[asyncio.Task] = []
    try:
        for topic in config.topics:
            sources = load_source_registry(data_dir, topic)

            # Auto-discover sources if registry is empty and discovery is enabled
            if not sources and config.sources.discover_new_sources:
                logger.info(f"[{topic.name}] No sources found — running auto-discovery")
                try:
                    from nexus.engine.sources.discovery import discover_sources
                    slug = topic.name.lower().replace(" ", "-").replace("/", "-")
                    disc = await discover_sources(
                        llm, topic.name, subtopics=topic.subtopics,
                        max_feeds=8, data_dir=data_dir,
                    )
                    if disc.feeds:
                        reg_path = data_dir / "sources" / slug / "registry.yaml"
                        reg_path.parent.mkdir(parents=True, exist_ok=True)
                        reg_path.write_text(
                            yaml.dump({"sources": disc.feeds}, default_flow_style=False)
                        )
                        sources = disc.feeds
                        logger.info(f"[{topic.name}] Discovered {len(sources)} sources")
                except Exception:
                    logger.exception(f"[{topic.name}] Auto-discovery failed")

            if not sources:
                logger.warning(f"[{topic.name}] No sources available — skipping")
                continue

            cap = None
            if capture:
                slug = topic.name.lower().replace(" ", "-").replace("/", "-")
                cap = FixtureCapture(fixture_dir, slug)
            try:
                syn = await run_topic_pipeline(llm, topic, data_dir, sources,
                                               run_date=date.today(),
                                               store=store, capture=cap,
                                               max_ingest=max_ingest,
                                               bg_tasks=bg_tasks)
                syntheses.append(syn)
            except Exception:
                logger.exception(f"[{topic.name}] Topic pipeline failed — skipping")
                continue

        today_date = date.today()
        if syntheses:
            await run_projection_pass(
                store,
                llm,
                syntheses,
                run_date=today_date,
                config=config.future_projection,
                experiments_dir=data_dir / "experiments",
            )

        # Save each TopicSynthesis to disk and store
        today = today_date.isoformat()
        synth_dir = data_dir / "artifacts" / "syntheses" / today
        synth_dir.mkdir(parents=True, exist_ok=True)
        for syn in syntheses:
            slug = syn.topic_name.lower().replace(" ", "-").replace("/", "-")
            synth_path = synth_dir / f"{slug}.yaml"
            synth_path.write_text(
                yaml.dump(syn.model_dump(), default_flow_style=False, allow_unicode=True)
            )
            await store.save_synthesis(syn.model_dump(), slug, today_date)
            logger.info(f"Saved synthesis: {synth_path}")

        # Compute and save run metrics
        for topic in config.topics:
            slug = topic.name.lower().replace(" ", "-").replace("/", "-")
            topic_events = await store.get_events(slug)
            all_events.extend(topic_events)

        metrics = compute_run_metrics(syntheses, all_articles, all_events, extracted_event_count,
                                       llm_usage=llm.usage.cost_summary())
        metrics_path = save_metrics(data_dir, metrics)
        logger.info(f"Saved metrics: {metrics_path}")

        # Refresh stale narrative pages
        topic_slugs = [_topic_slug(t.name) for t in config.topics]
        topic_names = {_topic_slug(t.name): t.name for t in config.topics}
        refreshed = await refresh_stale_pages(store, llm, topic_slugs, topic_names)
        if refreshed:
            logger.info(f"Refreshed {refreshed} stale narrative pages")

        # Render briefing from TopicSynthesis objects
        t0 = time.monotonic()
        briefing_text = await render_text_briefing(llm, config, syntheses)
        render_time = time.monotonic() - t0
        logger.info(f"Briefing rendered in {render_time:.1f}s")

        # Save artifact
        briefing_dir = data_dir / "artifacts" / "briefings"
        briefing_dir.mkdir(parents=True, exist_ok=True)
        briefing_path = briefing_dir / f"{today}.md"
        briefing_path.write_text(briefing_text)

        # Audio pipeline (if enabled)
        audio_path = None
        if config.audio.enabled:
            try:
                audio_path = await run_audio_pipeline(
                    llm, config, syntheses, data_dir,
                    gemini_api_key=gemini_api_key,
                    openai_api_key=openai_api_key,
                    elevenlabs_api_key=elevenlabs_api_key,
                )
                if audio_path:
                    logger.info(f"Audio saved to {audio_path}")
            except Exception as e:
                logger.warning(f"Audio pipeline failed (non-blocking): {e}")

        # Render additional language versions (briefing + audio)
        for lang in config.briefing.additional_languages:
            try:
                lang_config = config.model_copy(deep=True)
                lang_config.user.output_language = lang

                lang_text = await render_text_briefing(llm, lang_config, syntheses)
                lang_path = briefing_dir / f"{today}-{lang}.md"
                lang_path.write_text(lang_text)
                logger.info(f"[{lang}] Briefing saved to {lang_path}")

                if config.audio.enabled:
                    try:
                        lang_audio = await run_audio_pipeline(
                            llm, lang_config, syntheses, data_dir,
                            gemini_api_key=gemini_api_key,
                            openai_api_key=openai_api_key,
                            elevenlabs_api_key=elevenlabs_api_key,
                            lang_suffix=lang,
                        )
                        if lang_audio:
                            logger.info(f"[{lang}] Audio saved to {lang_audio}")
                    except Exception as e:
                        logger.warning(f"[{lang}] Audio pipeline failed (non-blocking): {e}")
            except Exception as e:
                logger.warning(f"[{lang}] Briefing render failed (non-blocking): {e}")

        # Record successful pipeline run
        total_articles = sum(s.metadata.get("article_count", 0) for s in syntheses)
        total_events = sum(s.metadata.get("event_count", 0) for s in syntheses)
        total_cost = llm.usage.total_cost if hasattr(llm.usage, "total_cost") else 0.0
        await store.complete_pipeline_run(
            run_id, article_count=total_articles,
            event_count=total_events, cost_usd=total_cost,
        )

        total_time = time.monotonic() - pipeline_start
        logger.info(f"Briefing saved to {briefing_path} (total pipeline: {total_time:.1f}s)")
        return briefing_path
    except Exception as e:
        if run_id:
            await store.fail_pipeline_run(run_id, str(e))
        raise
    finally:
        if bg_tasks:
            _, pending = await asyncio.wait(bg_tasks, timeout=30)
            for t in pending:
                t.cancel()
        await llm.flush_usage()
        await store.close()


def _topic_slug(name: str) -> str:
    return name.lower().replace(" ", "-").replace("/", "-")


async def run_backtest(
    config: NexusConfig, llm: LLMClient, data_dir: Path,
    label: str | None = None, fixture_dir: Path | None = None,
    max_days: int | None = None,
) -> None:
    """Replay captured fixtures day-by-day with real LLM calls. Evaluates pipeline quality.

    Args:
        max_days: If set, only process the last N days of fixture data.
    """
    if fixture_dir is None:
        fixture_dir = Path("tests/fixtures")

    # Auto-detect label from synthesis model provider
    if label is None:
        label = _resolve_provider(config.models.synthesis)

    # Namespaced output dirs
    bt_data = data_dir / "backtest" / label
    bt_data.mkdir(parents=True, exist_ok=True)

    # Initialize backtest knowledge store
    bt_store = KnowledgeStore(bt_data / "knowledge.db")
    await bt_store.initialize()

    # Attach store for persistent cost tracking
    await llm.set_store(bt_store)

    backtest_start = time.monotonic()
    logger.info(f"=== Backtest started (label={label}) ===")

    for topic in config.topics:
        slug = _topic_slug(topic.name)

        # Load all captured polled items across all capture days
        all_polled: list[ContentItem] = []
        topic_fixture_dir = fixture_dir / slug
        if not topic_fixture_dir.exists():
            logger.warning(f"[{topic.name}] No fixtures found at {topic_fixture_dir}, skipping")
            continue

        for day_dir in sorted(topic_fixture_dir.iterdir()):
            if not day_dir.is_dir():
                continue
            polled_path = day_dir / "polled.json"
            if polled_path.exists():
                import json
                items = [ContentItem(**d) for d in json.loads(polled_path.read_text())]
                all_polled.extend(items)
                logger.info(f"[{topic.name}] Loaded {len(items)} polled items from {day_dir.name}")

        if not all_polled:
            logger.warning(f"[{topic.name}] No polled items found in fixtures")
            continue

        # Partition by published date (drops stale/future articles)
        days = partition_by_date(all_polled)

        # --days: restrict to last N days
        if max_days and len(days) > max_days:
            all_dates = sorted(days.keys())
            keep = all_dates[-max_days:]
            days = {d: days[d] for d in keep}

        logger.info(f"[{topic.name}] {sum(len(v) for v in days.values())} articles across {len(days)} days")

        consecutive_failures = 0

        for day_date, day_items in days.items():
            day_start = time.monotonic()
            day_label = day_date.isoformat()
            logger.info(f"[{topic.name}] === Day {day_label}: {len(day_items)} articles ===")

            try:
                # Ingest (check if ingested fixture exists, else ingest via network)
                ingested_path = topic_fixture_dir / day_label / "ingested.json"
                if ingested_path.exists():
                    import json
                    ingested = [ContentItem(**d) for d in json.loads(ingested_path.read_text())]
                    logger.info(f"[{topic.name}] Loaded {len(ingested)} ingested from fixture")
                else:
                    ingested = await async_ingest_items(day_items)
                    logger.info(f"[{topic.name}] Ingested {len(ingested)} articles via network")

                # Load accumulated events for context
                existing_events = await bt_store.get_events(slug)
                cutoff = day_date - timedelta(days=7)
                recent_events = [e for e in existing_events if e.date >= cutoff][-30:]

                # Filter with real LLM
                filter_result = await filter_items(llm, ingested, topic, recent_events=recent_events)
                relevant = filter_result.accepted
                logger.info(f"[{topic.name}] {len(relevant)} passed filter")

                # Extract events with real LLM
                extraction_sem = asyncio.Semaphore(5)

                async def _extract(item, _day=day_date):
                    async with extraction_sem:
                        return await extract_event(llm, item, topic, existing_events, current_date=_day)

                top_relevant = sorted(
                    relevant, key=lambda x: x.relevance_score or 0, reverse=True
                )[:20]

                raw_events = await asyncio.gather(*[_extract(item) for item in top_relevant])
                extracted_events = [e for e in raw_events if e is not None]

                # Dedup
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

                # Entity resolution for backtest
                resolve_map: dict[str, tuple[int, str]] = {}
                if new_events:
                    all_raw = list({e_name for ev in new_events for e_name in ev.entities})
                    known = await bt_store.get_all_entities(slug)
                    resolutions = await resolve_entities(llm, all_raw, known)
                    for r in resolutions:
                        aliases = [r.raw] if r.raw != r.canonical else []
                        eid = await bt_store.upsert_entity(r.canonical, r.entity_type, aliases)
                        resolve_map[r.raw] = (eid, r.canonical)
                    for event in new_events:
                        event.raw_entities = event.raw_entities or list(event.entities)
                        canonical_entities = []
                        seen_entities: set[str] = set()
                        for entity_name in event.raw_entities:
                            canonical_name = resolve_map.get(entity_name, (None, entity_name))[1]
                            key = canonical_name.lower()
                            if key in seen_entities:
                                continue
                            seen_entities.add(key)
                            canonical_entities.append(canonical_name)
                        event.entities = canonical_entities

                if new_events:
                    event_ids = await bt_store.add_events(new_events, slug)
                    # Link events to resolved entities
                    if resolve_map:
                        for event_id, event in zip(event_ids, new_events):
                            event.event_id = event_id
                            entity_ids = [
                                resolve_map[e_name][0]
                                for e_name in event.raw_entities or event.entities
                                if e_name in resolve_map
                            ]
                            if entity_ids:
                                await bt_store.link_event_entities(event_id, entity_ids)

                # Synthesize with real LLM
                all_events = existing_events + new_events
                weekly = await bt_store.get_summaries(slug, "weekly")
                monthly = await bt_store.get_summaries(slug, "monthly")

                synthesis = await synthesize_topic(
                    llm, topic,
                    events=new_events or all_events[-10:],
                    articles=relevant,
                    weekly_summaries=weekly,
                    monthly_summaries=monthly,
                    store=bt_store,
                    topic_slug=slug,
                )

                # Save synthesis artifact
                synth_dir = bt_data / "artifacts" / "syntheses" / day_label
                synth_dir.mkdir(parents=True, exist_ok=True)
                synth_path = synth_dir / f"{slug}.yaml"
                synth_path.write_text(
                    yaml.dump(synthesis.model_dump(), default_flow_style=False, allow_unicode=True)
                )
                await bt_store.save_synthesis(synthesis.model_dump(), slug, day_date)

                # Save metrics
                metrics = compute_run_metrics([synthesis], relevant, all_events, len(extracted_events))
                metrics["date"] = day_label
                metrics_dir = bt_data / "metrics"
                metrics_dir.mkdir(parents=True, exist_ok=True)
                metrics_path = metrics_dir / f"{day_label}.yaml"
                metrics_path.write_text(
                    yaml.dump(metrics, default_flow_style=False, allow_unicode=True)
                )

                day_time = time.monotonic() - day_start
                logger.info(
                    f"[{topic.name}] Day {day_label} complete: "
                    f"{len(new_events)} new events, {len(relevant)} articles filtered, "
                    f"{day_time:.1f}s"
                )
                consecutive_failures = 0

            except Exception as e:
                consecutive_failures += 1
                day_time = time.monotonic() - day_start
                logger.error(
                    f"[{topic.name}] Day {day_label} FAILED ({day_time:.1f}s): {e}"
                )
                if consecutive_failures >= 3:
                    logger.error(
                        f"[{topic.name}] 3 consecutive failures — stopping backtest for this topic"
                    )
                    break

            # Early stop: day took >10 min
            if time.monotonic() - day_start > 600:
                logger.error(f"[{topic.name}] Day {day_label} exceeded 10 min — stopping")
                break

    total_time = time.monotonic() - backtest_start
    usage = llm.usage.summary()
    logger.info(f"=== Backtest complete (label={label}) in {total_time:.1f}s ===")
    logger.info(
        f"LLM usage: {usage['total_calls']} calls, "
        f"{usage['total_input_tokens']} in / {usage['total_output_tokens']} out tokens, "
        f"{usage['total_elapsed_s']:.1f}s LLM time"
    )
    for provider, stats in usage.get("by_provider", {}).items():
        logger.info(
            f"  {provider}: {stats['calls']} calls, "
            f"{stats['input_tokens']} in / {stats['output_tokens']} out, "
            f"{stats['elapsed_s']:.1f}s"
        )

    # Save usage summary alongside metrics
    usage_path = bt_data / "usage_summary.yaml"
    usage_path.write_text(yaml.dump(usage, default_flow_style=False))

    await llm.flush_usage()
    await bt_store.close()
