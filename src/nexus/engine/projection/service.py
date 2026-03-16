"""Runtime services for snapshots, causal links, projections, and page rendering."""

from __future__ import annotations

import json
import logging
from datetime import date, timedelta
from pathlib import Path

from nexus.config.models import FutureProjectionConfig, NexusConfig
from nexus.engine.knowledge.pages import compute_prompt_hash
from nexus.engine.knowledge.store import KnowledgeStore
from nexus.engine.projection.engines import ProjectionEngineInput, export_engine_payload
from nexus.engine.projection.forecasting import (
    ForecastEngineInput,
    get_forecast_engine,
    projection_from_forecast_run,
)
from nexus.engine.projection.graph import build_graph_snapshot, export_graph_snapshot
from nexus.engine.projection.historical import HistoricalTopicState, is_signal_rich_events
from nexus.engine.projection.models import CausalLink, ForecastRun, ThreadSnapshot, TopicProjection
from nexus.engine.synthesis.knowledge import TopicSynthesis, synthesize_topic
from nexus.engine.synthesis.threads import create_thread_slug
from nexus.llm.client import LLMClient

logger = logging.getLogger(__name__)


def topic_slug_from_name(topic_name: str) -> str:
    return topic_name.lower().replace(" ", "-").replace("/", "-")


async def _build_forecast_input(
    store: KnowledgeStore,
    synthesis: TopicSynthesis,
    *,
    topic_slug: str,
    run_date: date,
    cross_topic_signals,
    metadata: dict,
) -> ForecastEngineInput:
    graph_snapshot = build_graph_snapshot(
        topic_slug=topic_slug,
        run_date=run_date,
        threads=synthesis.threads,
        cross_topic_signals=cross_topic_signals,
    )
    return ForecastEngineInput(
        topic_slug=topic_slug,
        topic_name=synthesis.topic_name,
        run_date=run_date,
        threads=synthesis.threads,
        recent_events=await store.get_recent_events(topic_slug, days=14, limit=40, reference_date=run_date),
        cross_topic_signals=cross_topic_signals,
        graph_snapshot=graph_snapshot,
        metadata=metadata,
    )


async def _generate_forecast_artifacts(
    store: KnowledgeStore,
    llm: LLMClient | None,
    synthesis: TopicSynthesis,
    *,
    topic_slug: str,
    run_date: date,
    cross_topic_signals,
    engine_name: str,
    critic_pass: bool,
    max_items: int,
    experiments_dir: Path,
    metadata: dict,
) -> tuple[ForecastRun, TopicProjection]:
    payload = await _build_forecast_input(
        store,
        synthesis,
        topic_slug=topic_slug,
        run_date=run_date,
        cross_topic_signals=cross_topic_signals,
        metadata=metadata,
    )
    export_engine_payload(experiments_dir, "graphiti_kuzu", ProjectionEngineInput(
        topic_slug=payload.topic_slug,
        topic_name=payload.topic_name,
        run_date=payload.run_date,
        threads=payload.threads,
        recent_events=payload.recent_events,
        cross_topic_signals=payload.cross_topic_signals,
        trajectory_threads=[thread for thread in payload.threads if thread.snapshot_count],
        metadata=payload.metadata,
    ))
    export_graph_snapshot(experiments_dir, engine_name, payload.graph_snapshot)
    engine = get_forecast_engine(engine_name)
    calibration_data = await store.get_historical_calibration()
    forecast_run = await engine.generate(
        llm if engine_name == "native" else None,
        payload,
        critic_pass=critic_pass,
        max_questions=max_items,
        calibration_data=calibration_data or None,
    )
    forecast_run.metadata.update(metadata)
    forecast_run.metadata["graph_snapshot_metrics"] = payload.graph_snapshot.metrics
    await store.save_forecast_run(forecast_run)
    projection = projection_from_forecast_run(forecast_run, cross_topic_signals)
    projection.metadata["forecast_run_id"] = forecast_run.run_id
    return forecast_run, projection


async def capture_thread_snapshots(
    store: KnowledgeStore,
    synthesis: TopicSynthesis,
    run_date: date,
) -> None:
    """Persist one snapshot per thread and attach latest metrics back to the synthesis."""
    for thread in synthesis.threads:
        if not thread.thread_id:
            continue
        stats = await store.get_thread_event_stats(thread.thread_id, until=run_date)
        snapshot = ThreadSnapshot(
            thread_id=thread.thread_id,
            snapshot_date=run_date,
            status=thread.status or "emerging",
            significance=thread.significance,
            event_count=stats["event_count"],
            latest_event_date=stats["latest_event_date"],
        )
        await store.upsert_thread_snapshot(snapshot)
        latest = await store.get_latest_thread_snapshot(thread.thread_id)
        history = await store.get_thread_snapshots(thread.thread_id)
        if latest:
            thread.velocity_7d = latest.velocity_7d
            thread.acceleration_7d = latest.acceleration_7d
            thread.significance_trend_7d = latest.significance_trend_7d
            thread.momentum_score = latest.momentum_score
            thread.trajectory_label = latest.trajectory_label
            thread.snapshot_count = len(history)


async def rebuild_thread_causal_links(store: KnowledgeStore, synthesis: TopicSynthesis) -> None:
    """Persist simple within-thread causal chains using event order and relation text."""
    for thread in synthesis.threads:
        if not thread.thread_id:
            continue
        events = [event for event in thread.events if event.event_id]
        events.sort(key=lambda event: (event.date, event.event_id or 0))
        links = []
        for previous, current in zip(events, events[1:]):
            evidence_text = current.relation_to_prior or f"{current.summary[:120]}"
            strength = 0.8 if current.relation_to_prior else 0.5
            links.append(CausalLink(
                source_event_id=previous.event_id,
                target_event_id=current.event_id,
                relation_type="follow_on",
                evidence_text=evidence_text,
                strength=strength,
            ))
        await store.replace_thread_causal_links(thread.thread_id, links)


async def projection_eligibility(
    store: KnowledgeStore,
    topic_slug: str,
    threads,
    config: FutureProjectionConfig,
    *,
    as_of: date | None = None,
) -> tuple[bool, dict]:
    """Return whether a topic has enough historical depth for projections."""
    topic_range = await store.get_topic_event_range(topic_slug, until=as_of)
    if not topic_range["first_date"] or not topic_range["last_date"]:
        return False, {"reason": "no_events"}

    history_days = (topic_range["last_date"] - topic_range["first_date"]).days + 1
    def _snapshot_count(thread) -> int:
        if isinstance(thread, dict):
            return int(thread.get("snapshot_count") or 0)
        return int(getattr(thread, "snapshot_count", 0) or 0)

    max_snapshots = max((_snapshot_count(thread) for thread in threads), default=0)
    eligible = history_days >= config.min_history_days and max_snapshots >= config.min_thread_snapshots
    return eligible, {
        "history_days": history_days,
        "max_thread_snapshots": max_snapshots,
        "min_history_days": config.min_history_days,
        "min_thread_snapshots": config.min_thread_snapshots,
    }


async def hydrate_synthesis_threads(
    store: KnowledgeStore,
    synthesis: TopicSynthesis,
    *,
    topic_slug: str,
    as_of: date | None = None,
) -> TopicSynthesis:
    """Attach persisted thread and event identifiers to a stored synthesis snapshot."""
    persisted_threads = await store.get_all_threads_as_of(topic_slug=topic_slug, cutoff=as_of)
    by_slug = {thread["slug"]: thread for thread in persisted_threads}
    by_headline = {thread["headline"]: thread for thread in persisted_threads}

    for thread in synthesis.threads:
        candidate_slug = thread.slug or create_thread_slug(thread.headline)
        persisted = by_slug.get(candidate_slug) or by_headline.get(thread.headline)
        if not persisted:
            continue
        thread.thread_id = persisted["id"]
        thread.slug = persisted["slug"]
        thread.status = persisted["status"]
        thread.snapshot_count = persisted.get("snapshot_count")
        thread.trajectory_label = persisted.get("trajectory_label")
        thread.momentum_score = persisted.get("momentum_score")
        thread.velocity_7d = persisted.get("velocity_7d")
        thread.acceleration_7d = persisted.get("acceleration_7d")
        thread.significance_trend_7d = persisted.get("significance_trend_7d")
        for event in thread.events:
            event_date = event.date.isoformat() if hasattr(event.date, "isoformat") else str(event.date)
            event.event_id = await store.find_event_id(event.summary, event_date, topic_slug)
    return synthesis


async def load_historical_topic_state(
    store: KnowledgeStore,
    *,
    topic_slug: str,
    topic_name: str,
    cutoff: date,
    config: FutureProjectionConfig,
    profile: str = "signal-rich",
    min_thread_snapshots_override: int | None = None,
) -> HistoricalTopicState | None:
    """Load cutoff-bounded topic state for leakage-safe replay."""
    if profile == "signal-rich" and topic_slug == "formula-1":
        return None

    raw = await store.get_synthesis(topic_slug, cutoff)
    if not raw:
        return None

    synthesis = await hydrate_synthesis_threads(
        store,
        TopicSynthesis(**raw),
        topic_slug=topic_slug,
        as_of=cutoff,
    )
    recent_events = await store.get_recent_events(topic_slug, days=14, limit=40, reference_date=cutoff)
    if profile == "signal-rich" and not is_signal_rich_events(recent_events):
        return None

    elig_config = config
    if min_thread_snapshots_override is not None:
        elig_config = config.model_copy(update={"min_thread_snapshots": min_thread_snapshots_override})
    eligible, meta = await projection_eligibility(
        store,
        topic_slug,
        synthesis.threads,
        elig_config,
        as_of=cutoff,
    )
    if not eligible:
        return None

    return HistoricalTopicState(
        topic_slug=topic_slug,
        topic_name=topic_name,
        cutoff=cutoff,
        synthesis=synthesis,
        recent_events=recent_events,
        cross_topic_signals=await store.get_cross_topic_signals_as_of(topic_slug, cutoff, limit=5),
        metadata=meta,
    )


def render_projection_markdown(projection: TopicProjection) -> str:
    """Render a deterministic cached markdown page for a projection artifact."""
    lines = [
        f"# Forward Look: {projection.topic_name}",
        "",
        f"Status: **{projection.status}**",
        f"Engine: `{projection.engine}`",
        f"Generated for: {projection.generated_for.isoformat()}",
    ]
    if projection.summary:
        lines.extend(["", projection.summary])

    if projection.items:
        lines.extend(["", "## Projection Items"])
        for idx, item in enumerate(projection.items, start=1):
            lines.extend([
                "",
                f"### {idx}. {item.claim}",
                f"- Confidence: **{item.confidence}**",
                f"- Horizon: **{item.horizon_days} days**",
                f"- Signpost: {item.signpost}",
            ])
            if item.signals_cited:
                lines.append(f"- Signals cited: {', '.join(item.signals_cited)}")
            if item.evidence_event_ids:
                lines.append(f"- Evidence events: {', '.join(str(event_id) for event_id in item.evidence_event_ids)}")

    if projection.cross_topic_signals:
        lines.extend(["", "## Cross-Topic Bridges"])
        for signal in projection.cross_topic_signals:
            lines.append(
                f"- {signal.shared_entity}: {signal.topic_slug} <-> {signal.related_topic_slug} "
                f"on {signal.observed_at.isoformat()} ({signal.note})"
            )

    return "\n".join(lines)


async def save_projection_page(store: KnowledgeStore, projection: TopicProjection) -> None:
    """Persist a cached projection page with a 1-day TTL."""
    slug = f"projection:{projection.topic_slug}"
    prompt_hash = compute_prompt_hash({
        "topic_slug": projection.topic_slug,
        "generated_for": projection.generated_for.isoformat(),
        "items": [item.model_dump(mode="json") for item in projection.items],
        "status": projection.status,
    })
    await store.save_page(
        slug=slug,
        title=f"Forward Look: {projection.topic_name}",
        page_type="projection",
        content_md=render_projection_markdown(projection),
        topic_slug=projection.topic_slug,
        ttl_days=1,
        prompt_hash=prompt_hash,
    )


async def backfill_thread_snapshots(store: KnowledgeStore) -> dict:
    """Backfill snapshots from existing thread events and synthesis dates."""
    threads = await store.get_all_threads()
    snapshot_count = 0
    causal_link_count = 0

    for thread in threads:
        thread_id = thread["id"]
        events = await store.get_events_for_thread(thread_id)
        if not events:
            continue

        event_dates = {event.date for event in events}
        topics = await store.get_topics_for_thread(thread_id)
        synthesis_dates: set[date] = set()
        for topic_slug in topics:
            synthesis_dates.update(date.fromisoformat(raw) for raw in await store.get_synthesis_dates(topic_slug))

        first_event_date = min(event_dates)
        candidate_dates = sorted(event_dates | {day for day in synthesis_dates if day >= first_event_date})
        for snapshot_date in candidate_dates:
            stats = await store.get_thread_event_stats(thread_id, until=snapshot_date)
            if stats["event_count"] == 0:
                continue
            snapshot = ThreadSnapshot(
                thread_id=thread_id,
                snapshot_date=snapshot_date,
                status=thread["status"],
                significance=thread["significance"],
                event_count=stats["event_count"],
                latest_event_date=stats["latest_event_date"],
            )
            await store.upsert_thread_snapshot(snapshot)
            snapshot_count += 1

        ordered = [event for event in events if event.event_id]
        ordered.sort(key=lambda event: (event.date, event.event_id or 0))
        links = []
        for previous, current in zip(ordered, ordered[1:]):
            links.append(CausalLink(
                source_event_id=previous.event_id,
                target_event_id=current.event_id,
                relation_type="follow_on",
                evidence_text=current.relation_to_prior or current.summary[:120],
                strength=0.8 if current.relation_to_prior else 0.5,
            ))
        await store.replace_thread_causal_links(thread_id, links)
        causal_link_count += len(links)

    return {"snapshots": snapshot_count, "causal_links": causal_link_count}


async def run_projection_pass(
    store: KnowledgeStore,
    llm: LLMClient | None,
    syntheses: list[TopicSynthesis],
    *,
    run_date: date,
    config: FutureProjectionConfig,
    experiments_dir: Path,
) -> list[TopicSynthesis]:
    """Capture deterministic projection substrate, then optionally generate projections."""
    for synthesis in syntheses:
        await capture_thread_snapshots(store, synthesis, run_date)
        await rebuild_thread_causal_links(store, synthesis)

    all_signals = await store.detect_and_save_cross_topic_signals(reference_date=run_date)
    by_topic: dict[str, list] = {}
    for signal in all_signals:
        by_topic.setdefault(signal.topic_slug, []).append(signal)

    for synthesis in syntheses:
        slug = topic_slug_from_name(synthesis.topic_name)
        synthesis.cross_topic_signals = by_topic.get(slug, [])[:5]
        if not config.enabled:
            continue

        eligible, eligibility_meta = await projection_eligibility(store, slug, synthesis.threads, config)
        if not eligible:
            synthesis.projection = TopicProjection(
                topic_slug=slug,
                topic_name=synthesis.topic_name,
                engine="native",
                generated_for=run_date,
                status="insufficient_history",
                summary="Not enough history yet to activate future projection.",
                items=[],
                cross_topic_signals=synthesis.cross_topic_signals,
                metadata=eligibility_meta,
            )
            await store.save_projection(synthesis.projection)
            await save_projection_page(store, synthesis.projection)
            continue

        forecast_run, projection = await _generate_forecast_artifacts(
            store,
            llm,
            synthesis,
            topic_slug=slug,
            run_date=run_date,
            cross_topic_signals=synthesis.cross_topic_signals,
            engine_name="native",
            critic_pass=config.critic_pass,
            max_items=config.max_items_per_topic,
            experiments_dir=experiments_dir,
            metadata=eligibility_meta,
        )
        projection.metadata["forecast_question_count"] = len(forecast_run.questions)
        synthesis.projection = projection
        await store.save_projection(projection)
        await save_projection_page(store, projection)

    return syntheses


async def generate_projections_from_store(
    store: KnowledgeStore,
    llm: LLMClient | None,
    config,
    *,
    target_date: date | None = None,
    min_thread_snapshots_override: int | None = None,
    engine_override: str | None = None,
    experiments_dir: Path | None = None,
) -> list[dict]:
    """Generate projections from already-stored syntheses without rerunning ingestion."""
    projection_config = config.future_projection.model_copy(deep=True)
    projection_config.enabled = True
    if min_thread_snapshots_override is not None:
        projection_config.min_thread_snapshots = min_thread_snapshots_override
    if engine_override:
        projection_config.engine = engine_override

    run_results: list[dict] = []
    engine_name = projection_config.engine
    export_dir = experiments_dir or Path("data") / "experiments"

    for topic in config.topics:
        slug = topic_slug_from_name(topic.name)
        synthesis_date = None
        synthesis_dates = [date.fromisoformat(raw) for raw in await store.get_synthesis_dates(slug)]
        if target_date is None:
            synthesis_date = synthesis_dates[0] if synthesis_dates else None
        else:
            synthesis_date = next((candidate for candidate in synthesis_dates if candidate <= target_date), None)
        if synthesis_date is None:
            run_results.append({"topic": topic.name, "status": "missing_synthesis"})
            continue

        raw = await store.get_synthesis(slug, synthesis_date)
        if not raw:
            run_results.append({"topic": topic.name, "status": "missing_synthesis"})
            continue
        synthesis = await hydrate_synthesis_threads(
            store,
            TopicSynthesis(**raw),
            topic_slug=slug,
        )

        cross_topic_signals = await store.get_cross_topic_signals(slug, limit=5)
        eligible, meta = await projection_eligibility(store, slug, synthesis.threads, projection_config)
        if not eligible:
            projection = TopicProjection(
                topic_slug=slug,
                topic_name=topic.name,
                engine=engine_name,
                generated_for=synthesis_date,
                status="insufficient_history",
                summary="Not enough history yet to activate future projection.",
                items=[],
                cross_topic_signals=cross_topic_signals,
                metadata=meta,
            )
            await store.save_projection(projection)
            await save_projection_page(store, projection)
            run_results.append({"topic": topic.name, "status": projection.status, "meta": meta})
            continue

        forecast_run, projection = await _generate_forecast_artifacts(
            store,
            llm,
            synthesis,
            topic_slug=slug,
            run_date=synthesis_date,
            cross_topic_signals=cross_topic_signals,
            engine_name=engine_name,
            critic_pass=projection_config.critic_pass,
            max_items=projection_config.max_items_per_topic,
            experiments_dir=export_dir,
            metadata=meta,
        )
        projection.metadata["forecast_question_count"] = len(forecast_run.questions)
        await store.save_projection(projection)
        await save_projection_page(store, projection)
        run_results.append({
            "topic": topic.name,
            "date": synthesis_date.isoformat(),
            "status": projection.status,
            "summary": projection.summary,
            "items": [item.model_dump(mode="json") for item in projection.items],
            "forecasts": [question.model_dump(mode="json") for question in forecast_run.questions],
            "meta": meta,
        })

    return run_results


async def generate_forecasts_from_store(
    store: KnowledgeStore,
    llm: LLMClient | None,
    config,
    *,
    target_date: date | None = None,
    min_thread_snapshots_override: int | None = None,
    engine_override: str | None = None,
    experiments_dir: Path | None = None,
) -> list[dict]:
    """Public forecast generation entrypoint; projections remain derived artifacts."""
    return await generate_projections_from_store(
        store,
        llm,
        config,
        target_date=target_date,
        min_thread_snapshots_override=min_thread_snapshots_override,
        engine_override=engine_override,
        experiments_dir=experiments_dir,
    )


async def run_kalshi_loop(
    store: KnowledgeStore,
    llm: LLMClient | None,
    syntheses: list[TopicSynthesis],
    *,
    run_date: date,
    kalshi_client,
    kalshi_config,
) -> dict:
    """Daily Kalshi loop: scan markets, match to topics, predict, track divergence.

    Called after run_projection_pass() if kalshi_config.auto_scan=True.
    Returns {matched, questions_generated, divergences}.
    """
    from nexus.engine.projection.kalshi_matcher import (
        compute_divergences,
        generate_aligned_forecasts,
        scan_kalshi_markets,
    )

    all_questions = []
    all_divergences = []
    seen_tickers: set[str] = set()

    for synthesis in syntheses:
        slug = topic_slug_from_name(synthesis.topic_name)

        # Collect entity names from threads
        entity_names: list[str] = []
        for thread in synthesis.threads:
            for entity in thread.key_entities or []:
                if entity not in entity_names:
                    entity_names.append(entity)

        if not entity_names:
            continue

        # Scan Kalshi markets for this topic's entities
        matched = await scan_kalshi_markets(
            kalshi_client,
            entity_names=entity_names,
            topic_name=synthesis.topic_name,
            max_markets=kalshi_config.max_markets_per_topic,
        )

        # Filter by minimum match score and deduplicate across topics
        matched = [
            m for m in matched
            if m["match_score"] >= kalshi_config.auto_match_min_score
            and m["ticker"] not in seen_tickers
        ]
        for m in matched:
            seen_tickers.add(m["ticker"])

        if not matched:
            continue

        # Generate our probability for each matched market
        questions = await generate_aligned_forecasts(
            llm,
            store,
            matched,
            topic_slug=slug,
            run_date=run_date,
        )

        all_questions.extend(questions)

        # Compute divergences
        divergences = compute_divergences(questions)
        all_divergences.extend(divergences)

    # Save aligned forecast questions to store
    if all_questions:
        from nexus.engine.projection.models import ForecastRun
        kalshi_run = ForecastRun(
            topic_slug="kalshi-aligned",
            topic_name="Kalshi Market Alignment",
            engine="graph",
            generated_for=run_date,
            summary=f"Kalshi-aligned forecasts: {len(all_questions)} markets matched.",
            questions=all_questions,
            metadata={"kalshi_aligned": True, "markets_matched": len(all_questions)},
        )
        await store.save_forecast_run(kalshi_run)

    return {
        "markets_matched": len(all_questions),
        "questions_generated": len(all_questions),
        "divergences": all_divergences,
        "top_divergence": all_divergences[0] if all_divergences else None,
    }


async def backfill_signal_rich_profile(
    store: KnowledgeStore,
    config,
    *,
    target_dir: Path,
    lookback_days: int = 14,
) -> dict:
    """Export replay-safe, verifiable benchmark slices from existing stored history."""
    target_dir.mkdir(parents=True, exist_ok=True)
    exported = 0
    topics_selected = 0

    for topic in config.topics:
        slug = topic_slug_from_name(topic.name)
        if slug == "formula-1":
            continue
        dates = [date.fromisoformat(raw) for raw in await store.get_synthesis_dates(slug)]
        if not dates:
            continue
        topic_selected = False
        for cutoff in dates:
            state = await load_historical_topic_state(
                store,
                topic_slug=slug,
                topic_name=topic.name,
                cutoff=cutoff,
                config=config.future_projection,
                profile="signal-rich",
            )
            if not state:
                continue
            payload = await _build_forecast_input(
                store,
                state.synthesis,
                topic_slug=slug,
                run_date=cutoff,
                cross_topic_signals=state.cross_topic_signals,
                metadata={**state.metadata, "profile": "signal-rich"},
            )
            export_path = target_dir / f"{slug}-{cutoff.isoformat()}.json"
            export_path.write_text(json.dumps({
                "topic_slug": payload.topic_slug,
                "topic_name": payload.topic_name,
                "run_date": payload.run_date.isoformat(),
                "threads": [thread.model_dump(mode="json") for thread in payload.threads],
                "recent_events": [event.model_dump(mode="json") for event in payload.recent_events],
                "cross_topic_signals": [signal.model_dump(mode="json") for signal in payload.cross_topic_signals],
                "graph_snapshot": payload.graph_snapshot.model_dump(mode="json") if payload.graph_snapshot else None,
                "metadata": payload.metadata,
            }, indent=2))
            exported += 1
            topic_selected = True
        if topic_selected:
            topics_selected += 1

    return {
        "profile": "signal-rich",
        "topics_selected": topics_selected,
        "exports": exported,
        "target_dir": str(target_dir),
    }


async def backfill_syntheses(
    store: KnowledgeStore,
    llm: LLMClient,
    config: NexusConfig,
    *,
    topic_slug: str,
    start: date | None = None,
    end: date | None = None,
) -> dict:
    """Generate LLM syntheses for historical event dates that lack them.

    Leakage guarantees:
    - Events scoped to date <= backfill_date
    - Thread snapshots captured with run_date = backfill_date
    - Cross-topic signals scoped with reference_date = backfill_date
    - Dates processed in chronological order (earlier first)
    """
    # Find the topic config
    topic_config = None
    for topic in config.topics:
        if topic_slug_from_name(topic.name) == topic_slug:
            topic_config = topic
            break
    if not topic_config:
        return {"error": f"Topic '{topic_slug}' not found in config", "dates_backfilled": 0}

    # Get all unique event dates for this topic
    all_events = await store.get_events(topic_slug)
    event_dates = sorted(set(e.date for e in all_events))

    # Filter to [start, end] range
    if start:
        event_dates = [d for d in event_dates if d >= start]
    if end:
        event_dates = [d for d in event_dates if d <= end]

    # Exclude dates that already have syntheses
    existing_dates = set(
        date.fromisoformat(d) for d in await store.get_synthesis_dates(topic_slug)
    )
    dates_to_backfill = [d for d in event_dates if d not in existing_dates]

    if not dates_to_backfill:
        return {"dates_backfilled": 0, "skipped": len(event_dates), "note": "all dates already have syntheses"}

    dates_backfilled = 0
    for backfill_date in dates_to_backfill:
        # Load events up to and including backfill_date (leakage-safe)
        scoped_events = await store.get_events(topic_slug, until=backfill_date)
        if not scoped_events:
            continue

        # Generate synthesis via LLM
        synthesis = await synthesize_topic(
            llm,
            topic_config,
            scoped_events,
            articles=[],
            weekly_summaries=[],
            monthly_summaries=[],
            store=store,
            topic_slug=topic_slug,
        )

        # Save synthesis
        await store.save_synthesis(synthesis.model_dump(mode="json"), topic_slug, backfill_date)

        # Capture thread snapshots at this date
        await capture_thread_snapshots(store, synthesis, backfill_date)

        # Rebuild causal links
        await rebuild_thread_causal_links(store, synthesis)

        dates_backfilled += 1
        logger.info(
            "Backfilled synthesis for %s on %s (%d/%d)",
            topic_slug, backfill_date, dates_backfilled, len(dates_to_backfill),
        )

    # Detect cross-topic signals for the latest backfilled date
    if dates_to_backfill:
        await store.detect_and_save_cross_topic_signals(reference_date=dates_to_backfill[-1])

    return {
        "dates_backfilled": dates_backfilled,
        "skipped": len(event_dates) - len(dates_to_backfill),
        "total_event_dates": len(event_dates),
    }
