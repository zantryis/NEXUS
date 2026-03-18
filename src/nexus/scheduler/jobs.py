"""Scheduled jobs — daily pipeline and breaking news polling."""

import logging
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from nexus.config.models import NexusConfig
from nexus.engine.knowledge.store import KnowledgeStore
from nexus.llm.client import LLMClient

logger = logging.getLogger(__name__)


def _user_today(config: NexusConfig) -> date:
    """Get today's date in the user's configured timezone."""
    try:
        return datetime.now(ZoneInfo(config.user.timezone)).date()
    except Exception:
        return date.today()


async def daily_pipeline_job(
    config: NexusConfig,
    llm: LLMClient,
    data_dir: Path,
    store: KnowledgeStore,
    bot=None,
    gemini_api_key: str | None = None,
    openai_api_key: str | None = None,
    elevenlabs_api_key: str | None = None,
    max_ingest: int | None = None,
    trigger: str = "scheduled",
) -> None:
    """Run the daily pipeline + deliver briefing via Telegram."""
    from nexus.engine.pipeline import run_pipeline

    # Guard: skip if pipeline is already running (e.g., manual trigger)
    if await store.is_pipeline_running():
        logger.info("Scheduled pipeline skipped — another run is in progress")
        return

    logger.info("=== Scheduled daily pipeline starting ===")
    try:
        briefing_path = await run_pipeline(
            config, llm, data_dir,
            gemini_api_key=gemini_api_key,
            openai_api_key=openai_api_key,
            elevenlabs_api_key=elevenlabs_api_key,
            max_ingest=max_ingest,
            trigger="scheduled",
        )
        logger.info(f"Daily pipeline complete: {briefing_path}")

        # Deliver via Telegram if bot is running
        if bot and config.telegram.chat_id:
            from nexus.agent.delivery import deliver_briefing
            from nexus.agent.feedback import build_feedback_keyboard

            today = _user_today(config).isoformat()
            text = briefing_path.read_text()
            audio_path = data_dir / "artifacts" / "audio" / f"{today}.mp3"
            audio = audio_path if audio_path.exists() else None

            await deliver_briefing(
                bot._application.bot, config.telegram.chat_id, text, audio,
            )

            # Deliver additional language versions
            for lang in config.briefing.additional_languages:
                lang_briefing = data_dir / "artifacts" / "briefings" / f"{today}-{lang}.md"
                lang_audio = data_dir / "artifacts" / "audio" / f"{today}-{lang}.mp3"
                if lang_briefing.exists():
                    lang_text = lang_briefing.read_text()
                    lang_aud = lang_audio if lang_audio.exists() else None
                    await deliver_briefing(
                        bot._application.bot, config.telegram.chat_id,
                        lang_text, lang_aud,
                    )

            # Send feedback keyboard
            keyboard = build_feedback_keyboard(today)
            await bot._application.bot.send_message(
                chat_id=config.telegram.chat_id,
                text="How was today's briefing?",
                reply_markup=keyboard,
            )

    except Exception as e:
        logger.error(f"Daily pipeline failed: {e}", exc_info=True)


async def daily_prediction_job(
    config: NexusConfig,
    llm: LLMClient,
    store: KnowledgeStore,
    data_dir: Path,
) -> None:
    """Run daily predictions: Kalshi-aligned + KG-native + resolution checks."""
    from nexus.engine.projection.service import (
        generate_kg_native_predictions,
    )

    today = _user_today(config)
    proj_config = config.future_projection

    logger.info("=== Daily prediction job starting ===")
    results: dict = {"date": today.isoformat()}

    # 1. Kalshi-aligned predictions (if enabled)
    if proj_config.kalshi.enabled:
        try:
            from nexus.engine.projection.kalshi import build_kalshi_client
            from nexus.engine.projection.service import run_kalshi_loop
            from nexus.engine.synthesis.knowledge import TopicSynthesis

            kalshi_client = build_kalshi_client(proj_config.kalshi)

            # Load latest syntheses
            syntheses = []
            for topic in config.topics:
                slug = topic.name.lower().replace(" ", "-").replace("/", "-")
                raw = await store.get_synthesis(slug, today)
                if raw:
                    syntheses.append(TopicSynthesis(**raw))

            if syntheses:
                kalshi_result = await run_kalshi_loop(
                    store, llm, syntheses,
                    run_date=today,
                    kalshi_client=kalshi_client,
                    kalshi_config=proj_config.kalshi,
                    engine=proj_config.daily_engine,
                    topic_configs=config.topics,
                )
                results["kalshi"] = kalshi_result
                logger.info("Kalshi predictions: %d markets", kalshi_result.get("markets_matched", 0))
        except Exception as exc:
            logger.error("Kalshi prediction loop failed: %s", exc, exc_info=True)
            results["kalshi_error"] = str(exc)

    # 2. KG-native predictions
    if proj_config.kg_native_enabled:
        try:
            kg_result = await generate_kg_native_predictions(
                store, llm,
                config=config,
                run_date=today,
                max_per_topic=proj_config.max_kg_questions_per_topic,
            )
            results["kg_native"] = kg_result
            logger.info("KG-native predictions: %d generated", kg_result.get("total_generated", 0))
        except Exception as exc:
            logger.error("KG-native predictions failed: %s", exc, exc_info=True)
            results["kg_native_error"] = str(exc)

    # 3. Resolve settled Kalshi markets
    if proj_config.kalshi.enabled:
        try:
            from nexus.engine.projection.kalshi import build_kalshi_client
            from nexus.engine.projection.kalshi_resolution import resolve_kalshi_forecasts

            kalshi_client = build_kalshi_client(proj_config.kalshi)
            resolution_result = await resolve_kalshi_forecasts(store, kalshi_client)
            results["resolution"] = resolution_result
            logger.info("Resolution check: %s", resolution_result)
        except Exception as exc:
            logger.error("Kalshi resolution failed: %s", exc, exc_info=True)
            results["resolution_error"] = str(exc)

    logger.info("=== Daily prediction job complete: %s ===", results)


async def source_rediscovery_job(
    config: NexusConfig,
    llm: LLMClient,
    data_dir: Path,
) -> None:
    """Re-discover sources for all topics and merge new feeds into registries."""
    from nexus.engine.sources.discovery import discover_sources

    import yaml

    for topic in config.topics:
        slug = topic.name.lower().replace(" ", "-").replace("/", "-")
        registry_path = data_dir / "sources" / slug / "registry.yaml"

        # Load existing registry
        existing_sources: list[dict] = []
        if registry_path.exists():
            raw = yaml.safe_load(registry_path.read_text())
            if raw and "sources" in raw:
                existing_sources = raw["sources"]

        existing_urls = {s["url"] for s in existing_sources}

        try:
            result = await discover_sources(
                llm, topic.name,
                subtopics=topic.subtopics,
                existing_urls=existing_urls,
                data_dir=data_dir,
            )

            # Merge: add only feeds with new URLs
            new_feeds = [f for f in result.feeds if f["url"] not in existing_urls]
            if new_feeds:
                merged = existing_sources + new_feeds
                registry_path.parent.mkdir(parents=True, exist_ok=True)
                registry_path.write_text(
                    yaml.dump({"sources": merged}, default_flow_style=False)
                )
                logger.info(
                    f"[{topic.name}] Re-discovery: added {len(new_feeds)} new feeds "
                    f"({len(merged)} total)"
                )
            else:
                logger.info(f"[{topic.name}] Re-discovery: no new feeds found")

        except Exception as e:
            logger.error(f"[{topic.name}] Source re-discovery failed: {e}", exc_info=True)


async def breaking_news_job(
    config: NexusConfig,
    llm: LLMClient,
    store: KnowledgeStore,
    bot=None,
) -> None:
    """Check for breaking news and deliver alerts."""
    from nexus.agent.breaking import check_breaking_news

    logger.info("Checking for breaking news...")
    try:
        alerts_by_topic = await check_breaking_news(llm, config, store)

        if any(alerts_by_topic.values()) and bot and config.telegram.chat_id:
            from nexus.agent.delivery import deliver_breaking_digest
            await deliver_breaking_digest(
                bot._application.bot, config.telegram.chat_id, alerts_by_topic,
            )

        total = sum(len(a) for a in alerts_by_topic.values())
        if total:
            logger.info(f"Breaking news: {total} alerts across {len(alerts_by_topic)} topics")
        else:
            logger.info("No breaking news")

    except Exception as e:
        logger.error(f"Breaking news check failed: {e}", exc_info=True)


def schedule_jobs(
    scheduler,
    config: NexusConfig,
    llm: LLMClient,
    data_dir: Path,
    store: KnowledgeStore,
    bot=None,
    gemini_api_key: str | None = None,
    openai_api_key: str | None = None,
    elevenlabs_api_key: str | None = None,
) -> None:
    """Register scheduled jobs with an APScheduler AsyncIOScheduler."""
    # Parse schedule time (HH:MM)
    hour, minute = map(int, config.briefing.schedule.split(":"))
    timezone = config.user.timezone

    # Daily pipeline
    scheduler.add_job(
        daily_pipeline_job,
        "cron",
        hour=hour,
        minute=minute,
        timezone=timezone,
        args=[config, llm, data_dir, store],
        kwargs={
            "bot": bot,
            "gemini_api_key": gemini_api_key,
            "openai_api_key": openai_api_key,
            "elevenlabs_api_key": elevenlabs_api_key,
        },
        id="daily_pipeline",
        name="Daily Intelligence Pipeline",
        replace_existing=True,
    )
    logger.info(f"Scheduled daily pipeline at {hour:02d}:{minute:02d} {timezone}")

    # Daily predictions (if future projection enabled)
    if config.future_projection.enabled:
        offset = config.future_projection.prediction_schedule_offset_minutes
        pred_hour = hour
        pred_minute = minute + offset
        if pred_minute >= 60:
            pred_hour += pred_minute // 60
            pred_minute = pred_minute % 60
        if pred_hour >= 24:
            pred_hour = pred_hour % 24
        scheduler.add_job(
            daily_prediction_job,
            "cron",
            hour=pred_hour,
            minute=pred_minute,
            timezone=timezone,
            args=[config, llm, store, data_dir],
            id="daily_predictions",
            name="Daily Predictions",
            replace_existing=True,
        )
        logger.info(
            f"Scheduled daily predictions at {pred_hour:02d}:{pred_minute:02d} "
            f"{timezone} ({offset}min after pipeline)"
        )

    # Source re-discovery (if discovery enabled)
    if config.sources.discover_new_sources and config.topics:
        interval_days = config.sources.discovery_interval_days
        scheduler.add_job(
            source_rediscovery_job,
            "interval",
            days=interval_days,
            args=[config, llm, data_dir],
            id="source_rediscovery",
            name="Source Re-Discovery",
            replace_existing=True,
        )
        logger.info(f"Scheduled source re-discovery every {interval_days} days")

    # Breaking news (if enabled)
    if config.breaking_news.enabled:
        interval = config.breaking_news.poll_interval_hours
        scheduler.add_job(
            breaking_news_job,
            "interval",
            hours=interval,
            args=[config, llm, store],
            kwargs={"bot": bot},
            id="breaking_news",
            name="Breaking News Poller",
            replace_existing=True,
        )
        logger.info(f"Scheduled breaking news every {interval}h")
