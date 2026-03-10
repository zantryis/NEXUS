"""Scheduled jobs — daily pipeline and breaking news polling."""

import logging
from datetime import date
from pathlib import Path

from nexus.config.models import NexusConfig
from nexus.engine.knowledge.store import KnowledgeStore
from nexus.llm.client import LLMClient

logger = logging.getLogger(__name__)


async def daily_pipeline_job(
    config: NexusConfig,
    llm: LLMClient,
    data_dir: Path,
    store: KnowledgeStore,
    bot=None,
    gemini_api_key: str | None = None,
) -> None:
    """Run the daily pipeline + deliver briefing via Telegram."""
    from nexus.engine.pipeline import run_pipeline

    logger.info("=== Scheduled daily pipeline starting ===")
    try:
        briefing_path = await run_pipeline(
            config, llm, data_dir, gemini_api_key=gemini_api_key,
        )
        logger.info(f"Daily pipeline complete: {briefing_path}")

        # Deliver via Telegram if bot is running
        if bot and config.telegram.chat_id:
            from nexus.agent.delivery import deliver_briefing
            from nexus.agent.feedback import build_feedback_keyboard

            today = date.today().isoformat()
            text = briefing_path.read_text()
            audio_path = data_dir / "artifacts" / "audio" / f"{today}.mp3"
            audio = audio_path if audio_path.exists() else None

            await deliver_briefing(
                bot._application.bot, config.telegram.chat_id, text, audio,
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
        alerts = await check_breaking_news(llm, config, store)

        if alerts and bot and config.telegram.chat_id:
            from nexus.agent.delivery import deliver_breaking_alert
            for alert in alerts:
                await deliver_breaking_alert(
                    bot._application.bot, config.telegram.chat_id, alert,
                )

        if alerts:
            logger.info(f"Breaking news: {len(alerts)} alerts sent")
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
        kwargs={"bot": bot, "gemini_api_key": gemini_api_key},
        id="daily_pipeline",
        name="Daily Intelligence Pipeline",
        replace_existing=True,
    )
    logger.info(f"Scheduled daily pipeline at {hour:02d}:{minute:02d} {timezone}")

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
