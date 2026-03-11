"""Unified runner — starts dashboard + scheduler + Telegram bot."""

import asyncio
import logging
import os
import signal
from pathlib import Path

import uvicorn
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from nexus.config.models import NexusConfig
from nexus.engine.knowledge.store import KnowledgeStore
from nexus.llm.client import LLMClient
from nexus.scheduler.jobs import schedule_jobs
from nexus.web.app import create_app

logger = logging.getLogger(__name__)


async def run_all(
    config: NexusConfig,
    data_dir: Path,
    host: str = "0.0.0.0",
    port: int = 8080,
) -> None:
    """Start everything: dashboard, scheduler, Telegram bot.

    All services share one asyncio event loop.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("deepseek")

    # Initialize shared resources
    llm = LLMClient(
        config.models,
        api_key=api_key,
        anthropic_api_key=anthropic_api_key,
        deepseek_api_key=deepseek_api_key,
        budget_config=config.budget,
    )

    store = KnowledgeStore(data_dir / "knowledge.db")
    await store.initialize()

    bot = None
    scheduler = None

    try:
        # 1. Start Telegram bot (if configured)
        telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
        if config.telegram.enabled and telegram_token:
            from nexus.agent.bot import NexusBot
            bot = NexusBot(
                token=telegram_token,
                config=config,
                llm=llm,
                store=store,
                data_dir=data_dir,
            )
            await bot.start()
            logger.info("Telegram bot started")

        # 2. Start scheduler
        scheduler = AsyncIOScheduler()
        schedule_jobs(
            scheduler, config, llm, data_dir, store,
            bot=bot, gemini_api_key=api_key,
        )
        scheduler.start()
        logger.info("Scheduler started")

        # 3. Start web dashboard (non-blocking via uvicorn Server)
        app = create_app(data_dir / "knowledge.db")
        # Set audio dir for podcast RSS
        app.state.audio_dir = data_dir / "artifacts" / "audio"
        # Inject shared resources into app state
        app.state.store = store
        app.state.llm = llm
        app.state.config = config

        uv_config = uvicorn.Config(
            app, host=host, port=port,
            log_level="info",
        )
        server = uvicorn.Server(uv_config)

        logger.info(f"Dashboard at http://{host}:{port}")
        logger.info("Nexus is running. Press Ctrl+C to stop.")

        # Run uvicorn — blocks until shutdown signal
        await server.serve()

    finally:
        # Graceful shutdown
        if scheduler:
            scheduler.shutdown(wait=False)
            logger.info("Scheduler stopped")
        if bot:
            await bot.stop()
        await store.close()
        logger.info("Nexus shutdown complete")
