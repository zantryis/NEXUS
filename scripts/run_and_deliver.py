"""Run the full pipeline and deliver briefing + audio via Telegram."""

import asyncio
import logging
import os
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("run_and_deliver")


async def main():
    from nexus.config.loader import load_config
    from nexus.engine.pipeline import run_pipeline
    from nexus.llm.client import LLMClient

    data_dir = Path("data")
    config = load_config(data_dir / "config.yaml")

    api_key = os.getenv("GEMINI_API_KEY")
    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

    llm = LLMClient(
        config.models, api_key=api_key,
        anthropic_api_key=anthropic_api_key,
        deepseek_api_key=deepseek_api_key,
    )

    # Run pipeline
    logger.info("Starting pipeline...")
    briefing_path = await run_pipeline(
        config, llm, data_dir, gemini_api_key=api_key,
    )
    logger.info(f"Briefing saved: {briefing_path}")

    # Print usage
    usage = llm.usage.summary()
    logger.info(
        f"LLM usage: {usage['total_calls']} calls, "
        f"{usage['total_input_tokens']} in / {usage['total_output_tokens']} out, "
        f"{usage['total_elapsed_s']:.1f}s"
    )

    # Deliver via Telegram
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = config.telegram.chat_id

    if not token:
        logger.warning("No TELEGRAM_BOT_TOKEN — skipping delivery")
        return

    if not chat_id:
        logger.warning(
            "No telegram.chat_id set in config. "
            "Start the bot with `python -m nexus run` and send /start first, "
            "or set telegram.chat_id manually in data/config.yaml"
        )
        return

    from telegram import Bot
    from nexus.agent.delivery import deliver_briefing

    bot = Bot(token=token)
    today = date.today().isoformat()
    text = briefing_path.read_text()
    audio_path = data_dir / "artifacts" / "audio" / f"{today}.mp3"
    audio = audio_path if audio_path.exists() else None

    logger.info(f"Delivering to Telegram chat {chat_id}...")
    if audio:
        logger.info(f"Audio file: {audio} ({audio.stat().st_size / 1024:.0f} KB)")
    else:
        logger.info("No audio file generated")

    success = await deliver_briefing(bot, chat_id, text, audio)
    if success:
        logger.info("Delivery successful!")
    else:
        logger.error("Delivery failed")


if __name__ == "__main__":
    asyncio.run(main())
