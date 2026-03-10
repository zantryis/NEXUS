"""Retry just the audio pipeline + Telegram delivery using today's syntheses."""

import asyncio
import logging
import os
import sys
from datetime import date
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("retry_audio")


async def main():
    from nexus.config.loader import load_config
    from nexus.engine.audio.pipeline import run_audio_pipeline
    from nexus.engine.synthesis.knowledge import TopicSynthesis
    from nexus.llm.client import LLMClient

    data_dir = Path("data")
    config = load_config(data_dir / "config.yaml")
    api_key = os.getenv("GEMINI_API_KEY")

    llm = LLMClient(config.models, api_key=api_key)

    # Load today's syntheses
    today = date.today().isoformat()
    synth_dir = data_dir / "artifacts" / "syntheses" / today
    syntheses = []
    for path in sorted(synth_dir.glob("*.yaml")):
        raw = yaml.safe_load(path.read_text())
        syntheses.append(TopicSynthesis(**raw))
        logger.info(f"Loaded synthesis: {path.name}")

    if not syntheses:
        logger.error("No syntheses found")
        return

    # Run audio pipeline
    logger.info(f"Running audio pipeline with {len(syntheses)} topics...")
    audio_path = await run_audio_pipeline(
        llm, config, syntheses, data_dir, gemini_api_key=api_key,
    )

    if not audio_path:
        logger.error("Audio pipeline returned None")
        return

    logger.info(f"Audio saved: {audio_path} ({audio_path.stat().st_size / 1024:.0f} KB)")

    # Deliver via Telegram
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = config.telegram.chat_id
    if not token or not chat_id:
        logger.warning("No Telegram config — skipping delivery")
        return

    from telegram import Bot
    from nexus.agent.delivery import deliver_briefing

    bot = Bot(token=token)
    briefing_path = data_dir / "artifacts" / "briefings" / f"{today}.md"
    text = briefing_path.read_text()

    logger.info(f"Delivering briefing + audio to chat {chat_id}...")
    success = await deliver_briefing(bot, chat_id, text, audio_path)
    logger.info(f"Delivery {'successful' if success else 'failed'}!")


if __name__ == "__main__":
    asyncio.run(main())
