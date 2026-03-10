"""Re-deliver today's briefing + audio with updated formatting."""

import asyncio
import logging
import os
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("redeliver")


async def main():
    from nexus.config.loader import load_config
    from nexus.agent.delivery import deliver_briefing

    data_dir = Path("data")
    config = load_config(data_dir / "config.yaml")
    today = date.today().isoformat()

    briefing_path = data_dir / "artifacts" / "briefings" / f"{today}.md"
    audio_path = data_dir / "artifacts" / "audio" / f"{today}.mp3"

    if not briefing_path.exists():
        logger.error(f"No briefing at {briefing_path}")
        return

    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = config.telegram.chat_id

    from telegram import Bot
    bot = Bot(token=token)

    text = briefing_path.read_text()
    audio = audio_path if audio_path.exists() else None

    audio_info = f"{audio.stat().st_size // 1024}KB" if audio else "none"
    logger.info(f"Delivering briefing ({len(text)} chars) + audio ({audio_info}) to {chat_id}")
    success = await deliver_briefing(bot, chat_id, text, audio)
    logger.info(f"{'Success' if success else 'Failed'}!")


if __name__ == "__main__":
    asyncio.run(main())
