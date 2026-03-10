"""Get your Telegram chat ID by sending /start to the bot."""

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from dotenv import load_dotenv
load_dotenv()

from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    print(f"\n{'='*50}")
    print(f"  Your chat ID: {chat_id}")
    print(f"{'='*50}")
    print(f"\nAdd this to data/config.yaml:")
    print(f"  telegram:")
    print(f"    enabled: true")
    print(f"    chat_id: {chat_id}")
    print(f"\nStopping bot...")
    await update.message.reply_text(f"Registered! Your chat ID: {chat_id}")
    # Stop after getting the ID
    asyncio.get_event_loop().call_soon(lambda: asyncio.ensure_future(context.application.stop()))


async def main():
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        print("Set TELEGRAM_BOT_TOKEN in .env")
        sys.exit(1)

    print("Bot starting... Send /start to your bot on Telegram.")
    print("(Press Ctrl+C to cancel)")

    app = ApplicationBuilder().token(token).build()
    app.add_handler(CommandHandler("start", start))

    await app.initialize()
    await app.start()
    await app.updater.start_polling()

    # Wait until stopped
    try:
        await asyncio.Event().wait()
    except asyncio.CancelledError:
        pass
    finally:
        await app.updater.stop()
        await app.stop()
        await app.shutdown()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nCancelled.")
