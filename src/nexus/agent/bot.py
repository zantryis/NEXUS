"""Telegram bot — main bot class with command handlers."""

import asyncio
import logging
from datetime import date
from pathlib import Path

from nexus.config.models import NexusConfig
from nexus.engine.knowledge.store import KnowledgeStore
from nexus.llm.client import LLMClient
from nexus.agent.qa import answer_question
from nexus.agent.delivery import deliver_briefing, md_to_telegram_html_light, split_message
from nexus.agent.feedback import build_feedback_keyboard, handle_feedback_callback

logger = logging.getLogger(__name__)


class NexusBot:
    """Telegram bot for Nexus intelligence delivery."""

    def __init__(
        self,
        token: str,
        config: NexusConfig,
        llm: LLMClient,
        store: KnowledgeStore,
        data_dir: Path = Path("data"),
    ):
        self._token = token
        self._config = config
        self._llm = llm
        self._store = store
        self._data_dir = data_dir
        self._application = None

    def _is_authorized(self, chat_id: int) -> bool:
        """Check if the chat is authorized."""
        configured = self._config.telegram.chat_id
        return configured is None or configured == chat_id

    async def start(self):
        """Initialize and start the bot with long-polling."""
        from telegram.ext import (
            ApplicationBuilder, CommandHandler,
            MessageHandler, CallbackQueryHandler, filters,
        )

        self._application = (
            ApplicationBuilder()
            .token(self._token)
            .build()
        )

        # Register handlers
        self._application.add_handler(CommandHandler("start", self._handle_start))
        self._application.add_handler(CommandHandler("briefing", self._handle_briefing))
        self._application.add_handler(CommandHandler("status", self._handle_status))
        self._application.add_handler(CallbackQueryHandler(self._handle_callback))
        self._application.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_message)
        )

        await self._application.initialize()
        await self._application.start()
        await self._application.updater.start_polling(drop_pending_updates=True)
        logger.info("Nexus Telegram bot started (long-polling)")

    async def stop(self):
        """Gracefully stop the bot."""
        if self._application:
            await self._application.updater.stop()
            await self._application.stop()
            await self._application.shutdown()
            logger.info("Nexus Telegram bot stopped")

    async def _handle_start(self, update, context):
        """Handle /start — register user."""
        chat_id = update.effective_chat.id

        # Record chat_id if not set
        if self._config.telegram.chat_id is None:
            self._config.telegram.chat_id = chat_id
            logger.info(f"Registered Telegram chat_id: {chat_id}")

        await update.message.reply_text(
            "Welcome to Nexus Intelligence Briefing!\n\n"
            "Commands:\n"
            "/briefing — Get today's briefing\n"
            "/status — System status\n"
            "Or just send me a question about current events."
        )

    async def _handle_briefing(self, update, context):
        """Handle /briefing — send latest briefing."""
        chat_id = update.effective_chat.id
        if not self._is_authorized(chat_id):
            await update.message.reply_text("Unauthorized.")
            return

        today = date.today().isoformat()
        briefing_path = self._data_dir / "artifacts" / "briefings" / f"{today}.md"
        audio_path = self._data_dir / "artifacts" / "audio" / f"{today}.mp3"

        if not briefing_path.exists():
            await update.message.reply_text("No briefing available for today yet.")
            return

        text = briefing_path.read_text()
        audio = audio_path if audio_path.exists() else None
        await deliver_briefing(context.bot, chat_id, text, audio)

        # Deliver additional language versions
        for lang in self._config.briefing.additional_languages:
            lang_briefing = self._data_dir / "artifacts" / "briefings" / f"{today}-{lang}.md"
            lang_audio = self._data_dir / "artifacts" / "audio" / f"{today}-{lang}.mp3"
            if lang_briefing.exists():
                lang_text = lang_briefing.read_text()
                lang_aud = lang_audio if lang_audio.exists() else None
                await deliver_briefing(context.bot, chat_id, lang_text, lang_aud)

        # Send feedback keyboard
        keyboard = build_feedback_keyboard(today)
        await update.message.reply_text(
            "How was today's briefing?",
            reply_markup=keyboard,
        )

    async def _handle_status(self, update, context):
        """Handle /status — system status."""
        chat_id = update.effective_chat.id
        if not self._is_authorized(chat_id):
            await update.message.reply_text("Unauthorized.")
            return

        stats = await self._store.get_topic_stats()
        lines = ["<b>Nexus Status</b>\n"]
        for s in stats:
            lines.append(
                f"\u2022 {s['topic_slug']}: {s['event_count']} events, "
                f"{s['thread_count']} threads (latest: {s['latest_date']})"
            )

        today = date.today().isoformat()
        briefing_path = self._data_dir / "artifacts" / "briefings" / f"{today}.md"
        lines.append(f"\nToday's briefing: {'available' if briefing_path.exists() else 'pending'}")

        await update.message.reply_text(
            "\n".join(lines), parse_mode="HTML",
        )

    _SPINNER = ["\u280b", "\u2819", "\u2839", "\u2838", "\u283c", "\u2834", "\u2826", "\u2827", "\u2807", "\u280f"]

    async def _handle_message(self, update, context):
        """Handle text messages — route to Q&A."""
        chat_id = update.effective_chat.id
        if not self._is_authorized(chat_id):
            return

        question = update.message.text

        # Send loading message + typing indicator
        await context.bot.send_chat_action(chat_id=chat_id, action="typing")
        loading_msg = await update.message.reply_text(
            "\U0001f50d Searching knowledge base\u2026"
        )

        # Animate spinner while Q&A runs
        done = asyncio.Event()

        async def _animate():
            i = 0
            while not done.is_set():
                await asyncio.sleep(1.2)
                if done.is_set():
                    break
                frame = self._SPINNER[i % len(self._SPINNER)]
                try:
                    await context.bot.edit_message_text(
                        chat_id=chat_id,
                        message_id=loading_msg.message_id,
                        text=f"{frame} Analyzing\u2026",
                    )
                except Exception:
                    pass
                i += 1

        animation_task = asyncio.create_task(_animate())

        try:
            answer = await answer_question(
                self._llm, self._store, self._config, question,
            )
            done.set()
            await animation_task

            html_answer = md_to_telegram_html_light(answer)
            chunks = split_message(html_answer)

            # First chunk replaces the loading message
            try:
                await context.bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=loading_msg.message_id,
                    text=chunks[0],
                    parse_mode="HTML",
                    disable_web_page_preview=True,
                )
            except Exception:
                try:
                    await context.bot.edit_message_text(
                        chat_id=chat_id,
                        message_id=loading_msg.message_id,
                        text=chunks[0],
                    )
                except Exception:
                    pass

            # Additional chunks as new messages
            for chunk in chunks[1:]:
                try:
                    await update.message.reply_text(
                        chunk, parse_mode="HTML",
                        disable_web_page_preview=True,
                    )
                except Exception:
                    await update.message.reply_text(chunk)

        except Exception as e:
            done.set()
            await animation_task
            logger.error(f"Q&A failed: {e}")
            try:
                await context.bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=loading_msg.message_id,
                    text="\u26a0\ufe0f Sorry, I couldn't process that question.",
                )
            except Exception:
                pass

    async def _handle_callback(self, update, context):
        """Handle inline keyboard callbacks (feedback)."""
        query = update.callback_query
        await query.answer()

        response = await handle_feedback_callback(
            self._store, query.data,
        )
        await query.edit_message_text(response)
