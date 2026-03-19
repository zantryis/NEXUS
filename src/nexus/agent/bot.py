"""Telegram bot — main bot class with command handlers."""

import asyncio
import logging
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from nexus.config.models import NexusConfig
from nexus.engine.knowledge.store import KnowledgeStore
from nexus.llm.budget import BudgetExceededError
from nexus.llm.client import LLMClient
from nexus.agent.qa import answer_question
from nexus.agent.delivery import (
    deliver_briefing, md_to_telegram_html_light, split_message,
    format_breaking_digest,
)
from nexus.agent.feedback import (
    build_feedback_keyboard, handle_feedback_callback,
    handle_breaking_feedback,
)
from nexus.utils.health import build_health_snapshot, health_summary_lines

logger = logging.getLogger(__name__)


class CooldownTracker:
    """In-memory per-chat, per-command cooldown tracker."""

    def __init__(self):
        self._timestamps: dict[int, dict[str, float]] = {}

    def check(self, chat_id: int, command: str, cooldown_secs: float) -> float | None:
        """Check if command is on cooldown. Returns None if allowed, else remaining seconds."""
        import time
        now = time.monotonic()
        chat_ts = self._timestamps.setdefault(chat_id, {})
        last = chat_ts.get(command)
        if last is not None:
            elapsed = now - last
            if elapsed < cooldown_secs:
                return cooldown_secs - elapsed
        chat_ts[command] = now
        return None


# Cooldown durations per command (seconds)
COOLDOWNS = {
    "briefing": 30,
    "breaking": 60,
    "status": 10,
}


def user_today(timezone: str = "UTC") -> date:
    """Get today's date in the user's configured timezone."""
    try:
        return datetime.now(ZoneInfo(timezone)).date()
    except Exception:
        return date.today()


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
        self._cooldowns = CooldownTracker()

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
        self._application.add_handler(CommandHandler("breaking", self._handle_breaking))
        self._application.add_handler(CommandHandler("status", self._handle_status))
        self._application.add_handler(CommandHandler("health", self._handle_health))
        self._application.add_handler(CallbackQueryHandler(self._handle_callback))
        self._application.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_message)
        )

        await self._application.initialize()
        try:
            from telegram import BotCommand
            await self._application.bot.set_my_commands([
                BotCommand("start", "Register this chat with Nexus"),
                BotCommand("briefing", "Get today's briefing and podcast"),
                BotCommand("breaking", "Check for breaking news now"),
                BotCommand("status", "Show Nexus system status"),
                BotCommand("health", "Show delivery and model health"),
            ])
        except Exception:
            logger.debug("Failed to register Telegram slash commands", exc_info=True)
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

    def _persist_chat_id(self, chat_id: int) -> None:
        """Save chat_id to config.yaml so it survives restarts."""
        try:
            import yaml
            config_path = self._data_dir / "config.yaml"
            if config_path.exists():
                raw = yaml.safe_load(config_path.read_text()) or {}
                raw.setdefault("telegram", {})["chat_id"] = chat_id
                config_path.write_text(yaml.dump(raw, default_flow_style=False, sort_keys=False))
                logger.info(f"Persisted chat_id {chat_id} to {config_path}")
        except Exception as e:
            logger.warning(f"Failed to persist chat_id: {e}")

    async def _handle_start(self, update, context):
        """Handle /start — register user and persist chat_id."""
        chat_id = update.effective_chat.id

        # Record chat_id if not set, and persist to config.yaml
        if self._config.telegram.chat_id is None:
            self._config.telegram.chat_id = chat_id
            logger.info(f"Registered Telegram chat_id: {chat_id}")
            self._persist_chat_id(chat_id)

        await update.message.reply_text(
            "Welcome to Nexus Intelligence Briefing!\n\n"
            "Commands:\n"
            "/briefing — Get today's briefing\n"
            "/breaking — Check for breaking news now\n"
            "/status — System status\n"
            "/health — Delivery and model health\n"
            "Or just send me a question about current events."
        )

    async def _handle_briefing(self, update, context):
        """Handle /briefing — send latest briefing."""
        chat_id = update.effective_chat.id
        if not self._is_authorized(chat_id):
            await update.message.reply_text("Unauthorized.")
            return

        remaining = self._cooldowns.check(chat_id, "briefing", COOLDOWNS["briefing"])
        if remaining is not None:
            await update.message.reply_text(f"Please wait {int(remaining)}s before requesting another briefing.")
            return

        today = user_today(self._config.user.timezone).isoformat()
        briefing_path = self._data_dir / "artifacts" / "briefings" / f"{today}.md"
        audio_path = self._data_dir / "artifacts" / "audio" / f"{today}.mp3"

        if not briefing_path.exists():
            await update.message.reply_text(
                f"No briefing available for today ({today}) yet."
            )
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

    async def _handle_breaking(self, update, context):
        """Handle /breaking — on-demand breaking news check (animated single bubble)."""
        chat_id = update.effective_chat.id
        logger.info(f"Telegram /breaking received from chat {chat_id}")
        if not self._is_authorized(chat_id):
            await update.message.reply_text("Unauthorized.")
            return

        remaining = self._cooldowns.check(chat_id, "breaking", COOLDOWNS["breaking"])
        if remaining is not None:
            await update.message.reply_text(f"Please wait {int(remaining)}s before checking breaking news again.")
            return

        # Phase 1: initial status message
        status_msg = await update.message.reply_text(
            "\U0001f50d Checking wire feeds\u2026"
        )

        # Animate while running
        done = asyncio.Event()

        async def _animate():
            phases = [
                "\U0001f4e1 Polling sources\u2026",
                "\U0001f9e0 Scoring headlines\u2026",
                "\U0001f4ca Filtering results\u2026",
            ]
            i = 0
            while not done.is_set():
                await asyncio.sleep(2.0)
                if done.is_set():
                    break
                frame = self._SPINNER[i % len(self._SPINNER)]
                phase = phases[min(i, len(phases) - 1)]
                try:
                    await context.bot.edit_message_text(
                        chat_id=chat_id,
                        message_id=status_msg.message_id,
                        text=f"{frame} {phase}",
                    )
                except Exception:
                    pass
                i += 1

        animation_task = asyncio.create_task(_animate())

        try:
            from nexus.agent.breaking import check_breaking_news

            alerts_by_topic = await check_breaking_news(
                self._llm, self._config, self._store,
            )
            done.set()
            await animation_task

            if any(alerts_by_topic.values()):
                digest_text = format_breaking_digest(alerts_by_topic)
                chunks = split_message(digest_text)

                # First chunk replaces the status message
                try:
                    await context.bot.edit_message_text(
                        chat_id=chat_id,
                        message_id=status_msg.message_id,
                        text=chunks[0],
                        parse_mode="HTML",
                        disable_web_page_preview=True,
                    )
                except Exception:
                    try:
                        await context.bot.edit_message_text(
                            chat_id=chat_id,
                            message_id=status_msg.message_id,
                            text=chunks[0],
                        )
                    except Exception:
                        pass

                # Overflow chunks as new messages
                for chunk in chunks[1:]:
                    try:
                        await context.bot.send_message(
                            chat_id=chat_id,
                            text=chunk,
                            parse_mode="HTML",
                            disable_web_page_preview=True,
                        )
                    except Exception:
                        await context.bot.send_message(
                            chat_id=chat_id, text=chunk,
                        )
            else:
                recent_alerts = await self._store.get_recent_breaking_alerts(hours=24)
                if recent_alerts:
                    recent_by_topic: dict[str, list[dict]] = {}
                    for alert in recent_alerts:
                        recent_by_topic.setdefault(alert.get("topic_slug") or "other", []).append(alert)
                    digest_text = (
                        "ℹ️ No new breaking headlines since the last alert cycle. "
                        "Here are the most recent breaking items from the last 24 hours.\n\n"
                        + format_breaking_digest(recent_by_topic)
                    )
                    chunks = split_message(digest_text)
                    try:
                        await context.bot.edit_message_text(
                            chat_id=chat_id,
                            message_id=status_msg.message_id,
                            text=chunks[0],
                            parse_mode="HTML",
                            disable_web_page_preview=True,
                        )
                    except Exception:
                        await context.bot.edit_message_text(
                            chat_id=chat_id,
                            message_id=status_msg.message_id,
                            text=chunks[0],
                        )
                    for chunk in chunks[1:]:
                        try:
                            await context.bot.send_message(
                                chat_id=chat_id,
                                text=chunk,
                                parse_mode="HTML",
                                disable_web_page_preview=True,
                            )
                        except Exception:
                            await context.bot.send_message(chat_id=chat_id, text=chunk)
                else:
                    await context.bot.edit_message_text(
                        chat_id=chat_id,
                        message_id=status_msg.message_id,
                        text="No new breaking headlines in the monitored feeds right now.",
                    )

        except Exception as e:
            done.set()
            await animation_task
            logger.error(f"Breaking news check failed: {e}", exc_info=True)
            try:
                await context.bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=status_msg.message_id,
                    text="\u26a0\ufe0f Breaking news check failed.",
                )
            except Exception:
                pass

    async def _handle_status(self, update, context):
        """Handle /status — system status."""
        chat_id = update.effective_chat.id
        if not self._is_authorized(chat_id):
            await update.message.reply_text("Unauthorized.")
            return

        remaining = self._cooldowns.check(chat_id, "status", COOLDOWNS["status"])
        if remaining is not None:
            await update.message.reply_text(f"Please wait {int(remaining)}s before checking status again.")
            return

        stats = await self._store.get_topic_stats()
        lines = ["<b>Nexus Status</b>\n"]
        for s in stats:
            lines.append(
                f"\u2022 {s['topic_slug']}: {s['event_count']} events, "
                f"{s['thread_count']} threads (latest: {s['latest_date']})"
            )

        today = user_today(self._config.user.timezone).isoformat()
        briefing_path = self._data_dir / "artifacts" / "briefings" / f"{today}.md"
        lines.append(f"\nToday's briefing ({today}): {'available' if briefing_path.exists() else 'pending'}")

        await update.message.reply_text(
            "\n".join(lines), parse_mode="HTML",
        )

    async def _handle_health(self, update, context):
        """Handle /health — quick operational health summary."""
        chat_id = update.effective_chat.id
        if not self._is_authorized(chat_id):
            await update.message.reply_text("Unauthorized.")
            return

        snapshot = await build_health_snapshot(self._config, self._data_dir, self._store)
        await update.message.reply_text(
            "\n".join(health_summary_lines(snapshot)),
            parse_mode="HTML",
            disable_web_page_preview=True,
        )

    _SPINNER = ["\u280b", "\u2819", "\u2839", "\u2838", "\u283c", "\u2834", "\u2826", "\u2827", "\u2807", "\u280f"]

    async def _handle_message(self, update, context):
        """Handle text messages — route to Q&A."""
        chat_id = update.effective_chat.id
        logger.info(f"Telegram text received from chat {chat_id}: {getattr(update.message, 'text', '')[:120]}")
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

        except BudgetExceededError:
            done.set()
            await animation_task
            logger.warning(f"Q&A budget exceeded for chat {chat_id}")
            try:
                await context.bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=loading_msg.message_id,
                    text="Daily budget reached. Try again tomorrow or increase your budget in Settings.",
                )
            except Exception:
                pass

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
        """Handle inline keyboard callbacks (briefing feedback + breaking feedback)."""
        query = update.callback_query
        await query.answer()

        data = query.data
        if data.startswith("breaking_fb:"):
            response = await handle_breaking_feedback(self._store, data)
        else:
            response = await handle_feedback_callback(self._store, data)
        await query.edit_message_text(response)
