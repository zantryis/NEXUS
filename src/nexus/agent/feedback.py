"""Feedback handling — inline keyboard for briefing ratings."""

import logging
from datetime import date

from nexus.engine.knowledge.store import KnowledgeStore

logger = logging.getLogger(__name__)


def build_feedback_keyboard(briefing_date: str):
    """Build an inline keyboard with thumbs up/down buttons.

    Returns a telegram InlineKeyboardMarkup if telegram is available,
    otherwise returns a dict representation.
    """
    try:
        from telegram import InlineKeyboardButton, InlineKeyboardMarkup
        return InlineKeyboardMarkup([
            [
                InlineKeyboardButton("👍", callback_data=f"feedback:up:{briefing_date}"),
                InlineKeyboardButton("👎", callback_data=f"feedback:down:{briefing_date}"),
            ]
        ])
    except ImportError:
        return {
            "inline_keyboard": [[
                {"text": "up", "callback_data": f"feedback:up:{briefing_date}"},
                {"text": "down", "callback_data": f"feedback:down:{briefing_date}"},
            ]]
        }


async def handle_feedback_callback(
    store: KnowledgeStore,
    callback_data: str,
) -> str:
    """Parse callback data and record feedback. Returns response text."""
    parts = callback_data.split(":")
    if len(parts) != 3 or parts[0] != "feedback":
        return "Invalid feedback data."

    rating = parts[1]  # "up" or "down"
    briefing_date = parts[2]

    if rating not in ("up", "down"):
        return "Invalid rating."

    await store.add_feedback(briefing_date, rating)
    return "Thanks for the feedback!" if rating == "up" else "Thanks — I'll try to improve!"
