"""Feedback handling — inline keyboard for briefing ratings."""

import logging

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


def build_breaking_feedback_keyboard(headline_hash: str, topic_slug: str):
    """Build inline keyboard with Useful / Not Breaking buttons.

    Returns InlineKeyboardMarkup if telegram is available, else dict fallback.
    """
    useful_data = f"breaking_fb:useful:{headline_hash}:{topic_slug}"
    not_breaking_data = f"breaking_fb:not_breaking:{headline_hash}:{topic_slug}"
    try:
        from telegram import InlineKeyboardButton, InlineKeyboardMarkup
        return InlineKeyboardMarkup([
            [
                InlineKeyboardButton("Useful", callback_data=useful_data),
                InlineKeyboardButton("Not breaking", callback_data=not_breaking_data),
            ]
        ])
    except ImportError:
        return {
            "inline_keyboard": [[
                {"text": "Useful", "callback_data": useful_data},
                {"text": "Not breaking", "callback_data": not_breaking_data},
            ]]
        }


async def handle_breaking_feedback(
    store: KnowledgeStore,
    callback_data: str,
) -> str:
    """Parse breaking feedback callback and record it. Returns response text."""
    parts = callback_data.split(":")
    if len(parts) != 4 or parts[0] != "breaking_fb":
        return "Invalid feedback data."

    feedback_type = parts[1]  # "useful" or "not_breaking"
    headline_hash = parts[2]
    topic_slug = parts[3]

    if feedback_type not in ("useful", "not_breaking"):
        return "Invalid feedback type."

    await store.add_breaking_feedback(headline_hash, topic_slug, feedback_type)
    if feedback_type == "useful":
        return "Thanks — noted as useful!"
    return "Thanks — we'll recalibrate our breaking threshold."
