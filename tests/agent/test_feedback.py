"""Tests for feedback handling."""

from unittest.mock import AsyncMock

from nexus.agent.feedback import build_feedback_keyboard, handle_feedback_callback


def test_build_feedback_keyboard():
    kb = build_feedback_keyboard("2026-03-10")
    # Should return something (either InlineKeyboardMarkup or dict fallback)
    assert kb is not None


async def test_handle_feedback_up():
    store = AsyncMock()
    store.add_feedback = AsyncMock(return_value=1)

    response = await handle_feedback_callback(store, "feedback:up:2026-03-10")
    assert "Thanks" in response
    store.add_feedback.assert_called_once_with("2026-03-10", "up")


async def test_handle_feedback_down():
    store = AsyncMock()
    store.add_feedback = AsyncMock(return_value=1)

    response = await handle_feedback_callback(store, "feedback:down:2026-03-10")
    assert "improve" in response.lower() or "Thanks" in response
    store.add_feedback.assert_called_once_with("2026-03-10", "down")


async def test_handle_feedback_invalid():
    store = AsyncMock()
    response = await handle_feedback_callback(store, "invalid:data")
    assert "Invalid" in response
    store.add_feedback.assert_not_called()
