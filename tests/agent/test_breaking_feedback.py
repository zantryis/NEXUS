"""Tests for breaking news feedback keyboard and handling."""

import pytest

from nexus.engine.knowledge.store import KnowledgeStore
from nexus.agent.feedback import (
    build_breaking_feedback_keyboard,
    handle_breaking_feedback,
)


def test_build_breaking_feedback_keyboard():
    """Breaking feedback keyboard has useful/not_breaking buttons."""
    kb = build_breaking_feedback_keyboard("abc123", "iran-us")
    # Works even without telegram package installed — returns dict fallback
    if isinstance(kb, dict):
        buttons = kb["inline_keyboard"][0]
        assert len(buttons) == 2
        assert "breaking_fb:useful:" in buttons[0]["callback_data"]
        assert "breaking_fb:not_breaking:" in buttons[1]["callback_data"]
    else:
        # InlineKeyboardMarkup
        buttons = kb.inline_keyboard[0]
        assert len(buttons) == 2


@pytest.fixture
async def store(tmp_path):
    s = KnowledgeStore(tmp_path / "test.db")
    await s.initialize()
    yield s
    await s.close()


async def test_handle_breaking_feedback_useful(store):
    """Useful feedback is recorded and returns thank-you message."""
    resp = await handle_breaking_feedback(
        store, "breaking_fb:useful:abc123:iran-us"
    )
    assert "thanks" in resp.lower() or "noted" in resp.lower()


async def test_handle_breaking_feedback_not_breaking(store):
    """Not-breaking feedback is recorded."""
    resp = await handle_breaking_feedback(
        store, "breaking_fb:not_breaking:abc123:iran-us"
    )
    assert resp  # Should return acknowledgment


async def test_handle_breaking_feedback_invalid_format(store):
    """Invalid callback data returns error."""
    resp = await handle_breaking_feedback(store, "garbage")
    assert "invalid" in resp.lower()


async def test_breaking_fp_rate(store):
    """FP rate is computed from stored feedback."""
    await handle_breaking_feedback(store, "breaking_fb:useful:h1:topic-a")
    await handle_breaking_feedback(store, "breaking_fb:useful:h2:topic-a")
    await handle_breaking_feedback(store, "breaking_fb:not_breaking:h3:topic-a")

    rate = await store.get_breaking_fp_rate()
    # 1 out of 3 marked not_breaking = 33%
    assert 0.3 <= rate <= 0.34
