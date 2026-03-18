"""Tests for Telegram bot rate limiting (cooldown per command)."""

import time

from nexus.agent.bot import CooldownTracker


class TestCooldownTracker:
    def test_first_call_allowed(self):
        tracker = CooldownTracker()
        assert tracker.check(chat_id=1, command="briefing", cooldown_secs=30) is None

    def test_second_call_within_cooldown_blocked(self):
        tracker = CooldownTracker()
        tracker.check(chat_id=1, command="briefing", cooldown_secs=30)
        result = tracker.check(chat_id=1, command="briefing", cooldown_secs=30)
        assert result is not None
        assert result > 0

    def test_different_commands_independent(self):
        tracker = CooldownTracker()
        tracker.check(chat_id=1, command="briefing", cooldown_secs=30)
        result = tracker.check(chat_id=1, command="status", cooldown_secs=10)
        assert result is None

    def test_different_chats_independent(self):
        tracker = CooldownTracker()
        tracker.check(chat_id=1, command="briefing", cooldown_secs=30)
        result = tracker.check(chat_id=2, command="briefing", cooldown_secs=30)
        assert result is None

    def test_call_after_cooldown_allowed(self):
        tracker = CooldownTracker()
        # Manually set a past timestamp
        tracker._timestamps[1] = {"briefing": time.monotonic() - 60}
        result = tracker.check(chat_id=1, command="briefing", cooldown_secs=30)
        assert result is None

    def test_returns_remaining_seconds(self):
        tracker = CooldownTracker()
        tracker.check(chat_id=1, command="breaking", cooldown_secs=60)
        remaining = tracker.check(chat_id=1, command="breaking", cooldown_secs=60)
        assert remaining is not None
        assert 50 < remaining <= 60
