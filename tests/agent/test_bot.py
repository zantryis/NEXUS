"""Tests for the NexusBot class."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import date
from pathlib import Path

from nexus.config.models import NexusConfig, UserConfig, TelegramConfig, BriefingConfig
from nexus.agent.bot import NexusBot


@pytest.fixture
def config():
    return NexusConfig(
        user=UserConfig(name="Tristan"),
        telegram=TelegramConfig(chat_id=12345),
    )


@pytest.fixture
def bot(config, tmp_path):
    return NexusBot(
        token="fake-token",
        config=config,
        llm=AsyncMock(),
        store=AsyncMock(),
        data_dir=tmp_path,
    )


def test_is_authorized_matching(bot):
    assert bot._is_authorized(12345) is True


def test_is_authorized_wrong_id(bot):
    assert bot._is_authorized(99999) is False


def test_is_authorized_no_restriction():
    config = NexusConfig(
        user=UserConfig(name="Tristan"),
        telegram=TelegramConfig(chat_id=None),
    )
    b = NexusBot("token", config, AsyncMock(), AsyncMock())
    assert b._is_authorized(99999) is True


async def test_handle_start(bot):
    update = MagicMock()
    update.effective_chat.id = 12345
    update.message.reply_text = AsyncMock()

    await bot._handle_start(update, MagicMock())
    update.message.reply_text.assert_called_once()
    assert "Welcome" in update.message.reply_text.call_args.args[0]


async def test_handle_status(bot):
    bot._store.get_topic_stats = AsyncMock(return_value=[
        {"topic_slug": "iran-us", "event_count": 10, "thread_count": 2, "latest_date": "2026-03-10"},
    ])

    update = MagicMock()
    update.effective_chat.id = 12345
    update.message.reply_text = AsyncMock()

    await bot._handle_status(update, MagicMock())
    text = update.message.reply_text.call_args.args[0]
    assert "iran-us" in text


@patch("nexus.agent.bot.deliver_briefing", new_callable=AsyncMock)
@patch("nexus.agent.bot.build_feedback_keyboard")
async def test_handle_briefing_additional_languages(mock_keyboard, mock_deliver, tmp_path):
    """Bot delivers additional language briefings when available."""
    config = NexusConfig(
        user=UserConfig(name="Tristan"),
        telegram=TelegramConfig(chat_id=12345),
        briefing=BriefingConfig(additional_languages=["zh"]),
    )
    b = NexusBot("token", config, AsyncMock(), AsyncMock(), data_dir=tmp_path)

    today = date.today().isoformat()

    # Create English briefing
    briefing_dir = tmp_path / "artifacts" / "briefings"
    briefing_dir.mkdir(parents=True, exist_ok=True)
    (briefing_dir / f"{today}.md").write_text("English briefing")

    # Create Chinese briefing
    (briefing_dir / f"{today}-zh.md").write_text("中文简报")

    mock_keyboard.return_value = MagicMock()

    update = MagicMock()
    update.effective_chat.id = 12345
    update.message.reply_text = AsyncMock()
    context = MagicMock()

    await b._handle_briefing(update, context)

    # Should have been called twice: once for English, once for Chinese
    assert mock_deliver.call_count == 2
    assert mock_deliver.call_args_list[0].args[2] == "English briefing"
    assert mock_deliver.call_args_list[1].args[2] == "中文简报"
