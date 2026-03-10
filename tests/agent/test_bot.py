"""Tests for the NexusBot class."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import date
from pathlib import Path

from nexus.config.models import NexusConfig, UserConfig, TelegramConfig
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
