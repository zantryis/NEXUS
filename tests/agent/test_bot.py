"""Tests for the NexusBot class."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import date

from nexus.config.models import NexusConfig, UserConfig, TelegramConfig, BriefingConfig
from nexus.agent.bot import NexusBot, user_today


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


def test_user_today_utc():
    """user_today with UTC should return a date."""
    result = user_today("UTC")
    assert isinstance(result, date)


def test_user_today_timezone():
    """user_today respects timezone."""
    result = user_today("America/Denver")
    assert isinstance(result, date)


def test_user_today_invalid_falls_back():
    """Invalid timezone falls back to UTC date.today()."""
    result = user_today("Not/A/Timezone")
    assert isinstance(result, date)


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
    call_kwargs = update.message.reply_text.call_args
    text = call_kwargs.args[0]
    assert "iran-us" in text
    assert call_kwargs.kwargs.get("parse_mode") == "HTML"


@patch("nexus.agent.bot.build_health_snapshot", new_callable=AsyncMock)
async def test_handle_health(mock_health, bot):
    mock_health.return_value = {
        "status": "warning",
        "pipeline": {
            "last_run": {"status": "completed", "event_count": 5, "topics": ["AI"]},
            "configured_topic_count": 2,
        },
        "deliverables": {"briefing_today": True, "audio_today": False},
        "telegram": {"enabled": True, "chat_id_configured": True},
        "litellm": {"used": True, "configured": True, "token_ttl_minutes": 8},
        "issues": [{"message": "Today's briefing exists, but today's podcast audio is missing.", "severity": "warning"}],
    }
    update = MagicMock()
    update.effective_chat.id = 12345
    update.message.reply_text = AsyncMock()

    await bot._handle_health(update, MagicMock())

    sent = update.message.reply_text.call_args.args[0]
    assert "System health" in sent
    assert "podcast=no" in sent
    assert update.message.reply_text.call_args.kwargs["parse_mode"] == "HTML"


@patch("nexus.agent.bot.answer_question", new_callable=AsyncMock)
async def test_handle_message_loading_animation(mock_qa, bot):
    """Q&A shows loading message, then edits it with the answer."""
    mock_qa.return_value = "**Iran** imposed *new* sanctions."

    loading_msg = MagicMock()
    loading_msg.message_id = 42

    update = MagicMock()
    update.effective_chat.id = 12345
    update.message.text = "What happened?"
    update.message.reply_text = AsyncMock(return_value=loading_msg)
    context = MagicMock()
    context.bot.send_chat_action = AsyncMock()
    context.bot.edit_message_text = AsyncMock()

    await bot._handle_message(update, context)

    # Should send typing indicator
    context.bot.send_chat_action.assert_called_once_with(
        chat_id=12345, action="typing",
    )

    # Should send loading message first
    update.message.reply_text.assert_called_once()
    assert "Searching" in update.message.reply_text.call_args.args[0]

    # Answer should edit the loading message (not send a new one)
    context.bot.edit_message_text.assert_called()
    final_edit = context.bot.edit_message_text.call_args
    assert final_edit.kwargs["message_id"] == 42
    assert final_edit.kwargs.get("parse_mode") == "HTML"
    assert "<b>Iran</b>" in final_edit.kwargs["text"]


@patch("nexus.agent.breaking.check_breaking_news", new_callable=AsyncMock)
async def test_handle_breaking_falls_back_to_recent_alerts(mock_check_breaking, bot):
    mock_check_breaking.return_value = {}
    bot._store.get_recent_breaking_alerts = AsyncMock(return_value=[
        {
            "headline": "Major Iran escalation",
            "source_url": "https://example.com/iran",
            "significance_score": 8,
            "topic_slug": "iran-us-relations",
            "alerted_at": "2026-03-15 08:00:00",
        },
    ])

    status_msg = MagicMock()
    status_msg.message_id = 99

    update = MagicMock()
    update.effective_chat.id = 12345
    update.message.reply_text = AsyncMock(return_value=status_msg)

    context = MagicMock()
    context.bot.edit_message_text = AsyncMock()
    context.bot.send_message = AsyncMock()

    await bot._handle_breaking(update, context)

    assert context.bot.edit_message_text.called
    edited = context.bot.edit_message_text.call_args.kwargs["text"]
    assert "No new breaking headlines" in edited
    assert "Major Iran escalation" in edited


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

    today = user_today().isoformat()

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
