"""Tests for Telegram bot API utilities."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

import httpx

from nexus.agent.telegram_utils import validate_token, poll_for_chat_id


@pytest.mark.asyncio
async def test_validate_token_success():
    """Valid token returns bot info dict."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "ok": True,
        "result": {
            "id": 123456,
            "is_bot": True,
            "first_name": "NexusBot",
            "username": "my_nexus_bot",
        },
    }

    with patch("nexus.agent.telegram_utils.httpx.AsyncClient") as MockClient:
        instance = AsyncMock()
        instance.get.return_value = mock_resp
        instance.__aenter__ = AsyncMock(return_value=instance)
        instance.__aexit__ = AsyncMock(return_value=False)
        MockClient.return_value = instance

        result = await validate_token("valid-token")

    assert result is not None
    assert result["username"] == "my_nexus_bot"
    assert result["id"] == 123456


@pytest.mark.asyncio
async def test_validate_token_invalid():
    """Invalid token returns None."""
    mock_resp = MagicMock()
    mock_resp.status_code = 401
    mock_resp.json.return_value = {"ok": False, "description": "Unauthorized"}

    with patch("nexus.agent.telegram_utils.httpx.AsyncClient") as MockClient:
        instance = AsyncMock()
        instance.get.return_value = mock_resp
        instance.__aenter__ = AsyncMock(return_value=instance)
        instance.__aexit__ = AsyncMock(return_value=False)
        MockClient.return_value = instance

        result = await validate_token("bad-token")

    assert result is None


@pytest.mark.asyncio
async def test_validate_token_network_error():
    """Network error returns None."""
    with patch("nexus.agent.telegram_utils.httpx.AsyncClient") as MockClient:
        instance = AsyncMock()
        instance.get.side_effect = httpx.ConnectError("Connection refused")
        instance.__aenter__ = AsyncMock(return_value=instance)
        instance.__aexit__ = AsyncMock(return_value=False)
        MockClient.return_value = instance

        result = await validate_token("some-token")

    assert result is None


@pytest.mark.asyncio
async def test_poll_finds_chat_id():
    """Poll finds a /start message and returns chat_id."""
    updates_resp = MagicMock()
    updates_resp.status_code = 200
    updates_resp.json.return_value = {
        "ok": True,
        "result": [
            {
                "update_id": 100,
                "message": {
                    "text": "/start",
                    "chat": {"id": 987654321, "type": "private"},
                    "from": {"id": 987654321, "first_name": "Tristan"},
                },
            }
        ],
    }

    ack_resp = MagicMock()
    ack_resp.status_code = 200
    ack_resp.json.return_value = {"ok": True, "result": []}

    with patch("nexus.agent.telegram_utils.httpx.AsyncClient") as MockClient:
        instance = AsyncMock()
        instance.get.side_effect = [updates_resp, ack_resp]
        instance.__aenter__ = AsyncMock(return_value=instance)
        instance.__aexit__ = AsyncMock(return_value=False)
        MockClient.return_value = instance

        result = await poll_for_chat_id("valid-token", timeout=1.0)

    assert result == 987654321


@pytest.mark.asyncio
async def test_poll_no_start_message():
    """Poll returns None when no /start message found."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"ok": True, "result": []}

    with patch("nexus.agent.telegram_utils.httpx.AsyncClient") as MockClient:
        instance = AsyncMock()
        instance.get.return_value = mock_resp
        instance.__aenter__ = AsyncMock(return_value=instance)
        instance.__aexit__ = AsyncMock(return_value=False)
        MockClient.return_value = instance

        result = await poll_for_chat_id("valid-token", timeout=1.0)

    assert result is None


@pytest.mark.asyncio
async def test_poll_non_start_messages_ignored():
    """Non-/start messages are ignored."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "ok": True,
        "result": [
            {
                "update_id": 50,
                "message": {
                    "text": "Hello bot!",
                    "chat": {"id": 111, "type": "private"},
                },
            },
            {
                "update_id": 51,
                "message": {
                    "text": "/help",
                    "chat": {"id": 222, "type": "private"},
                },
            },
        ],
    }

    with patch("nexus.agent.telegram_utils.httpx.AsyncClient") as MockClient:
        instance = AsyncMock()
        instance.get.return_value = mock_resp
        instance.__aenter__ = AsyncMock(return_value=instance)
        instance.__aexit__ = AsyncMock(return_value=False)
        MockClient.return_value = instance

        result = await poll_for_chat_id("valid-token", timeout=1.0)

    assert result is None
