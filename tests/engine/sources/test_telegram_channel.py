"""Tests for Telegram channel source adapter."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

import httpx

from nexus.engine.sources.telegram_channel import TelegramChannelAdapter
from nexus.engine.sources.polling import ContentItem


SAMPLE_TELEGRAM_HTML = """
<html>
<body>
<div class="tgme_widget_message_wrap" data-post="testchannel/101">
  <div class="tgme_widget_message_text js-message_text" dir="auto">Breaking: Major geopolitical event unfolding in the region with significant implications for international relations.</div>
</div>
<div class="tgme_widget_message_wrap" data-post="testchannel/102">
  <div class="tgme_widget_message_text js-message_text" dir="auto">Analysis: Economic indicators show a shift in global trade patterns affecting multiple sectors and markets worldwide.</div>
</div>
<div class="tgme_widget_message_wrap" data-post="testchannel/103">
  <div class="tgme_widget_message_text js-message_text" dir="auto">Short msg</div>
</div>
</body>
</html>
"""


def test_parse_messages():
    """Parse Telegram HTML into ContentItems."""
    adapter = TelegramChannelAdapter()
    config = {"language": "en", "affiliation": "social", "country": "US", "tier": "C"}
    items = adapter._parse_messages(SAMPLE_TELEGRAM_HTML, "tg-test", config)

    # Third message is < 20 chars, should be skipped
    assert len(items) == 2
    assert items[0].source_id == "tg-test"
    assert "geopolitical" in items[0].snippet
    assert items[0].url == "https://t.me/testchannel/101"
    assert items[0].source_language == "en"
    assert items[0].source_affiliation == "social"
    assert items[0].source_tier == "C"
    assert items[1].url == "https://t.me/testchannel/102"


@patch("nexus.engine.sources.telegram_channel.httpx.AsyncClient")
async def test_poll_success(mock_client_cls):
    """Successful HTTP fetch returns parsed items."""
    mock_resp = MagicMock()
    mock_resp.text = SAMPLE_TELEGRAM_HTML
    mock_resp.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_resp)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client_cls.return_value = mock_client

    adapter = TelegramChannelAdapter()
    items = await adapter.poll({
        "channel": "@testchannel",
        "id": "tg-test",
        "language": "en",
    })
    assert len(items) == 2
    mock_client.get.assert_called_once_with("https://t.me/s/testchannel")


@patch("nexus.engine.sources.telegram_channel.httpx.AsyncClient")
async def test_poll_network_error(mock_client_cls):
    """Network errors return empty list."""
    mock_client = AsyncMock()
    mock_client.get = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client_cls.return_value = mock_client

    adapter = TelegramChannelAdapter()
    items = await adapter.poll({"channel": "@deadchannel", "id": "tg-dead"})
    assert items == []


@patch("nexus.engine.sources.telegram_channel.httpx.AsyncClient")
async def test_poll_empty_channel(mock_client_cls):
    """Valid response but no messages returns empty list."""
    mock_resp = MagicMock()
    mock_resp.text = "<html><body><div>No messages</div></body></html>"
    mock_resp.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_resp)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client_cls.return_value = mock_client

    adapter = TelegramChannelAdapter()
    items = await adapter.poll({"channel": "@emptychannel", "id": "tg-empty"})
    assert items == []


async def test_poll_missing_channel():
    """Missing channel config returns empty list without HTTP call."""
    adapter = TelegramChannelAdapter()
    items = await adapter.poll({"id": "tg-none"})
    assert items == []
