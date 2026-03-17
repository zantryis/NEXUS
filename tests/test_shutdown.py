"""Tests for graceful shutdown behavior in runner.py."""

import asyncio
import sys

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from nexus.config.models import NexusConfig, UserConfig, TelegramConfig
from nexus.runner import run_all

# Pre-import so patch("nexus.agent.bot.NexusBot") can resolve the module
import nexus.agent.bot  # noqa: F401


@pytest.fixture
def base_config():
    return NexusConfig(
        user=UserConfig(name="Tristan"),
        telegram=TelegramConfig(enabled=False),
    )


def _make_runner_mocks():
    """Create the standard set of mocks for run_all."""
    mock_store = AsyncMock()
    mock_llm = MagicMock()
    mock_llm.set_store = AsyncMock()
    mock_llm.flush_usage = AsyncMock()
    mock_sched = MagicMock()
    mock_server = AsyncMock()
    return mock_store, mock_llm, mock_sched, mock_server


@pytest.mark.asyncio
async def test_scheduler_shutdown_waits(base_config, tmp_path):
    """Scheduler shutdown should wait for in-flight jobs with a timeout."""
    mock_store, mock_llm, mock_sched, mock_server = _make_runner_mocks()

    with patch("nexus.runner.KnowledgeStore", return_value=mock_store), \
         patch("nexus.runner.LLMClient", return_value=mock_llm), \
         patch("nexus.runner.AsyncIOScheduler", return_value=mock_sched), \
         patch("nexus.runner.schedule_jobs"), \
         patch("nexus.runner.create_app", return_value=MagicMock()), \
         patch("nexus.runner.uvicorn") as mock_uvicorn:

        mock_uvicorn.Config.return_value = MagicMock()
        mock_uvicorn.Server.return_value = mock_server

        await run_all(base_config, tmp_path)

    mock_sched.shutdown.assert_called_once_with(wait=True)


@pytest.mark.asyncio
async def test_bot_stop_has_timeout(tmp_path):
    """Bot stop should be wrapped in a timeout so it can't hang forever."""
    config = NexusConfig(
        user=UserConfig(name="Tristan"),
        telegram=TelegramConfig(enabled=True),
    )
    mock_store, mock_llm, mock_sched, mock_server = _make_runner_mocks()

    # Simulate a bot that hangs on stop
    mock_bot = AsyncMock()
    mock_bot.stop = AsyncMock(side_effect=asyncio.TimeoutError("hung"))

    with patch("nexus.runner.KnowledgeStore", return_value=mock_store), \
         patch("nexus.runner.LLMClient", return_value=mock_llm), \
         patch("nexus.runner.AsyncIOScheduler", return_value=mock_sched), \
         patch("nexus.runner.schedule_jobs"), \
         patch("nexus.runner.create_app", return_value=MagicMock()), \
         patch("nexus.runner.uvicorn") as mock_uvicorn, \
         patch.dict("os.environ", {"TELEGRAM_BOT_TOKEN": "fake-token"}), \
         patch("nexus.agent.bot.NexusBot", return_value=mock_bot):

        mock_uvicorn.Config.return_value = MagicMock()
        mock_uvicorn.Server.return_value = mock_server

        # Should not raise even when bot.stop times out
        await run_all(config, tmp_path)

    # Store should still be closed even if bot.stop timed out
    mock_store.close.assert_called_once()


@pytest.mark.asyncio
async def test_store_close_has_timeout(base_config, tmp_path):
    """Store close should be wrapped in a timeout so it can't hang forever."""
    mock_store, mock_llm, mock_sched, mock_server = _make_runner_mocks()

    # Simulate store.close() hanging (replaced by timeout wrapper raising TimeoutError)
    original_close = mock_store.close
    mock_store.close = AsyncMock(side_effect=asyncio.TimeoutError("hung"))

    with patch("nexus.runner.KnowledgeStore", return_value=mock_store), \
         patch("nexus.runner.LLMClient", return_value=mock_llm), \
         patch("nexus.runner.AsyncIOScheduler", return_value=mock_sched), \
         patch("nexus.runner.schedule_jobs"), \
         patch("nexus.runner.create_app", return_value=MagicMock()), \
         patch("nexus.runner.uvicorn") as mock_uvicorn:

        mock_uvicorn.Config.return_value = MagicMock()
        mock_uvicorn.Server.return_value = mock_server

        # Should not raise even if store.close hangs
        await run_all(base_config, tmp_path)

    # Shutdown should complete without error
    mock_sched.shutdown.assert_called_once()
