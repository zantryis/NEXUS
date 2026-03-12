"""Tests for the unified runner."""

from unittest.mock import AsyncMock, MagicMock, patch

from nexus.config.models import NexusConfig, UserConfig, TelegramConfig
from nexus.runner import run_all


async def test_runner_initializes_store(tmp_path):
    """Runner should initialize the knowledge store."""
    config = NexusConfig(
        user=UserConfig(name="Tristan"),
        telegram=TelegramConfig(enabled=False),
    )

    with patch("nexus.runner.KnowledgeStore") as MockStore, \
         patch("nexus.runner.LLMClient") as MockLLM, \
         patch("nexus.runner.AsyncIOScheduler") as MockScheduler, \
         patch("nexus.runner.schedule_jobs"), \
         patch("nexus.runner.create_app") as mock_app, \
         patch("nexus.runner.uvicorn") as mock_uvicorn:

        mock_store = AsyncMock()
        MockStore.return_value = mock_store

        mock_llm = MagicMock()
        mock_llm.set_store = AsyncMock()
        MockLLM.return_value = mock_llm

        mock_sched = MagicMock()
        MockScheduler.return_value = mock_sched

        mock_server = AsyncMock()
        mock_uvicorn.Config.return_value = MagicMock()
        mock_uvicorn.Server.return_value = mock_server

        mock_app.return_value = MagicMock()

        await run_all(config, tmp_path)

        mock_store.initialize.assert_called_once()
        mock_sched.start.assert_called_once()
        mock_server.serve.assert_called_once()
        mock_store.close.assert_called_once()
