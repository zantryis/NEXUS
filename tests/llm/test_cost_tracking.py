"""Tests for persistent cost tracking: LLMClient → SQLite store."""

import asyncio

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from nexus.config.models import BudgetConfig, ModelsConfig
from nexus.engine.knowledge.store import KnowledgeStore
from nexus.llm.budget import BudgetGuard
from nexus.llm.client import LLMClient


@pytest.fixture
def models_config():
    return ModelsConfig()


@pytest.fixture
def client(models_config):
    return LLMClient(models_config, api_key="test-key")


@pytest.fixture
async def store(tmp_path):
    s = KnowledgeStore(tmp_path / "knowledge.db")
    await s.initialize()
    yield s
    await s.close()


@pytest.mark.asyncio
async def test_complete_persists_to_store(client, store):
    """After complete(), usage record is persisted to SQLite store."""
    await client.set_store(store)

    mock_response = MagicMock()
    mock_response.text = "Test"
    mock_response.usage_metadata = MagicMock(
        prompt_token_count=100, candidates_token_count=50,
    )

    with patch.object(
        client._gemini_client.aio.models, "generate_content", new_callable=AsyncMock
    ) as mock_gen:
        mock_gen.return_value = mock_response
        await client.complete(
            config_key="filtering",
            system_prompt="test",
            user_prompt="test",
        )

    # Let the fire-and-forget task complete
    await asyncio.sleep(0.1)

    cursor = await store.db.execute("SELECT COUNT(*) FROM usage_log")
    row = await cursor.fetchone()
    assert row[0] == 1

    cursor = await store.db.execute(
        "SELECT provider, model, config_key, input_tokens, output_tokens FROM usage_log"
    )
    row = await cursor.fetchone()
    assert row[0] == "gemini"
    assert row[1] == "gemini-3-flash-preview"
    assert row[2] == "filtering"


@pytest.mark.asyncio
async def test_complete_without_store_does_not_fail(client):
    """complete() works fine without a store attached."""
    mock_response = MagicMock()
    mock_response.text = "Test"

    with patch.object(
        client._gemini_client.aio.models, "generate_content", new_callable=AsyncMock
    ) as mock_gen:
        mock_gen.return_value = mock_response
        result = await client.complete(
            config_key="filtering",
            system_prompt="test",
            user_prompt="test",
        )
    assert result == "Test"
    assert client._store is None


@pytest.mark.asyncio
async def test_budget_guard_syncs_from_store(store):
    """BudgetGuard loads today's spend from store on set_store()."""
    from datetime import date

    # Seed store with existing spend
    await store.add_usage_record(
        date=date.today().isoformat(), provider="gemini",
        model="gemini-3-flash-preview", config_key="filtering",
        input_tokens=1000, output_tokens=500, cost_usd=0.42,
    )

    config = BudgetConfig(daily_limit_usd=1.00, warning_threshold_usd=0.50)
    client = LLMClient(ModelsConfig(), api_key="test-key", budget_config=config)

    assert client._budget_guard.today_spend == 0.0
    await client.set_store(store)
    assert client._budget_guard.today_spend == pytest.approx(0.42)


@pytest.mark.asyncio
async def test_set_store_without_budget_guard(store):
    """set_store() works even without a budget guard."""
    client = LLMClient(ModelsConfig(), api_key="test-key")
    assert client._budget_guard is None
    await client.set_store(store)
    assert client._store is store
