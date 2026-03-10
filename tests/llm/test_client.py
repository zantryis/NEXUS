"""Tests for LLM abstraction layer."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from nexus.config.models import ModelsConfig
from nexus.llm.client import LLMClient


@pytest.fixture
def models_config():
    return ModelsConfig()


@pytest.fixture
def client(models_config):
    return LLMClient(models_config, api_key="test-key")


def test_resolve_model(client):
    assert client.resolve_model("filtering") == "gemini-3-flash-preview"
    assert client.resolve_model("synthesis") == "gemini-3.1-pro-preview"


def test_resolve_model_invalid_key(client):
    with pytest.raises(ValueError, match="Unknown config key"):
        client.resolve_model("nonexistent")


@pytest.mark.asyncio
async def test_complete_calls_gemini(client):
    mock_response = MagicMock()
    mock_response.text = "Test response"

    with patch.object(
        client._client.models, "generate_content", new_callable=AsyncMock
    ) as mock_gen:
        mock_gen.return_value = mock_response
        result = await client.complete(
            config_key="filtering",
            system_prompt="You are a filter.",
            user_prompt="Is this relevant?",
        )
        assert result == "Test response"
        mock_gen.assert_called_once()


@pytest.mark.asyncio
async def test_complete_with_json_response(client):
    mock_response = MagicMock()
    mock_response.text = '{"score": 8, "reason": "Highly relevant"}'

    with patch.object(
        client._client.models, "generate_content", new_callable=AsyncMock
    ) as mock_gen:
        mock_gen.return_value = mock_response
        result = await client.complete(
            config_key="filtering",
            system_prompt="Score relevance.",
            user_prompt="Article text here.",
            json_response=True,
        )
        assert '"score": 8' in result
