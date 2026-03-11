"""Tests for Ollama provider integration."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from nexus.config.models import ModelsConfig
from nexus.llm.client import LLMClient, _resolve_provider


def test_resolve_provider_ollama():
    """Model names starting with 'ollama/' resolve to 'ollama' provider."""
    assert _resolve_provider("ollama/qwen2") == "ollama"
    assert _resolve_provider("ollama/llama3") == "ollama"


@pytest.fixture
def ollama_config():
    return ModelsConfig(
        discovery="ollama/qwen2",
        filtering="ollama/qwen2",
        synthesis="ollama/qwen2",
        dialogue_script="ollama/qwen2",
        knowledge_summary="ollama/qwen2",
        breaking_news="ollama/qwen2",
        agent="ollama/qwen2",
    )


@pytest.fixture
def ollama_client(ollama_config):
    return LLMClient(ollama_config)


async def test_complete_ollama_success(ollama_client):
    """Mock httpx POST, verify request format and response parsing."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "message": {"content": "Hello from Ollama"},
        "prompt_eval_count": 10,
        "eval_count": 20,
    }
    mock_response.raise_for_status = MagicMock()

    mock_httpx_client = AsyncMock()
    mock_httpx_client.post = AsyncMock(return_value=mock_response)
    mock_httpx_client.__aenter__ = AsyncMock(return_value=mock_httpx_client)
    mock_httpx_client.__aexit__ = AsyncMock(return_value=None)

    with patch("nexus.llm.client.httpx.AsyncClient", return_value=mock_httpx_client):
        result = await ollama_client.complete(
            config_key="filtering",
            system_prompt="You are a filter.",
            user_prompt="Test input",
        )

    assert result == "Hello from Ollama"
    # Verify the POST was called with correct args
    call_args = mock_httpx_client.post.call_args
    assert "/api/chat" in call_args[0][0]
    body = call_args[1]["json"]
    assert body["model"] == "qwen2"
    assert body["stream"] is False
    assert len(body["messages"]) == 2
    assert body["messages"][0]["role"] == "system"
    assert body["messages"][1]["role"] == "user"
    assert "format" not in body


async def test_complete_ollama_connection_error(ollama_client):
    """ConnectionError raises RuntimeError with clear message."""
    import httpx

    mock_httpx_client = AsyncMock()
    mock_httpx_client.post = AsyncMock(
        side_effect=httpx.ConnectError("Connection refused")
    )
    mock_httpx_client.__aenter__ = AsyncMock(return_value=mock_httpx_client)
    mock_httpx_client.__aexit__ = AsyncMock(return_value=None)

    with patch("nexus.llm.client.httpx.AsyncClient", return_value=mock_httpx_client):
        with pytest.raises(RuntimeError, match="Ollama not running"):
            await ollama_client.complete(
                config_key="filtering",
                system_prompt="Test",
                user_prompt="Test",
            )


async def test_complete_ollama_json_mode(ollama_client):
    """json_response=True adds format: 'json' to request body."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "message": {"content": '{"score": 8}'},
        "prompt_eval_count": 5,
        "eval_count": 10,
    }
    mock_response.raise_for_status = MagicMock()

    mock_httpx_client = AsyncMock()
    mock_httpx_client.post = AsyncMock(return_value=mock_response)
    mock_httpx_client.__aenter__ = AsyncMock(return_value=mock_httpx_client)
    mock_httpx_client.__aexit__ = AsyncMock(return_value=None)

    with patch("nexus.llm.client.httpx.AsyncClient", return_value=mock_httpx_client):
        result = await ollama_client.complete(
            config_key="filtering",
            system_prompt="Score this.",
            user_prompt="Article text",
            json_response=True,
        )

    assert result == '{"score": 8}'
    body = mock_httpx_client.post.call_args[1]["json"]
    assert body["format"] == "json"
