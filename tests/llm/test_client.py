"""Tests for LLM abstraction layer."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from nexus.config.models import ModelsConfig
from nexus.llm.client import LLMClient, _resolve_provider


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
        client._gemini_client.aio.models, "generate_content", new_callable=AsyncMock
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
        client._gemini_client.aio.models, "generate_content", new_callable=AsyncMock
    ) as mock_gen:
        mock_gen.return_value = mock_response
        result = await client.complete(
            config_key="filtering",
            system_prompt="Score relevance.",
            user_prompt="Article text here.",
            json_response=True,
        )
        assert '"score": 8' in result


# ── Provider resolution ──


def test_resolve_provider_openai():
    assert _resolve_provider("gpt-4o") == "openai"
    assert _resolve_provider("gpt-4.1-mini") == "openai"
    assert _resolve_provider("gpt-4.1-nano") == "openai"


def test_resolve_provider_openai_reasoning():
    assert _resolve_provider("o1") == "openai"
    assert _resolve_provider("o3") == "openai"
    assert _resolve_provider("o3-mini") == "openai"
    assert _resolve_provider("o4-mini") == "openai"


def test_resolve_provider_existing():
    assert _resolve_provider("gemini-3-flash-preview") == "gemini"
    assert _resolve_provider("claude-sonnet-4-20250514") == "anthropic"
    assert _resolve_provider("deepseek-chat") == "deepseek"
    assert _resolve_provider("ollama/qwen2") == "ollama"


def test_resolve_provider_unknown():
    with pytest.raises(ValueError, match="Unknown model provider"):
        _resolve_provider("llama-3.1-70b")


# ── OpenAI client initialization ──


def test_openai_client_initialized():
    client = LLMClient(ModelsConfig(), openai_api_key="test-openai-key")
    assert client._openai_client is not None


def test_openai_client_not_initialized_without_key():
    client = LLMClient(ModelsConfig())
    assert client._openai_client is None


# ── LiteLLM provider ──


def test_resolve_provider_litellm():
    assert _resolve_provider("litellm/claude-opus-4-6") == "litellm"
    assert _resolve_provider("litellm/gpt-5.4") == "litellm"
    assert _resolve_provider("litellm/claude-sonnet-4-6") == "litellm"


def test_resolve_provider_litellm_over_gemini():
    """litellm/ prefix takes priority over gemini prefix match."""
    assert _resolve_provider("litellm/gemini-3.1-pro-preview") == "litellm"


def test_litellm_client_initialized():
    client = LLMClient(
        ModelsConfig(),
        litellm_base_url="http://localhost:4000",
        litellm_api_key="test-litellm-key",
    )
    assert client._litellm_client is not None


def test_litellm_client_not_initialized_without_both():
    """Needs BOTH base_url and api_key."""
    client = LLMClient(ModelsConfig(), litellm_api_key="key-only")
    assert client._litellm_client is None
    client = LLMClient(ModelsConfig(), litellm_base_url="http://localhost:4000")
    assert client._litellm_client is None


def test_resolve_runtime_model_hosted_alias(monkeypatch):
    monkeypatch.setenv("LITELLM_MODEL_GPT", "gpt-5.4")
    client = LLMClient(ModelsConfig())
    assert client._resolve_runtime_model("litellm/gpt") == "gpt-5.4"


def test_resolve_runtime_model_env_reference(monkeypatch):
    monkeypatch.setenv("NEXUS_TEST_MODEL", "gpt-5.4")
    client = LLMClient(ModelsConfig())
    assert client._resolve_runtime_model("env:NEXUS_TEST_MODEL") == "gpt-5.4"


def test_get_litellm_client_refreshes_when_proxy_creds_rotate(monkeypatch):
    monkeypatch.setenv("LITELLM_PROXY_URL", "https://proxy.example/v1")
    monkeypatch.setenv("LITELLM_PROXY_API_KEY", "key-1")

    client = LLMClient(ModelsConfig())
    first_client = MagicMock()
    second_client = MagicMock()

    with patch("openai.AsyncOpenAI", side_effect=[first_client, second_client]) as mock_openai:
        assert client._get_litellm_client() is first_client
        assert client._get_litellm_client() is first_client
        monkeypatch.setenv("LITELLM_PROXY_API_KEY", "key-2")
        assert client._get_litellm_client() is second_client

    assert mock_openai.call_count == 2


@pytest.mark.asyncio
async def test_complete_uses_runtime_model_for_litellm_alias(monkeypatch):
    monkeypatch.setenv("LITELLM_MODEL_GPT", "gpt-5.4")
    client = LLMClient(ModelsConfig(filtering="litellm/gpt"))
    client._complete_litellm = AsyncMock(return_value=("ok", 11, 7))

    result = await client.complete(
        config_key="filtering",
        system_prompt="system",
        user_prompt="user",
    )

    assert result == "ok"
    client._complete_litellm.assert_awaited_once_with("gpt-5.4", "system", "user", False, timeout_s=60.0)


def test_refresh_hosted_env_from_env_file(tmp_path, monkeypatch):
    env_file = tmp_path / "runtime.env"
    env_file.write_text(
        "LITELLM_PROXY_URL=https://proxy.example/v1\n"
        "LITELLM_PROXY_API_KEY=fresh-key\n"
        "LITELLM_MODEL_GPT=gpt-5.4\n"
    )
    monkeypatch.setenv("NEXUS_ENV_FILE", str(env_file))
    monkeypatch.delenv("LITELLM_PROXY_URL", raising=False)
    monkeypatch.delenv("LITELLM_PROXY_API_KEY", raising=False)

    client = LLMClient(ModelsConfig())
    assert client._refresh_hosted_env() is True
    assert client._get_litellm_client() is not None


@pytest.mark.asyncio
async def test_complete_litellm_retries_after_auth_refresh(tmp_path, monkeypatch):
    env_file = tmp_path / "runtime.env"
    env_file.write_text(
        "LITELLM_PROXY_URL=https://proxy.example/v1\n"
        "LITELLM_PROXY_API_KEY=fresh-key\n"
    )
    monkeypatch.setenv("NEXUS_ENV_FILE", str(env_file))
    monkeypatch.setenv("LITELLM_PROXY_URL", "https://proxy.example/v1")
    monkeypatch.setenv("LITELLM_PROXY_API_KEY", "stale-key")

    client = LLMClient(ModelsConfig())

    class FakeAuthError(Exception):
        status_code = 401

    success = MagicMock()
    success.choices = [MagicMock(message=MagicMock(content="ok"))]
    success.usage = MagicMock(prompt_tokens=3, completion_tokens=2)

    stale_client = MagicMock()
    stale_client.chat.completions.create = AsyncMock(side_effect=FakeAuthError("stale"))
    fresh_client = MagicMock()
    fresh_client.chat.completions.create = AsyncMock(return_value=success)

    with patch.object(client, "_get_litellm_client", side_effect=[stale_client, fresh_client]) as mock_get_client:
        result = await client._complete_litellm("gpt-5.4", "system", "user", False)

    assert result == ("ok", 3, 2)
    assert mock_get_client.call_args_list[1].kwargs["force_refresh"] is True
