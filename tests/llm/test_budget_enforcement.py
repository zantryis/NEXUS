"""Tests for budget enforcement integration in LLMClient."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from nexus.config.models import ModelsConfig, BudgetConfig
from nexus.llm.client import LLMClient
from nexus.llm.budget import BudgetExceededError, BudgetDegradedError


@pytest.fixture
def models_config():
    # Use deepseek models to avoid Gemini client init (needs cryptography)
    return ModelsConfig(
        filtering="deepseek-chat",
        synthesis="deepseek-chat",
        dialogue_script="deepseek-chat",
        knowledge_summary="deepseek-chat",
        breaking_news="deepseek-chat",
        agent="deepseek-chat",
        discovery="deepseek-chat",
    )


@pytest.fixture
def client_no_budget(models_config):
    return LLMClient(models_config, deepseek_api_key="test-key")


@pytest.fixture
def client_with_budget(models_config):
    return LLMClient(
        models_config,
        deepseek_api_key="test-key",
        budget_config=BudgetConfig(daily_limit_usd=1.00, warning_threshold_usd=0.50),
    )


@pytest.fixture
def client_stop_all(models_config):
    return LLMClient(
        models_config,
        deepseek_api_key="test-key",
        budget_config=BudgetConfig(degradation_strategy="stop_all"),
    )


def _mock_deepseek_response(text="Test response", in_tok=100, out_tok=50):
    """Helper to create a mock DeepSeek (OpenAI-compatible) response."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = text
    usage = MagicMock()
    usage.prompt_tokens = in_tok
    usage.completion_tokens = out_tok
    mock_response.usage = usage
    return mock_response


async def test_llm_client_no_budget(client_no_budget):
    """LLMClient works normally without budget config."""
    mock_response = _mock_deepseek_response()
    with patch.object(
        client_no_budget._deepseek_client.chat.completions,
        "create",
        new_callable=AsyncMock,
    ) as mock_gen:
        mock_gen.return_value = mock_response
        result = await client_no_budget.complete(
            config_key="filtering",
            system_prompt="Filter.",
            user_prompt="Test article.",
        )
        assert result == "Test response"


async def test_llm_client_budget_ok(client_with_budget):
    """Under budget, complete works normally."""
    mock_response = _mock_deepseek_response()
    with patch.object(
        client_with_budget._deepseek_client.chat.completions,
        "create",
        new_callable=AsyncMock,
    ) as mock_gen:
        mock_gen.return_value = mock_response
        result = await client_with_budget.complete(
            config_key="filtering",
            system_prompt="Filter.",
            user_prompt="Test article.",
        )
        assert result == "Test response"


async def test_llm_client_budget_degraded(client_with_budget):
    """Raises BudgetDegradedError for expensive key when over limit."""
    client_with_budget._budget_guard.record_cost(1.50)

    with pytest.raises(BudgetDegradedError):
        await client_with_budget.complete(
            config_key="synthesis",
            system_prompt="Synthesize.",
            user_prompt="Data.",
        )


async def test_llm_client_budget_blocked(client_stop_all):
    """Raises BudgetExceededError when stop_all and over limit."""
    client_stop_all._budget_guard.record_cost(1.50)

    with pytest.raises(BudgetExceededError):
        await client_stop_all.complete(
            config_key="filtering",
            system_prompt="Filter.",
            user_prompt="Test.",
        )


async def test_llm_client_records_cost(client_with_budget):
    """After complete(), budget guard has updated spend."""
    mock_response = _mock_deepseek_response(in_tok=1000, out_tok=500)
    with patch.object(
        client_with_budget._deepseek_client.chat.completions,
        "create",
        new_callable=AsyncMock,
    ) as mock_gen:
        mock_gen.return_value = mock_response
        await client_with_budget.complete(
            config_key="filtering",
            system_prompt="Filter.",
            user_prompt="Test.",
        )

    assert client_with_budget.today_spend > 0.0


def test_llm_client_budget_status_property(client_with_budget):
    """budget_status returns correct status string."""
    assert client_with_budget.budget_status == "ok"
    client_with_budget._budget_guard.record_cost(0.60)
    assert client_with_budget.budget_status == "warning"
    client_with_budget._budget_guard.record_cost(0.50)
    assert client_with_budget.budget_status == "over_limit"


def test_llm_client_today_spend_property(client_with_budget):
    """today_spend returns 0.0 initially."""
    assert client_with_budget.today_spend == 0.0


def test_llm_client_no_budget_properties(client_no_budget):
    """Properties work even without budget config."""
    assert client_no_budget.budget_status == "ok"
    assert client_no_budget.today_spend == 0.0
