"""Tests for cost tracking module."""

import pytest
from nexus.llm.cost import estimate_cost, cost_summary
from nexus.llm.client import UsageTracker


def test_estimate_cost_known_model():
    """Gemini flash: known pricing applies correctly."""
    # 1M input tokens at $0.10, 1M output tokens at $0.40
    cost = estimate_cost("gemini-3-flash-preview", 1_000_000, 1_000_000)
    assert cost == pytest.approx(0.10 + 0.40)

    # Smaller amounts
    cost = estimate_cost("gemini-3-flash-preview", 1000, 500)
    expected = (1000 / 1_000_000) * 0.10 + (500 / 1_000_000) * 0.40
    assert cost == pytest.approx(expected)


def test_estimate_cost_unknown_model():
    """Unknown models return 0."""
    assert estimate_cost("some-unknown-model", 1000, 500) == 0.0


def test_estimate_cost_ollama():
    """Ollama models (prefixed with ollama/) return 0."""
    assert estimate_cost("ollama/qwen2", 10000, 5000) == 0.0
    assert estimate_cost("ollama/llama3", 10000, 5000) == 0.0


def test_cost_summary_empty():
    """Empty call list returns zeroed summary."""
    result = cost_summary([])
    assert result["total_usd"] == 0.0
    assert result["by_provider"] == {}
    assert result["by_config_key"] == {}
    assert result["by_model"] == {}


def test_cost_summary_mixed_providers():
    """Mixed providers aggregate correctly."""
    calls = [
        {
            "provider": "gemini",
            "model": "gemini-3-flash-preview",
            "config_key": "filtering",
            "input_tokens": 1000,
            "output_tokens": 500,
        },
        {
            "provider": "deepseek",
            "model": "deepseek-chat",
            "config_key": "synthesis",
            "input_tokens": 2000,
            "output_tokens": 1000,
        },
        {
            "provider": "gemini",
            "model": "gemini-3-flash-preview",
            "config_key": "filtering",
            "input_tokens": 500,
            "output_tokens": 200,
        },
    ]
    result = cost_summary(calls)

    # Total should be sum of all individual costs
    assert result["total_usd"] > 0
    assert "gemini" in result["by_provider"]
    assert "deepseek" in result["by_provider"]
    assert "filtering" in result["by_config_key"]
    assert "synthesis" in result["by_config_key"]
    assert "gemini-3-flash-preview" in result["by_model"]
    assert "deepseek-chat" in result["by_model"]

    # Verify gemini filtering cost (1500 in, 700 out)
    gemini_cost = estimate_cost("gemini-3-flash-preview", 1500, 700)
    deepseek_cost = estimate_cost("deepseek-chat", 2000, 1000)
    assert result["total_usd"] == pytest.approx(gemini_cost + deepseek_cost)


def test_estimate_cost_litellm_prefix():
    """LiteLLM-prefixed models use the underlying model's pricing."""
    base_cost = estimate_cost("claude-opus-4-6", 1_000_000, 1_000_000)
    litellm_cost = estimate_cost("litellm/claude-opus-4-6", 1_000_000, 1_000_000)
    assert litellm_cost == base_cost
    assert litellm_cost > 0  # sanity: opus is expensive

    # Also works for other providers behind litellm
    assert estimate_cost("litellm/gpt-5.4", 1000, 500) == estimate_cost("gpt-5.4", 1000, 500)
    assert estimate_cost("litellm/claude-sonnet-4-6", 1000, 500) == estimate_cost("claude-sonnet-4-6", 1000, 500)


def test_estimate_cost_litellm_unknown():
    """LiteLLM-prefixed unknown model returns 0."""
    assert estimate_cost("litellm/some-unknown-model", 1000, 500) == 0.0


def test_usage_tracker_cost_summary():
    """UsageTracker.cost_summary() delegates to cost module."""
    tracker = UsageTracker()
    tracker.record("gemini", "gemini-3-flash-preview", "filtering", 1000, 500, 0.5)
    tracker.record("deepseek", "deepseek-chat", "synthesis", 2000, 1000, 1.0)

    result = tracker.cost_summary()
    assert result["total_usd"] > 0
    assert "gemini" in result["by_provider"]
    assert "deepseek" in result["by_provider"]
