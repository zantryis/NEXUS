"""Tests for LLM client provider resolution and usage tracking."""

import pytest
from nexus.llm.client import _resolve_provider, UsageTracker


def test_resolve_gemini():
    assert _resolve_provider("gemini-3-flash-preview") == "gemini"
    assert _resolve_provider("gemini-3.1-pro-preview") == "gemini"


def test_resolve_anthropic():
    assert _resolve_provider("claude-sonnet-4-6") == "anthropic"
    assert _resolve_provider("claude-opus-4-6") == "anthropic"


def test_resolve_deepseek():
    assert _resolve_provider("deepseek-chat") == "deepseek"
    assert _resolve_provider("deepseek-reasoner") == "deepseek"


def test_resolve_openai():
    assert _resolve_provider("gpt-4o") == "openai"
    assert _resolve_provider("gpt-4.1-mini") == "openai"
    assert _resolve_provider("o3-mini") == "openai"


def test_resolve_unknown():
    with pytest.raises(ValueError, match="Unknown model provider"):
        _resolve_provider("totally-unknown-model")


def test_usage_tracker_accumulates():
    tracker = UsageTracker()
    tracker.record("gemini", "gemini-3-flash", "filtering", 100, 50, 1.5)
    tracker.record("gemini", "gemini-3-flash", "filtering", 200, 80, 2.0)
    tracker.record("deepseek", "deepseek-chat", "synthesis", 500, 300, 0.8)

    s = tracker.summary()
    assert s["total_calls"] == 3
    assert s["total_input_tokens"] == 800
    assert s["total_output_tokens"] == 430
    assert s["by_provider"]["gemini"]["calls"] == 2
    assert s["by_provider"]["gemini"]["input_tokens"] == 300
    assert s["by_provider"]["deepseek"]["calls"] == 1
    assert s["by_config_key"]["filtering"]["calls"] == 2
    assert s["by_config_key"]["synthesis"]["input_tokens"] == 500


def test_usage_tracker_reset():
    tracker = UsageTracker()
    tracker.record("gemini", "gemini-3-flash", "filtering", 100, 50, 1.0)
    tracker.reset()
    s = tracker.summary()
    assert s["total_calls"] == 0
    assert s["total_input_tokens"] == 0
