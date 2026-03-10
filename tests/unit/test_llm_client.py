"""Tests for LLM client provider resolution."""

import pytest
from nexus.llm.client import _resolve_provider


def test_resolve_gemini():
    assert _resolve_provider("gemini-3-flash-preview") == "gemini"
    assert _resolve_provider("gemini-3.1-pro-preview") == "gemini"


def test_resolve_anthropic():
    assert _resolve_provider("claude-sonnet-4-6") == "anthropic"
    assert _resolve_provider("claude-opus-4-6") == "anthropic"


def test_resolve_unknown():
    with pytest.raises(ValueError, match="Unknown model provider"):
        _resolve_provider("gpt-4o")
