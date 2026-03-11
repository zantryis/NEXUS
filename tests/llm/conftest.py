"""Shared fixtures for LLM tests."""

import sys
from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def mock_google_genai():
    """Mock google.genai to avoid cryptography import issues in CI."""
    mock_genai_module = MagicMock()
    mock_google = MagicMock()
    mock_google.genai = mock_genai_module

    # Store originals
    orig_google = sys.modules.get("google")
    orig_genai = sys.modules.get("google.genai")
    orig_types = sys.modules.get("google.genai.types")

    # Inject mocks before any test code runs
    sys.modules["google"] = mock_google
    sys.modules["google.genai"] = mock_genai_module
    sys.modules["google.genai.types"] = MagicMock()

    yield mock_genai_module

    # Restore
    if orig_google is None:
        sys.modules.pop("google", None)
    else:
        sys.modules["google"] = orig_google
    if orig_genai is None:
        sys.modules.pop("google.genai", None)
    else:
        sys.modules["google.genai"] = orig_genai
    if orig_types is None:
        sys.modules.pop("google.genai.types", None)
    else:
        sys.modules["google.genai.types"] = orig_types
