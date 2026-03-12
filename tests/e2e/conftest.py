"""Shared fixtures for E2E tests. Skip if no API keys available."""

import os
import pytest
from pathlib import Path


def _has_api_keys() -> bool:
    """Check if any LLM API keys are available."""
    return bool(
        os.getenv("GEMINI_API_KEY")
        or os.getenv("DEEPSEEK_API_KEY")
        or os.getenv("deepseek")
    )


# Skip entire module if no API keys
pytestmark = pytest.mark.e2e


@pytest.fixture(autouse=True)
def skip_without_keys():
    if not _has_api_keys():
        pytest.skip("No API keys available for E2E tests")


@pytest.fixture
def smoke_data_dir(tmp_path):
    """Temporary data directory for smoke tests."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return data_dir
