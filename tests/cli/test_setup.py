"""Tests for the setup wizard."""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from nexus.cli.setup import run_setup, TOPIC_CHOICES


def test_topic_choices_not_empty():
    assert len(TOPIC_CHOICES) > 0
    for slug, name in TOPIC_CHOICES:
        assert slug
        assert name


def test_run_setup_creates_config(tmp_path):
    """Setup wizard writes config.yaml and .env."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    inputs = iter([
        "2",          # preset: cheap
        "sk-fake123", # api key
        "1,3",        # topics: first and third
        "Tester",     # user name
        "UTC",        # timezone
    ])
    with patch("builtins.input", lambda prompt="": next(inputs)):
        run_setup(data_dir)

    assert (data_dir / "config.yaml").exists()
    assert (tmp_path / ".env").exists()

    import yaml
    config = yaml.safe_load((data_dir / "config.yaml").read_text())
    assert config["preset"] == "cheap"
    assert config["user"]["name"] == "Tester"
    assert len(config["topics"]) == 2


def test_run_setup_balanced_preset_needs_gemini_key(tmp_path):
    """Balanced preset should ask for GEMINI_API_KEY."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    inputs = iter([
        "3",          # balanced
        "AIza-fake",  # gemini key
        "1",          # topic
        "Tester",
        "UTC",
    ])
    with patch("builtins.input", lambda prompt="": next(inputs)):
        run_setup(data_dir)

    env_text = (tmp_path / ".env").read_text()
    assert "GEMINI_API_KEY=AIza-fake" in env_text


def test_run_setup_free_preset_no_key(tmp_path):
    """Free preset should not ask for an API key."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    inputs = iter([
        "1",          # free
        "1",          # topic
        "Tester",
        "UTC",
    ])
    with patch("builtins.input", lambda prompt="": next(inputs)):
        run_setup(data_dir)

    config = __import__("yaml").safe_load((data_dir / "config.yaml").read_text())
    assert config["preset"] == "free"
    # .env may not exist or have no API keys
    env_path = tmp_path / ".env"
    if env_path.exists():
        assert "API_KEY" not in env_path.read_text()
