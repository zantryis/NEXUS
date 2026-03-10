"""Tests for config YAML loader."""

import pytest
from pathlib import Path
from nexus.config.loader import load_config


SAMPLE_YAML = """
user:
  name: "Tristan"
  timezone: "America/Denver"
  output_language: "en"

briefing:
  schedule: "06:00"
  format: "two-host-dialogue"

topics:
  - name: "Iran-US Relations"
    priority: high
    subtopics: ["sanctions", "nuclear program"]
    source_languages: ["en", "fa"]
    perspective_diversity: high

  - name: "AI/ML Research"
    priority: low
    subtopics: ["agents", "benchmarks"]
    source_languages: ["en"]

models:
  filtering: "gemini-3-flash-preview"
  synthesis: "gemini-3.1-pro-preview"
"""


def test_load_config_from_file(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(SAMPLE_YAML)

    config = load_config(config_file)
    assert config.user.name == "Tristan"
    assert config.user.timezone == "America/Denver"
    assert len(config.topics) == 2
    assert config.topics[0].name == "Iran-US Relations"
    assert config.topics[0].source_languages == ["en", "fa"]
    assert config.models.filtering == "gemini-3-flash-preview"


def test_load_config_missing_file():
    with pytest.raises(FileNotFoundError):
        load_config(Path("/nonexistent/config.yaml"))


def test_load_config_minimal(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text('user:\n  name: "Test"\n')

    config = load_config(config_file)
    assert config.user.name == "Test"
    assert config.briefing.schedule == "06:00"
    assert config.topics == []
