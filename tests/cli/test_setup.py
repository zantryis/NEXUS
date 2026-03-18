"""Tests for the setup wizard."""

import yaml
from unittest.mock import patch

from nexus.cli.setup import run_setup, PRESET_INFO, TOPIC_CHOICES


def test_topic_choices_not_empty():
    assert len(TOPIC_CHOICES) > 0
    for slug, name in TOPIC_CHOICES:
        assert slug
        assert name


def test_preset_info_has_all_providers():
    """Presets cover free, DeepSeek, Gemini, OpenAI, and Anthropic."""
    preset_names = [p[0] for p in PRESET_INFO]
    assert "free" in preset_names
    assert "cheap" in preset_names
    assert "balanced" in preset_names
    assert any("openai" in n for n in preset_names)
    assert any("anthropic" in n for n in preset_names)


def test_run_setup_creates_config(tmp_path):
    """Setup wizard writes config.yaml and .env."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    inputs = iter([
        "2",          # preset: cheap
        "sk-fake123", # api key
        "",           # no custom topics (skip)
        "1,3",        # pre-configured: first and third
        "Tester",     # user name
        "UTC",        # timezone
    ])
    with patch("builtins.input", lambda prompt="": next(inputs)):
        run_setup(data_dir)

    assert (data_dir / "config.yaml").exists()
    assert (tmp_path / ".env").exists()

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
        "",           # no custom topics
        "1",          # pre-configured: first
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
        "",           # no custom topics
        "1",          # pre-configured: first
        "Tester",
        "UTC",
    ])
    with patch("builtins.input", lambda prompt="": next(inputs)):
        run_setup(data_dir)

    config = yaml.safe_load((data_dir / "config.yaml").read_text())
    assert config["preset"] == "free"
    env_path = tmp_path / ".env"
    if env_path.exists():
        assert "API_KEY" not in env_path.read_text()


def test_setup_invalid_preset_retries(tmp_path):
    """Invalid preset input should re-prompt, not crash."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    inputs = iter([
        "abc",        # invalid first try
        "99",         # out of range
        "2",          # valid
        "sk-key",     # api key
        "",           # no custom topics
        "1",          # pre-configured: first
        "Tester",
        "UTC",
    ])
    with patch("builtins.input", lambda prompt="": next(inputs)):
        run_setup(data_dir)

    config = yaml.safe_load((data_dir / "config.yaml").read_text())
    assert config["preset"] == "cheap"


def test_setup_invalid_topic_input_skipped(tmp_path):
    """Invalid topic numbers should be skipped, not crash."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    inputs = iter([
        "1",          # free preset
        "",           # no custom topics
        "1,abc,99",   # mix of valid, invalid, out-of-range
        "Tester",
        "UTC",
    ])
    with patch("builtins.input", lambda prompt="": next(inputs)):
        run_setup(data_dir)

    config = yaml.safe_load((data_dir / "config.yaml").read_text())
    # Should have 1 valid topic (first one), invalid entries skipped
    assert len(config["topics"]) >= 1


def test_setup_custom_topic(tmp_path):
    """User can add a custom topic name."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    inputs = iter([
        "1",                    # free preset
        "Horticulture",         # custom topic
        "",                     # done adding custom topics
        "",                     # skip pre-configured
        "Tester",
        "UTC",
    ])
    with patch("builtins.input", lambda prompt="": next(inputs)):
        run_setup(data_dir)

    config = yaml.safe_load((data_dir / "config.yaml").read_text())
    topic_names = [t["name"] for t in config["topics"]]
    assert "Horticulture" in topic_names


def test_setup_openai_preset(tmp_path):
    """OpenAI preset should ask for OPENAI_API_KEY."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    # Find the index of an openai preset
    openai_idx = None
    for i, (name, _, _, _) in enumerate(PRESET_INFO):
        if "openai" in name:
            openai_idx = i + 1
            break
    assert openai_idx is not None, "No OpenAI preset found"

    inputs = iter([
        str(openai_idx),
        "sk-openai-fake",   # api key
        "",                 # no custom topics
        "1",                # pre-configured: first
        "Tester",
        "UTC",
    ])
    with patch("builtins.input", lambda prompt="": next(inputs)):
        run_setup(data_dir)

    env_text = (tmp_path / ".env").read_text()
    assert "OPENAI_API_KEY=sk-openai-fake" in env_text


def test_setup_preserves_existing_env(tmp_path):
    """Setup should append to existing .env, not clobber it."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    env_path = tmp_path / ".env"
    env_path.write_text("TELEGRAM_BOT_TOKEN=existing-token\n")

    inputs = iter([
        "2",          # cheap (DeepSeek)
        "sk-deep",    # api key
        "",           # no custom topics
        "1",          # pre-configured: first
        "Tester",
        "UTC",
    ])
    with patch("builtins.input", lambda prompt="": next(inputs)):
        run_setup(data_dir)

    env_text = env_path.read_text()
    assert "TELEGRAM_BOT_TOKEN=existing-token" in env_text
    assert "DEEPSEEK_API_KEY=sk-deep" in env_text


def test_setup_empty_api_key_reprompts(tmp_path):
    """Empty API key should re-prompt when key is required."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    inputs = iter([
        "2",          # cheap (needs key)
        "",           # empty first try
        "sk-real",    # valid second try
        "",           # no custom topics
        "1",          # pre-configured: first
        "Tester",
        "UTC",
    ])
    with patch("builtins.input", lambda prompt="": next(inputs)):
        run_setup(data_dir)

    env_text = (tmp_path / ".env").read_text()
    assert "DEEPSEEK_API_KEY=sk-real" in env_text
