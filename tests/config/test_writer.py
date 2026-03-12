"""Tests for config writer module."""

import stat

from nexus.config.writer import write_config, write_env
from nexus.config.loader import load_config


def test_write_config_yaml(tmp_path):
    """write_config should produce valid YAML loadable by load_config."""
    data_dir = tmp_path / "data"
    config_dict = {
        "preset": "balanced",
        "user": {"name": "TestUser", "timezone": "UTC", "output_language": "en"},
        "topics": [{"name": "AI/ML Research", "priority": "high"}],
        "briefing": {"schedule": "06:00"},
        "audio": {"enabled": True},
        "breaking_news": {"enabled": True, "threshold": 7},
        "telegram": {"enabled": False},
    }

    path = write_config(data_dir, config_dict)

    assert path.exists()
    assert path.name == "config.yaml"
    # Should be loadable by the real config loader
    config = load_config(path)
    assert config.user.name == "TestUser"
    assert config.user.timezone == "UTC"
    assert len(config.topics) == 1
    assert config.topics[0].name == "AI/ML Research"


def test_write_env_file(tmp_path):
    """write_env should write API keys to .env file."""
    keys = {
        "GEMINI_API_KEY": "test-gemini-key",
        "OPENAI_API_KEY": "test-openai-key",
    }

    path = write_env(tmp_path, keys)

    assert path.exists()
    content = path.read_text()
    assert "GEMINI_API_KEY=test-gemini-key" in content
    assert "OPENAI_API_KEY=test-openai-key" in content


def test_write_env_preserves_existing(tmp_path):
    """write_env should preserve existing keys when adding new ones."""
    # Pre-populate .env
    env_path = tmp_path / ".env"
    env_path.write_text("EXISTING_KEY=existing-value\nGEMINI_API_KEY=old-key\n")

    # Write new keys (should update GEMINI, keep EXISTING)
    keys = {"GEMINI_API_KEY": "new-gemini-key", "OPENAI_API_KEY": "new-oai-key"}
    write_env(tmp_path, keys)

    content = env_path.read_text()
    assert "EXISTING_KEY=existing-value" in content
    assert "GEMINI_API_KEY=new-gemini-key" in content
    assert "OPENAI_API_KEY=new-oai-key" in content
    # Old key should be replaced
    assert "old-key" not in content


def test_write_env_empty_keys(tmp_path):
    """write_env with empty dict should not fail."""
    path = write_env(tmp_path, {})
    # Should either create empty file or not create at all
    assert not path.exists() or path.read_text().strip() == ""


def test_write_config_creates_directories(tmp_path):
    """write_config should create data_dir if it doesn't exist."""
    data_dir = tmp_path / "nested" / "data"
    config_dict = {
        "user": {"name": "Test", "timezone": "UTC"},
        "topics": [{"name": "Test Topic"}],
    }

    path = write_config(data_dir, config_dict)
    assert path.exists()
    assert data_dir.exists()


def test_write_config_restricts_permissions(tmp_path):
    """config.yaml should be written with owner-only permissions."""
    path = write_config(tmp_path / "data", {"topics": [{"name": "Test Topic"}]})
    mode = stat.S_IMODE(path.stat().st_mode)
    assert mode == 0o600
