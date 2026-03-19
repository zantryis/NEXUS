"""Write config.yaml and .env files from structured data."""

import os
import stat
from pathlib import Path

import yaml

DEFAULT_BOOTSTRAP_TOPIC = {"name": "AI/ML Research", "priority": "high"}
DEFAULT_BOOTSTRAP_BRIEFING = {"schedule": "06:00", "style": "analytical"}
DEFAULT_BOOTSTRAP_BREAKING_NEWS = {"enabled": True, "threshold": 7}
DEFAULT_BOOTSTRAP_TELEGRAM = {"enabled": True}
DEFAULT_BOOTSTRAP_SOURCES = {"discover_new_sources": True}


def build_initial_config(
    *,
    preset: str,
    topics: list[dict] | None = None,
    user_name: str = "User",
    timezone: str = "UTC",
    output_language: str = "en",
) -> dict:
    """Build the canonical first-run config used by setup flows."""
    selected_topics = [dict(topic) for topic in (topics or [DEFAULT_BOOTSTRAP_TOPIC])]
    return {
        "preset": preset,
        "user": {
            "name": user_name,
            "timezone": timezone,
            "output_language": output_language,
        },
        "topics": selected_topics,
        "briefing": dict(DEFAULT_BOOTSTRAP_BRIEFING),
        "audio": {"enabled": preset != "free"},
        "breaking_news": dict(DEFAULT_BOOTSTRAP_BREAKING_NEWS),
        "telegram": dict(DEFAULT_BOOTSTRAP_TELEGRAM),
        "sources": dict(DEFAULT_BOOTSTRAP_SOURCES),
    }


def write_config(data_dir: Path, config_dict: dict) -> Path:
    """Write config.yaml from a dict. Creates data_dir if needed. Returns path."""
    data_dir.mkdir(parents=True, exist_ok=True)
    config_path = data_dir / "config.yaml"
    config_path.write_text(yaml.dump(config_dict, default_flow_style=False, sort_keys=False))
    os.chmod(config_path, stat.S_IRUSR | stat.S_IWUSR)
    return config_path


def write_env(project_root: Path, keys: dict[str, str]) -> Path:
    """Merge API keys into .env, preserving existing entries. Returns path."""
    env_path = project_root / ".env"

    # Load existing entries
    existing: dict[str, str] = {}
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                existing[k.strip()] = v.strip()

    # Merge new keys (overwrite existing, add new)
    existing.update(keys)

    if existing:
        env_text = "\n".join(f"{k}={v}" for k, v in existing.items()) + "\n"
        env_path.write_text(env_text)
        # Restrict permissions — .env contains API keys
        os.chmod(env_path, 0o600)

    return env_path
