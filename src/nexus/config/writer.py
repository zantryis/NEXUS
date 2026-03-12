"""Write config.yaml and .env files from structured data."""

import os
from pathlib import Path

import yaml


def write_config(data_dir: Path, config_dict: dict) -> Path:
    """Write config.yaml from a dict. Creates data_dir if needed. Returns path."""
    data_dir.mkdir(parents=True, exist_ok=True)
    config_path = data_dir / "config.yaml"
    config_path.write_text(yaml.dump(config_dict, default_flow_style=False, sort_keys=False))
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
