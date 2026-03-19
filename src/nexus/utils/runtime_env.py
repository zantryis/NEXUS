"""Helpers for loading and rebuilding the runtime environment from .env."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

from dotenv import dotenv_values, load_dotenv


def runtime_env_path(root: str | Path | None = None) -> Path:
    """Return the canonical runtime .env path.

    Callers may pass either the project root or the data directory.
    """
    if root is None:
        return Path(".env")
    root_path = Path(root)
    if root_path.name == "data":
        root_path = root_path.parent
    return root_path / ".env"


def load_runtime_env(env_path: str | Path | None = None) -> Path:
    """Load .env into the current process, overriding stale values."""
    path = Path(env_path) if env_path is not None else runtime_env_path()
    load_dotenv(path, override=True)
    return path


def build_runtime_env(
    env_path: str | Path | None = None,
    *,
    base_env: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Merge the latest .env values onto a process environment snapshot."""
    path = Path(env_path) if env_path is not None else runtime_env_path()
    merged = dict(base_env or os.environ)
    if path.exists():
        for key, value in dotenv_values(path).items():
            if value is not None:
                merged[key] = value
    return merged
