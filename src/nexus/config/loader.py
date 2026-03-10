"""Load and validate nexus configuration from YAML."""

from pathlib import Path
import yaml
from .models import NexusConfig


def load_config(path: Path) -> NexusConfig:
    """Load config from a YAML file, validate with Pydantic."""
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    raw = yaml.safe_load(path.read_text())
    return NexusConfig(**raw)
