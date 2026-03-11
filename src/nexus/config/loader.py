"""Load and validate nexus configuration from YAML."""

from pathlib import Path
import yaml
from .models import NexusConfig
from .presets import apply_preset


def load_config(path: Path) -> NexusConfig:
    """Load config from a YAML file, validate with Pydantic.

    If ``preset`` is set, use it as the base ModelsConfig,
    then apply any explicit ``models:`` overrides on top.
    """
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    raw = yaml.safe_load(path.read_text())

    preset_name = raw.get("preset")
    if preset_name:
        model_overrides = raw.get("models")
        resolved = apply_preset(preset_name, overrides=model_overrides)
        raw["models"] = resolved.model_dump()

    return NexusConfig(**raw)
