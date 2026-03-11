"""Tests for model presets."""

import pytest
from nexus.config.models import ModelsConfig, NexusConfig
from nexus.config.presets import PRESETS, apply_preset


def test_all_presets_valid():
    """Each preset returns a valid ModelsConfig."""
    for name, config in PRESETS.items():
        assert isinstance(config, ModelsConfig), f"Preset '{name}' is not a ModelsConfig"
        # All fields should be non-empty strings
        for field in ModelsConfig.model_fields:
            val = getattr(config, field)
            assert isinstance(val, str) and len(val) > 0, (
                f"Preset '{name}' field '{field}' is empty or not a string"
            )


def test_apply_preset_with_overrides():
    """Override a single field from a preset."""
    config = apply_preset("balanced", overrides={"filtering": "deepseek-chat"})
    assert config.filtering == "deepseek-chat"
    # Other fields should remain from preset
    assert config.synthesis == "gemini-3.1-pro-preview"
    assert config.discovery == "gemini-3-flash-preview"


def test_unknown_preset_raises():
    """ValueError for unknown preset name."""
    with pytest.raises(ValueError, match="Unknown preset"):
        apply_preset("nonexistent")


def test_preset_in_nexus_config():
    """NexusConfig with preset='cheap' gets correct models."""
    config = NexusConfig(user={"name": "Test"}, preset="cheap")
    assert config.preset == "cheap"


def test_preset_with_model_override():
    """apply_preset with overrides: explicit field wins over preset."""
    config = apply_preset("quality", overrides={"agent": "deepseek-chat"})
    assert config.agent == "deepseek-chat"
    # Rest stays from quality preset
    assert config.synthesis == "gemini-3.1-pro-preview"
