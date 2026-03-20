"""Tests for model presets."""

import pytest
from nexus.config.models import ModelsConfig, NexusConfig
from nexus.config.presets import PRESETS, apply_preset, MODEL_CHOICES, PIPELINE_STAGES, all_model_choices


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


def test_model_choices_has_all_providers():
    """MODEL_CHOICES covers all supported providers."""
    assert "gemini" in MODEL_CHOICES
    assert "openai" in MODEL_CHOICES
    assert "anthropic" in MODEL_CHOICES
    assert "deepseek" in MODEL_CHOICES
    assert "ollama" in MODEL_CHOICES
    assert "litellm" in MODEL_CHOICES


def test_model_choices_has_latest_models():
    """Latest models are included in choices."""
    assert "gpt-5.4" in MODEL_CHOICES["openai"]
    assert "gpt-5.4-mini" in MODEL_CHOICES["openai"]
    assert "claude-sonnet-4-6" in MODEL_CHOICES["anthropic"]
    assert "claude-opus-4-6" in MODEL_CHOICES["anthropic"]
    assert "litellm/gpt" in MODEL_CHOICES["litellm"]
    assert "litellm/opus" in MODEL_CHOICES["litellm"]


def test_pipeline_stages_cover_all_config_fields():
    """PIPELINE_STAGES should match ModelsConfig fields."""
    stage_keys = {s[0] for s in PIPELINE_STAGES}
    config_fields = set(ModelsConfig.model_fields.keys())
    assert stage_keys == config_fields


def test_all_model_choices_returns_flat_list():
    """all_model_choices() returns a deduplicated flat list."""
    choices = all_model_choices()
    assert len(choices) > 10
    assert len(choices) == len(set(choices))  # no dupes
    assert "gpt-5.4" in choices
    assert "claude-sonnet-4-6" in choices
