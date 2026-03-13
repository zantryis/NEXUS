"""Tests for multi-config smoke test orchestrator."""

import sys
from pathlib import Path

import pytest

# Ensure scripts/ is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from multi_smoke import VARIANTS, build_variant_config
from nexus.config.models import NexusConfig


# Pre-registered topic slugs that must NOT appear in smoke variants
PRE_REGISTERED = {"ai-ml-research", "iran-us-relations", "formula-1", "global-energy-transition"}


def test_variant_count():
    """Should have exactly 4 test variants."""
    assert len(VARIANTS) == 4


def test_all_variants_have_unique_ports():
    ports = [v["port"] for v in VARIANTS]
    assert len(set(ports)) == len(ports)


def test_all_variants_have_unique_ids():
    ids = [v["id"] for v in VARIANTS]
    assert len(set(ids)) == len(ids)


def test_variant_configs_are_valid_nexus_configs():
    """Each variant config dict should be valid NexusConfig."""
    for v in VARIANTS:
        config_dict = build_variant_config(v)
        config = NexusConfig(**config_dict)
        assert len(config.topics) == 1
        assert config.topics[0].name == v["topic"]
        assert config.audio.tts_backend == v["tts_backend"]


def test_variant_topics_are_novel():
    """Smoke test topics must require auto-discovery (not pre-registered)."""
    for v in VARIANTS:
        slug = v["topic"].lower().replace(" ", "-").replace("/", "-")
        assert slug not in PRE_REGISTERED, f"Variant {v['id']} uses pre-registered topic: {slug}"


def test_variant_d_has_telegram_enabled():
    """Only variant D should have Telegram enabled."""
    for v in VARIANTS:
        if v["id"] == "d":
            assert v["telegram"] is True
        else:
            assert v["telegram"] is False


def test_build_variant_config_telegram_chat_id():
    """build_variant_config should include chat_id when provided for Telegram variant."""
    d_variant = next(v for v in VARIANTS if v["id"] == "d")
    config_dict = build_variant_config(d_variant, chat_id=123456)
    assert config_dict["telegram"]["chat_id"] == 123456
    assert config_dict["telegram"]["enabled"] is True


def test_build_variant_config_no_chat_id_for_non_telegram():
    """Non-Telegram variants should not have chat_id even if provided."""
    a_variant = next(v for v in VARIANTS if v["id"] == "a")
    config_dict = build_variant_config(a_variant, chat_id=123456)
    assert "chat_id" not in config_dict["telegram"]
    assert config_dict["telegram"]["enabled"] is False


def test_elevenlabs_variants_have_voice_settings():
    """ElevenLabs variants should include voice tuning params."""
    for v in VARIANTS:
        config_dict = build_variant_config(v)
        if v["tts_backend"] == "elevenlabs":
            assert "elevenlabs_stability" in config_dict["audio"]
            assert "elevenlabs_similarity_boost" in config_dict["audio"]
        else:
            assert "elevenlabs_stability" not in config_dict["audio"]


def test_all_presets_are_usable():
    """All presets in the matrix must be valid (no openai/anthropic without keys)."""
    usable = {"balanced", "quality", "cheap", "free"}
    for v in VARIANTS:
        assert v["preset"] in usable, f"Variant {v['id']} uses unavailable preset: {v['preset']}"


def test_breaking_news_disabled():
    """All variants should have breaking_news disabled for speed."""
    for v in VARIANTS:
        config_dict = build_variant_config(v)
        assert config_dict["breaking_news"]["enabled"] is False
