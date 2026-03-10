"""Tests for config Pydantic models."""

import pytest
from nexus.config.models import (
    UserConfig,
    BriefingConfig,
    TopicConfig,
    ModelsConfig,
    SourcesConfig,
    NexusConfig,
)


def test_minimal_topic():
    topic = TopicConfig(name="AI Research", priority="low")
    assert topic.name == "AI Research"
    assert topic.priority == "low"
    assert topic.subtopics == []
    assert topic.source_languages == ["en"]
    assert topic.perspective_diversity == "low"


def test_full_topic():
    topic = TopicConfig(
        name="Iran-US Relations",
        priority="high",
        subtopics=["sanctions", "nuclear program"],
        source_languages=["en", "fa"],
        perspective_diversity="high",
    )
    assert topic.source_languages == ["en", "fa"]
    assert len(topic.subtopics) == 2


def test_user_config_defaults():
    user = UserConfig(name="Tristan")
    assert user.timezone == "UTC"
    assert user.output_language == "en"


def test_briefing_config_defaults():
    briefing = BriefingConfig()
    assert briefing.schedule == "06:00"
    assert briefing.format == "two-host-dialogue"
    assert briefing.style == "analytical"
    assert briefing.depth == "detailed"


def test_models_config_defaults():
    models = ModelsConfig()
    assert models.filtering == "gemini-3-flash-preview"
    assert models.synthesis == "gemini-3.1-pro-preview"
    assert models.knowledge_summary == "gemini-3-flash-preview"


def test_full_nexus_config():
    config = NexusConfig(
        user=UserConfig(name="Tristan", timezone="America/Denver"),
        topics=[
            TopicConfig(name="AI Research", priority="high"),
        ],
    )
    assert config.user.name == "Tristan"
    assert len(config.topics) == 1
    assert config.briefing.schedule == "06:00"
    assert config.models.filtering == "gemini-3-flash-preview"


def test_sources_config_defaults():
    sources = SourcesConfig()
    assert sources.global_feeds == []
    assert sources.blocked_sources == []
    assert sources.discover_new_sources is True
    assert sources.discovery_interval_days == 7


def test_topic_scope_defaults():
    topic = TopicConfig(name="Test")
    assert topic.scope == "medium"
    assert topic.max_events is None


def test_topic_scope_values():
    narrow = TopicConfig(name="Iran-US", scope="narrow")
    broad = TopicConfig(name="AI/ML", scope="broad")
    assert narrow.scope == "narrow"
    assert broad.scope == "broad"


def test_topic_max_events_override():
    topic = TopicConfig(name="Test", max_events=50)
    assert topic.max_events == 50


def test_audio_config_defaults():
    from nexus.config.models import AudioConfig
    audio = AudioConfig()
    assert audio.enabled is True
    assert audio.tts_backend == "gemini"
    assert audio.tts_model == "gemini-2.5-flash-preview-tts"
    assert audio.voice_host_a == "Kore"
    assert audio.voice_host_b == "Puck"


def test_breaking_news_config_defaults():
    from nexus.config.models import BreakingNewsConfig
    bn = BreakingNewsConfig()
    assert bn.enabled is True
    assert bn.poll_interval_hours == 3
    assert bn.threshold == 7


def test_telegram_config_defaults():
    from nexus.config.models import TelegramConfig
    tg = TelegramConfig()
    assert tg.enabled is True
    assert tg.chat_id is None


def test_nexus_config_with_new_sections():
    config = NexusConfig(
        user=UserConfig(name="Tristan"),
        audio={"tts_backend": "openai", "voice_host_a": "nova"},
        breaking_news={"threshold": 8},
        telegram={"chat_id": 12345},
    )
    assert config.audio.tts_backend == "openai"
    assert config.audio.voice_host_a == "nova"
    assert config.breaking_news.threshold == 8
    assert config.telegram.chat_id == 12345
