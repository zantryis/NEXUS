"""Tests for config Pydantic models."""

import pytest
from pydantic import ValidationError

from nexus.config.models import (
    UserConfig,
    BriefingConfig,
    FutureProjectionConfig,
    TopicConfig,
    ModelsConfig,
    SourcesConfig,
    NexusConfig,
    BudgetConfig,
    BreakingNewsConfig,
)


def test_minimal_topic():
    topic = TopicConfig(name="AI Research", priority="low")
    assert topic.name == "AI Research"
    assert topic.priority == "low"
    assert topic.subtopics == []
    assert topic.source_languages == ["en"]
    assert topic.perspective_diversity == "high"


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


def test_audio_elevenlabs_defaults():
    from nexus.config.models import AudioConfig
    audio = AudioConfig()
    assert audio.elevenlabs_stability == 0.7
    assert audio.elevenlabs_similarity_boost == 0.8
    assert audio.elevenlabs_style == 0.35
    assert audio.elevenlabs_speaker_boost is True


def test_audio_elevenlabs_custom():
    from nexus.config.models import AudioConfig
    audio = AudioConfig(
        elevenlabs_stability=0.9,
        elevenlabs_similarity_boost=0.6,
        elevenlabs_style=0.5,
        elevenlabs_speaker_boost=False,
    )
    assert audio.elevenlabs_stability == 0.9
    assert audio.elevenlabs_speaker_boost is False


def test_audio_elevenlabs_stability_too_high():
    from nexus.config.models import AudioConfig
    with pytest.raises(ValidationError):
        AudioConfig(elevenlabs_stability=1.5)


def test_audio_elevenlabs_similarity_negative():
    from nexus.config.models import AudioConfig
    with pytest.raises(ValidationError):
        AudioConfig(elevenlabs_similarity_boost=-0.1)


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
        future_projection={"enabled": True, "engine": "native"},
    )
    assert config.audio.tts_backend == "openai"
    assert config.audio.voice_host_a == "nova"
    assert config.breaking_news.threshold == 8
    assert config.telegram.chat_id == 12345
    assert config.future_projection.enabled is True


def test_briefing_additional_languages_default():
    briefing = BriefingConfig()
    assert briefing.additional_languages == []


def test_briefing_additional_languages():
    briefing = BriefingConfig(additional_languages=["zh", "es"])
    assert briefing.additional_languages == ["zh", "es"]


def test_nexus_config_additional_languages():
    config = NexusConfig(
        user=UserConfig(name="Tristan"),
        briefing=BriefingConfig(additional_languages=["zh"]),
    )
    assert config.briefing.additional_languages == ["zh"]


# --- Validation constraint tests ---


def test_topic_filter_threshold_too_high():
    with pytest.raises(ValidationError):
        TopicConfig(name="Test", filter_threshold=11.0)


def test_topic_filter_threshold_negative():
    with pytest.raises(ValidationError):
        TopicConfig(name="Test", filter_threshold=-1.0)


def test_topic_invalid_scope():
    with pytest.raises(ValidationError):
        TopicConfig(name="Test", scope="wide")


def test_topic_invalid_priority():
    with pytest.raises(ValidationError):
        TopicConfig(name="Test", priority="critical")


def test_topic_invalid_perspective_diversity():
    with pytest.raises(ValidationError):
        TopicConfig(name="Test", perspective_diversity="extreme")


def test_budget_negative_limit():
    with pytest.raises(ValidationError):
        BudgetConfig(daily_limit_usd=-5.0)


def test_future_projection_defaults():
    cfg = FutureProjectionConfig()
    assert cfg.enabled is False
    assert cfg.engine == "actor"
    assert cfg.horizons == [3, 7, 14]
    assert cfg.max_items_per_topic == 3
    assert cfg.graph_sidecars.adapters == ["kuzu", "graphiti"]
    assert cfg.kalshi.ledger_path == "data/benchmarks/kalshi.sqlite"
    # Daily prediction scheduling
    assert cfg.prediction_schedule_offset_minutes == 30
    assert cfg.daily_engine == "structural"
    assert cfg.kg_native_enabled is True
    assert cfg.max_kg_questions_per_topic == 5


def test_budget_negative_warning():
    with pytest.raises(ValidationError):
        BudgetConfig(warning_threshold_usd=-1.0)


def test_budget_invalid_strategy():
    with pytest.raises(ValidationError):
        BudgetConfig(degradation_strategy="panic")


def test_breaking_news_poll_zero():
    with pytest.raises(ValidationError):
        BreakingNewsConfig(poll_interval_hours=0)


def test_breaking_news_threshold_zero():
    with pytest.raises(ValidationError):
        BreakingNewsConfig(threshold=0)


def test_breaking_news_threshold_too_high():
    with pytest.raises(ValidationError):
        BreakingNewsConfig(threshold=11)


def test_valid_boundary_values():
    """Valid boundary values should not raise."""
    TopicConfig(name="Test", filter_threshold=0.0, scope="narrow", priority="high")
    TopicConfig(name="Test", filter_threshold=10.0, scope="broad", priority="low")
    BudgetConfig(daily_limit_usd=0.0)
    BreakingNewsConfig(poll_interval_hours=1, threshold=1)
    BreakingNewsConfig(threshold=10)
