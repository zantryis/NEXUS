"""Pydantic models for nexus configuration."""

from typing import Literal, Optional

from pydantic import BaseModel, Field


class UserConfig(BaseModel):
    name: str
    timezone: str = "UTC"
    output_language: str = "en"


class BriefingConfig(BaseModel):
    schedule: str = "06:00"
    duration_target_minutes: int = 30
    format: str = "two-host-dialogue"
    style: str = "analytical"
    depth: str = "detailed"
    additional_languages: list[str] = Field(default_factory=list)


class TopicConfig(BaseModel):
    name: str
    priority: Literal["low", "medium", "high"] = "medium"
    subtopics: list[str] = Field(default_factory=list)
    source_languages: list[str] = Field(default_factory=lambda: ["en"])
    perspective_diversity: Literal["low", "medium", "high"] = "high"
    filter_threshold: float = Field(default=5.0, ge=0.0, le=10.0)
    scope: Literal["narrow", "medium", "broad"] = "medium"
    max_events: Optional[int] = None  # override auto-calculated event cap


class ModelsConfig(BaseModel):
    discovery: str = "gemini-3-flash-preview"
    filtering: str = "gemini-3-flash-preview"
    synthesis: str = "gemini-3.1-pro-preview"
    dialogue_script: str = "gemini-3.1-pro-preview"
    knowledge_summary: str = "gemini-3-flash-preview"
    breaking_news: str = "gemini-3-flash-preview"
    agent: str = "gemini-3.1-pro-preview"


class AudioConfig(BaseModel):
    enabled: bool = True
    tts_backend: str = "gemini"  # "gemini" | "openai" | "elevenlabs"
    tts_model: str = "gemini-2.5-flash-preview-tts"
    voice_host_a: str = "Kore"
    voice_host_b: str = "Puck"
    # ElevenLabs voice tuning (only used when tts_backend == "elevenlabs")
    elevenlabs_stability: float = Field(default=0.7, ge=0.0, le=1.0)
    elevenlabs_similarity_boost: float = Field(default=0.8, ge=0.0, le=1.0)
    elevenlabs_style: float = Field(default=0.35, ge=0.0, le=1.0)
    elevenlabs_speaker_boost: bool = True


class BreakingNewsConfig(BaseModel):
    enabled: bool = True
    poll_interval_hours: int = Field(default=3, ge=1)
    threshold: int = Field(default=7, ge=1, le=10)
    wire_feeds: list[dict] = Field(default_factory=list)
    default_feeds: bool = True


class TelegramConfig(BaseModel):
    enabled: bool = True
    chat_id: Optional[int] = None


class SourcesConfig(BaseModel):
    global_feeds: list[str] = Field(default_factory=list)
    blocked_sources: list[str] = Field(default_factory=list)
    discover_new_sources: bool = True
    discovery_interval_days: int = 7


class BudgetConfig(BaseModel):
    daily_limit_usd: float = Field(default=1.00, ge=0.0)
    warning_threshold_usd: float = Field(default=0.50, ge=0.0)
    degradation_strategy: Literal["skip_expensive", "stop_all"] = "skip_expensive"


class NexusConfig(BaseModel):
    user: UserConfig
    briefing: BriefingConfig = Field(default_factory=BriefingConfig)
    topics: list[TopicConfig] = Field(default_factory=list)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    breaking_news: BreakingNewsConfig = Field(default_factory=BreakingNewsConfig)
    telegram: TelegramConfig = Field(default_factory=TelegramConfig)
    sources: SourcesConfig = Field(default_factory=SourcesConfig)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    preset: Optional[str] = None
    demo_mode: bool = False
