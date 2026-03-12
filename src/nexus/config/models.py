"""Pydantic models for nexus configuration."""

from typing import Optional

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
    priority: str = "medium"
    subtopics: list[str] = Field(default_factory=list)
    source_languages: list[str] = Field(default_factory=lambda: ["en"])
    perspective_diversity: str = "low"
    filter_threshold: float = 6.0
    scope: str = "medium"  # "narrow" | "medium" | "broad"
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


class BreakingNewsConfig(BaseModel):
    enabled: bool = True
    poll_interval_hours: int = 3
    threshold: int = 7
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
    daily_limit_usd: float = 1.00
    warning_threshold_usd: float = 0.50
    degradation_strategy: str = "skip_expensive"  # "skip_expensive" | "stop_all"


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
