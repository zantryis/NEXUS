"""Pydantic models for nexus configuration."""

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


class TopicConfig(BaseModel):
    name: str
    priority: str = "medium"
    subtopics: list[str] = Field(default_factory=list)
    source_languages: list[str] = Field(default_factory=lambda: ["en"])
    perspective_diversity: str = "low"


class ModelsConfig(BaseModel):
    discovery: str = "gemini-3-flash-preview"
    filtering: str = "gemini-3-flash-preview"
    synthesis: str = "gemini-3.1-pro-preview"
    dialogue_script: str = "gemini-3.1-pro-preview"
    knowledge_summary: str = "gemini-3-flash-preview"
    breaking_news: str = "gemini-3-flash-preview"
    agent: str = "gemini-3.1-pro-preview"


class SourcesConfig(BaseModel):
    global_feeds: list[str] = Field(default_factory=list)
    blocked_sources: list[str] = Field(default_factory=list)
    discover_new_sources: bool = True
    discovery_interval_days: int = 7


class NexusConfig(BaseModel):
    user: UserConfig
    briefing: BriefingConfig = Field(default_factory=BriefingConfig)
    topics: list[TopicConfig] = Field(default_factory=list)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    sources: SourcesConfig = Field(default_factory=SourcesConfig)
