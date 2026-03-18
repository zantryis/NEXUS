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
    filter_threshold: float = Field(default=4.0, ge=0.0, le=10.0)
    scope: Literal["narrow", "medium", "broad"] = "medium"
    pairwise_filtering: bool = False  # opt-in hybrid funnel with pairwise comparison
    max_events: Optional[int] = None  # override auto-calculated event cap
    projection_eligible: bool = True  # set False to skip future-projection pipeline


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
    podcast_style: str = "conversational"  # conversational | analytical | energetic | formal
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


class GraphSidecarConfig(BaseModel):
    enabled: bool = False
    export_schema_version: int = Field(default=1, ge=1)
    adapters: list[Literal["kuzu", "graphiti"]] = Field(default_factory=lambda: ["kuzu", "graphiti"])
    export_dir: str = "data/benchmarks/graph"
    max_evidence_ids: int = Field(default=8, ge=1)


class KalshiBenchmarkConfig(BaseModel):
    enabled: bool = False
    base_url: str = "https://api.elections.kalshi.com/trade-api/v2"
    api_key_id_env: str = "KALSHI_API_KEY_ID"
    private_key_path_env: str = "KALSHI_PRIVATE_KEY_PATH"
    private_key_pem_env: str = "KALSHI_PRIVATE_KEY_PEM"
    ledger_path: str = "data/benchmarks/kalshi.sqlite"
    mapping_file: str = "data/benchmarks/kalshi_mappings.json"
    comparison_tolerance_minutes: int = Field(default=30, ge=1)
    default_status: str = "open"
    auto_scan: bool = False
    auto_match_min_score: int = Field(default=2, ge=1)
    max_markets_per_topic: int = Field(default=5, ge=1, le=20)
    max_horizon_days: int = Field(default=90, ge=1)
    min_horizon_days: int = Field(default=1, ge=0)


class FutureProjectionConfig(BaseModel):
    enabled: bool = False
    # Production: actor (default, multi-actor swarm), structural (KG-native daily).
    # Experimental: graphrag, perspective, debate, naked (benchmark/research only).
    engine: Literal["actor", "native", "graphrag", "perspective", "debate", "naked", "structural"] = "actor"
    min_history_days: int = Field(default=7, ge=1)
    min_thread_snapshots: int = Field(default=2, ge=1)
    horizons: list[int] = Field(default_factory=lambda: [3, 7, 14])  # reserved for future use
    max_items_per_topic: int = Field(default=3, ge=1, le=5)
    critic_pass: bool = True
    # Daily prediction scheduling
    prediction_schedule_offset_minutes: int = Field(default=30, ge=0)
    daily_engine: str = "structural"
    kg_native_enabled: bool = True
    max_kg_questions_per_topic: int = Field(default=5, ge=1, le=10)
    graph_sidecars: GraphSidecarConfig = Field(default_factory=GraphSidecarConfig)
    kalshi: KalshiBenchmarkConfig = Field(default_factory=KalshiBenchmarkConfig)


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
    future_projection: FutureProjectionConfig = Field(default_factory=FutureProjectionConfig)
    preset: Optional[str] = None
    demo_mode: bool = False
