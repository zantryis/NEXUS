"""Model presets for common configurations."""

from nexus.config.models import ModelsConfig

# Model choices per provider (for custom model picker UI)
MODEL_CHOICES = {
    "gemini": [
        "gemini-3-flash-preview",
        "gemini-3.1-pro-preview",
    ],
    "openai": [
        "gpt-4.1-nano",
        "gpt-4.1-mini",
        "gpt-4.1",
        "gpt-5-mini",
        "gpt-5.4",
    ],
    "anthropic": [
        "claude-haiku-4-5-20251001",
        "claude-sonnet-4-6",
        "claude-opus-4-6",
    ],
    "deepseek": [
        "deepseek-chat",
        "deepseek-reasoner",
    ],
    "ollama": [
        "ollama/qwen2",
        "ollama/llama3",
        "ollama/mistral",
    ],
    "litellm": [
        "litellm/claude-sonnet-4-6",
        "litellm/claude-opus-4-6",
        "litellm/gpt-5.4",
    ],
}

# Pipeline stages that map to model config fields
PIPELINE_STAGES = [
    ("filtering", "Filtering", "Fast/cheap model for relevance scoring"),
    ("synthesis", "Synthesis", "Smart model for narrative synthesis"),
    ("dialogue_script", "Dialogue Script", "Smart model for podcast script generation"),
    ("knowledge_summary", "Knowledge Summary", "Fast model for entity/thread summaries"),
    ("breaking_news", "Breaking News", "Fast model for breaking news detection"),
    ("agent", "Agent / Q&A", "Smart model for chat Q&A and Telegram"),
    ("discovery", "Source Discovery", "Fast model for RSS feed discovery"),
]

PRESETS = {
    # ── Free (local Ollama) ──
    "free": ModelsConfig(
        discovery="ollama/qwen2", filtering="ollama/qwen2", synthesis="ollama/qwen2",
        dialogue_script="ollama/qwen2", knowledge_summary="ollama/qwen2",
        breaking_news="ollama/qwen2", agent="ollama/qwen2",
    ),
    # ── DeepSeek (very cheap) ──
    "cheap": ModelsConfig(
        discovery="deepseek-chat", filtering="deepseek-chat", synthesis="deepseek-chat",
        dialogue_script="deepseek-chat", knowledge_summary="deepseek-chat",
        breaking_news="deepseek-chat", agent="deepseek-chat",
    ),
    # ── Gemini (balanced) ──
    "balanced": ModelsConfig(
        discovery="gemini-3-flash-preview", filtering="gemini-3-flash-preview",
        synthesis="gemini-3.1-pro-preview", dialogue_script="gemini-3.1-pro-preview",
        knowledge_summary="gemini-3-flash-preview", breaking_news="gemini-3-flash-preview",
        agent="gemini-3-flash-preview",
    ),
    # ── Gemini (quality) ──
    "quality": ModelsConfig(
        discovery="gemini-3-flash-preview", filtering="gemini-3-flash-preview",
        synthesis="gemini-3.1-pro-preview", dialogue_script="gemini-3.1-pro-preview",
        knowledge_summary="gemini-3-flash-preview", breaking_news="gemini-3-flash-preview",
        agent="gemini-3.1-pro-preview",
    ),
    # ── OpenAI (cheap) ──
    "openai-cheap": ModelsConfig(
        discovery="gpt-4.1-nano", filtering="gpt-4.1-nano", synthesis="gpt-5-mini",
        dialogue_script="gpt-5-mini", knowledge_summary="gpt-4.1-nano",
        breaking_news="gpt-4.1-nano", agent="gpt-5-mini",
    ),
    # ── OpenAI (balanced) ──
    "openai-balanced": ModelsConfig(
        discovery="gpt-4.1-mini", filtering="gpt-4.1-mini", synthesis="gpt-5.4",
        dialogue_script="gpt-5.4", knowledge_summary="gpt-4.1-mini",
        breaking_news="gpt-4.1-mini", agent="gpt-5-mini",
    ),
    # ── OpenAI (quality) ──
    "openai-quality": ModelsConfig(
        discovery="gpt-5-mini", filtering="gpt-5-mini", synthesis="gpt-5.4",
        dialogue_script="gpt-5.4", knowledge_summary="gpt-5-mini",
        breaking_news="gpt-5-mini", agent="gpt-5.4",
    ),
    # ── Anthropic ──
    "anthropic": ModelsConfig(
        discovery="claude-haiku-4-5-20251001", filtering="claude-haiku-4-5-20251001",
        synthesis="claude-sonnet-4-6", dialogue_script="claude-sonnet-4-6",
        knowledge_summary="claude-haiku-4-5-20251001", breaking_news="claude-haiku-4-5-20251001",
        agent="claude-sonnet-4-6",
    ),
    # ── Cloud (via LiteLLM proxy) ──
    "cloud-balanced": ModelsConfig(
        discovery="litellm/claude-sonnet-4-6", filtering="litellm/claude-sonnet-4-6",
        synthesis="litellm/gpt-5.4", dialogue_script="litellm/gpt-5.4",
        knowledge_summary="litellm/claude-sonnet-4-6", breaking_news="litellm/claude-sonnet-4-6",
        agent="litellm/gpt-5.4",
    ),
    "cloud-quality": ModelsConfig(
        discovery="litellm/claude-sonnet-4-6", filtering="litellm/claude-sonnet-4-6",
        synthesis="litellm/claude-opus-4-6", dialogue_script="litellm/claude-opus-4-6",
        knowledge_summary="litellm/claude-sonnet-4-6", breaking_news="litellm/claude-sonnet-4-6",
        agent="litellm/claude-opus-4-6",
    ),
}

# Info dict for UI display (preset picker cards)
PRESET_INFO = {
    "free": {"label": "Free (Ollama local)", "cost": "$0/day", "provider": "ollama"},
    "cheap": {"label": "Cheap (DeepSeek)", "cost": "~$0.01/day", "provider": "deepseek"},
    "balanced": {"label": "Balanced (Gemini)", "cost": "~$0.05/day", "provider": "gemini"},
    "quality": {"label": "Quality (Gemini Pro)", "cost": "~$0.15/day", "provider": "gemini"},
    "openai-cheap": {"label": "OpenAI Cheap", "cost": "~$0.03/day", "provider": "openai"},
    "openai-balanced": {"label": "OpenAI Balanced", "cost": "~$0.10/day", "provider": "openai"},
    "openai-quality": {"label": "OpenAI Quality", "cost": "~$0.25/day", "provider": "openai"},
    "anthropic": {"label": "Anthropic (Claude)", "cost": "~$0.10/day", "provider": "anthropic"},
    "cloud-balanced": {"label": "Cloud Balanced (LiteLLM)", "cost": "~$0.50/day", "provider": "litellm"},
    "cloud-quality": {"label": "Cloud Quality (LiteLLM)", "cost": "~$2.00/day", "provider": "litellm"},
}


def preset_names() -> set[str]:
    """Return the named presets supported by the config loader."""
    return set(PRESETS.keys()) | {"custom"}


def apply_preset(preset_name: str, overrides: dict | None = None) -> ModelsConfig:
    """Get a ModelsConfig from a preset name, with optional field overrides."""
    if preset_name not in PRESETS:
        raise ValueError(f"Unknown preset: '{preset_name}'. Available: {list(PRESETS.keys())}")

    base = PRESETS[preset_name].model_dump()
    if overrides:
        base.update(overrides)
    return ModelsConfig(**base)


def all_model_choices() -> list[str]:
    """Flat list of all known model IDs (for validation/autocomplete)."""
    seen = set()
    result = []
    for models in MODEL_CHOICES.values():
        for m in models:
            if m not in seen:
                seen.add(m)
                result.append(m)
    return result
