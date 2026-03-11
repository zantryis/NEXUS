"""Model presets for common configurations."""

from nexus.config.models import ModelsConfig

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
        discovery="gpt-4.1-nano", filtering="gpt-4.1-nano", synthesis="gpt-4.1-mini",
        dialogue_script="gpt-4.1-mini", knowledge_summary="gpt-4.1-nano",
        breaking_news="gpt-4.1-nano", agent="gpt-4.1-mini",
    ),
    # ── OpenAI (balanced) ──
    "openai-balanced": ModelsConfig(
        discovery="gpt-4.1-mini", filtering="gpt-4.1-mini", synthesis="gpt-4.1",
        dialogue_script="gpt-4.1", knowledge_summary="gpt-4.1-mini",
        breaking_news="gpt-4.1-mini", agent="gpt-4.1-mini",
    ),
    # ── OpenAI (quality) ──
    "openai-quality": ModelsConfig(
        discovery="gpt-4.1-mini", filtering="gpt-4.1-mini", synthesis="gpt-4o",
        dialogue_script="gpt-4o", knowledge_summary="gpt-4.1-mini",
        breaking_news="gpt-4.1-mini", agent="gpt-4o",
    ),
    # ── Anthropic ──
    "anthropic": ModelsConfig(
        discovery="claude-haiku-3-5-20241022", filtering="claude-haiku-3-5-20241022",
        synthesis="claude-sonnet-4-20250514", dialogue_script="claude-sonnet-4-20250514",
        knowledge_summary="claude-haiku-3-5-20241022", breaking_news="claude-haiku-3-5-20241022",
        agent="claude-sonnet-4-20250514",
    ),
}


def apply_preset(preset_name: str, overrides: dict | None = None) -> ModelsConfig:
    """Get a ModelsConfig from a preset name, with optional field overrides."""
    if preset_name not in PRESETS:
        raise ValueError(f"Unknown preset: '{preset_name}'. Available: {list(PRESETS.keys())}")

    base = PRESETS[preset_name].model_dump()
    if overrides:
        base.update(overrides)
    return ModelsConfig(**base)
