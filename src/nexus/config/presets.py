"""Model presets for common configurations."""

from nexus.config.models import ModelsConfig

PRESETS = {
    "free": ModelsConfig(
        discovery="ollama/qwen2", filtering="ollama/qwen2", synthesis="ollama/qwen2",
        dialogue_script="ollama/qwen2", knowledge_summary="ollama/qwen2",
        breaking_news="ollama/qwen2", agent="ollama/qwen2",
    ),
    "cheap": ModelsConfig(
        discovery="deepseek-chat", filtering="deepseek-chat", synthesis="deepseek-chat",
        dialogue_script="deepseek-chat", knowledge_summary="deepseek-chat",
        breaking_news="deepseek-chat", agent="deepseek-chat",
    ),
    "balanced": ModelsConfig(
        discovery="gemini-3-flash-preview", filtering="gemini-3-flash-preview",
        synthesis="gemini-3.1-pro-preview", dialogue_script="gemini-3.1-pro-preview",
        knowledge_summary="gemini-3-flash-preview", breaking_news="gemini-3-flash-preview",
        agent="gemini-3-flash-preview",
    ),
    "quality": ModelsConfig(
        discovery="gemini-3-flash-preview", filtering="gemini-3-flash-preview",
        synthesis="gemini-3.1-pro-preview", dialogue_script="gemini-3.1-pro-preview",
        knowledge_summary="gemini-3-flash-preview", breaking_news="gemini-3-flash-preview",
        agent="gemini-3.1-pro-preview",
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
