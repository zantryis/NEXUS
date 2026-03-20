"""Cost tracking for LLM API calls."""

from collections import defaultdict

# Per-million-token pricing (USD)
PRICING = {
    # Gemini
    "gemini-3-flash-preview": {"input": 0.10, "output": 0.40},
    "gemini-3.1-pro-preview": {"input": 1.25, "output": 5.00},
    "gemini-2.5-flash-preview-tts": {"input": 0.15, "output": 0.60},
    # DeepSeek
    "deepseek-chat": {"input": 0.14, "output": 0.28},
    "deepseek-reasoner": {"input": 0.55, "output": 2.19},
    # OpenAI
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-5.4-mini": {"input": 0.40, "output": 1.60},
    "gpt-5.4": {"input": 2.00, "output": 8.00},
    "o3": {"input": 2.00, "output": 8.00},
    "o3-mini": {"input": 1.10, "output": 4.40},
    "o4-mini": {"input": 1.10, "output": 4.40},
    # Anthropic
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
    "claude-opus-4-6": {"input": 15.00, "output": 75.00},
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate cost in USD for a single call. Returns 0.0 for unknown/free models."""
    if model.startswith("ollama/"):
        return 0.0
    if model.startswith("litellm/"):
        model = model.removeprefix("litellm/")
    pricing = PRICING.get(model)
    if not pricing:
        return 0.0
    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]
    return input_cost + output_cost


def cost_summary(tracker_calls: list[dict]) -> dict:
    """Aggregate cost from UsageTracker._calls list.

    Returns: {"total_usd": float, "by_provider": {str: float},
              "by_config_key": {str: float}, "by_model": {str: float}}
    """
    by_provider: dict[str, float] = defaultdict(float)
    by_config_key: dict[str, float] = defaultdict(float)
    by_model: dict[str, float] = defaultdict(float)
    total_usd = 0.0

    for call in tracker_calls:
        cost = estimate_cost(
            call["model"],
            call["input_tokens"],
            call["output_tokens"],
        )
        total_usd += cost
        by_provider[call["provider"]] += cost
        by_config_key[call["config_key"]] += cost
        by_model[call["model"]] += cost

    return {
        "total_usd": total_usd,
        "by_provider": dict(by_provider),
        "by_config_key": dict(by_config_key),
        "by_model": dict(by_model),
    }
