"""Provider-agnostic LLM client. Supports Gemini, Anthropic, and DeepSeek."""

import logging
import time
from collections import defaultdict
from typing import Optional

from nexus.config.models import ModelsConfig

logger = logging.getLogger(__name__)


def _resolve_provider(model_name: str) -> str:
    """Determine provider from model name prefix."""
    if model_name.startswith("gemini"):
        return "gemini"
    elif model_name.startswith("claude"):
        return "anthropic"
    elif model_name.startswith("deepseek"):
        return "deepseek"
    else:
        raise ValueError(f"Unknown model provider for: {model_name}")


class UsageTracker:
    """Track token usage and timing across LLM calls."""

    def __init__(self):
        self.reset()

    def reset(self):
        self._calls: list[dict] = []

    def record(self, provider: str, model: str, config_key: str,
               input_tokens: int, output_tokens: int, elapsed_s: float):
        self._calls.append({
            "provider": provider,
            "model": model,
            "config_key": config_key,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "elapsed_s": round(elapsed_s, 2),
        })

    def summary(self) -> dict:
        """Aggregate usage by provider and by config_key."""
        by_provider: dict = defaultdict(lambda: {
            "calls": 0, "input_tokens": 0, "output_tokens": 0, "elapsed_s": 0.0,
        })
        by_key: dict = defaultdict(lambda: {
            "calls": 0, "input_tokens": 0, "output_tokens": 0, "elapsed_s": 0.0,
        })

        for c in self._calls:
            for group, key in [(by_provider, c["provider"]), (by_key, c["config_key"])]:
                group[key]["calls"] += 1
                group[key]["input_tokens"] += c["input_tokens"]
                group[key]["output_tokens"] += c["output_tokens"]
                group[key]["elapsed_s"] = round(group[key]["elapsed_s"] + c["elapsed_s"], 2)

        total_in = sum(c["input_tokens"] for c in self._calls)
        total_out = sum(c["output_tokens"] for c in self._calls)
        total_s = sum(c["elapsed_s"] for c in self._calls)

        return {
            "total_calls": len(self._calls),
            "total_input_tokens": total_in,
            "total_output_tokens": total_out,
            "total_elapsed_s": round(total_s, 2),
            "by_provider": dict(by_provider),
            "by_config_key": dict(by_key),
        }


class LLMClient:
    def __init__(
        self,
        models_config: ModelsConfig,
        api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        deepseek_api_key: Optional[str] = None,
    ):
        self._config = models_config
        self._gemini_client = None
        self._anthropic_client = None
        self._deepseek_client = None
        self.usage = UsageTracker()

        # Lazy-init Gemini
        if api_key:
            from google import genai
            self._gemini_client = genai.Client(api_key=api_key)

        # Lazy-init Anthropic
        if anthropic_api_key:
            import anthropic
            self._anthropic_client = anthropic.AsyncAnthropic(api_key=anthropic_api_key)

        # Lazy-init DeepSeek (OpenAI-compatible)
        if deepseek_api_key:
            from openai import AsyncOpenAI
            self._deepseek_client = AsyncOpenAI(
                api_key=deepseek_api_key,
                base_url="https://api.deepseek.com",
            )

    def resolve_model(self, config_key: str) -> str:
        """Look up model name from config key (e.g. 'filtering' -> 'gemini-3-flash-preview')."""
        if not hasattr(self._config, config_key):
            raise ValueError(f"Unknown config key: {config_key}")
        return getattr(self._config, config_key)

    async def complete(
        self,
        config_key: str,
        system_prompt: str,
        user_prompt: str,
        json_response: bool = False,
    ) -> str:
        """Send a completion request through the configured model."""
        model = self.resolve_model(config_key)
        provider = _resolve_provider(model)

        t0 = time.monotonic()

        if provider == "gemini":
            text, in_tok, out_tok = await self._complete_gemini(
                model, system_prompt, user_prompt, json_response)
        elif provider == "anthropic":
            text, in_tok, out_tok = await self._complete_anthropic(
                model, system_prompt, user_prompt, json_response)
        elif provider == "deepseek":
            text, in_tok, out_tok = await self._complete_deepseek(
                model, system_prompt, user_prompt, json_response)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

        elapsed = time.monotonic() - t0
        self.usage.record(provider, model, config_key, in_tok, out_tok, elapsed)
        return text

    async def _complete_gemini(
        self, model: str, system_prompt: str, user_prompt: str, json_response: bool
    ) -> tuple[str, int, int]:
        if not self._gemini_client:
            raise RuntimeError("Gemini client not initialized — set GEMINI_API_KEY")

        from google.genai import types
        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
        )
        if json_response:
            config.response_mime_type = "application/json"

        response = await self._gemini_client.aio.models.generate_content(
            model=model,
            contents=user_prompt,
            config=config,
        )

        in_tok = getattr(getattr(response, 'usage_metadata', None), 'prompt_token_count', 0) or 0
        out_tok = getattr(getattr(response, 'usage_metadata', None), 'candidates_token_count', 0) or 0
        return response.text, in_tok, out_tok

    async def _complete_anthropic(
        self, model: str, system_prompt: str, user_prompt: str, json_response: bool
    ) -> tuple[str, int, int]:
        if not self._anthropic_client:
            raise RuntimeError("Anthropic client not initialized — set ANTHROPIC_API_KEY")

        effective_system = system_prompt
        if json_response:
            effective_system += "\n\nIMPORTANT: Respond with valid JSON only. No markdown, no explanation."

        response = await self._anthropic_client.messages.create(
            model=model,
            max_tokens=4096,
            system=effective_system,
            messages=[{"role": "user", "content": user_prompt}],
        )

        in_tok = getattr(getattr(response, 'usage', None), 'input_tokens', 0) or 0
        out_tok = getattr(getattr(response, 'usage', None), 'output_tokens', 0) or 0
        return response.content[0].text, in_tok, out_tok

    async def _complete_deepseek(
        self, model: str, system_prompt: str, user_prompt: str, json_response: bool
    ) -> tuple[str, int, int]:
        if not self._deepseek_client:
            raise RuntimeError("DeepSeek client not initialized — set DEEPSEEK_API_KEY")

        effective_system = system_prompt
        if json_response:
            effective_system += "\n\nIMPORTANT: Respond with valid JSON only. No markdown, no explanation."

        kwargs = {
            "model": model,
            "messages": [
                {"role": "system", "content": effective_system},
                {"role": "user", "content": user_prompt},
            ],
        }
        if json_response:
            kwargs["response_format"] = {"type": "json_object"}

        response = await self._deepseek_client.chat.completions.create(**kwargs)

        in_tok = getattr(getattr(response, 'usage', None), 'prompt_tokens', 0) or 0
        out_tok = getattr(getattr(response, 'usage', None), 'completion_tokens', 0) or 0
        return response.choices[0].message.content, in_tok, out_tok
