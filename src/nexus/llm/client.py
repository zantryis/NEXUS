"""Provider-agnostic LLM client. Supports Gemini, Anthropic, and DeepSeek."""

import logging
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

        if provider == "gemini":
            return await self._complete_gemini(model, system_prompt, user_prompt, json_response)
        elif provider == "anthropic":
            return await self._complete_anthropic(model, system_prompt, user_prompt, json_response)
        elif provider == "deepseek":
            return await self._complete_deepseek(model, system_prompt, user_prompt, json_response)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    async def _complete_gemini(
        self, model: str, system_prompt: str, user_prompt: str, json_response: bool
    ) -> str:
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
        return response.text

    async def _complete_anthropic(
        self, model: str, system_prompt: str, user_prompt: str, json_response: bool
    ) -> str:
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
        return response.content[0].text

    async def _complete_deepseek(
        self, model: str, system_prompt: str, user_prompt: str, json_response: bool
    ) -> str:
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
        return response.choices[0].message.content
