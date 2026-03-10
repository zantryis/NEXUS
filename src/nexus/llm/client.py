"""Provider-agnostic LLM client. Currently Gemini only."""

from google import genai
from google.genai import types
from nexus.config.models import ModelsConfig


class LLMClient:
    def __init__(self, models_config: ModelsConfig, api_key: str):
        self._config = models_config
        self._client = genai.Client(api_key=api_key)

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
        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
        )
        if json_response:
            config.response_mime_type = "application/json"

        response = await self._client.aio.models.generate_content(
            model=model,
            contents=user_prompt,
            config=config,
        )
        return response.text
