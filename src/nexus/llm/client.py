"""Provider-agnostic LLM client. Supports Gemini, Anthropic, OpenAI, DeepSeek, and Ollama."""

import logging
import time
from collections import defaultdict
from typing import Optional

import httpx

from nexus.config.models import BudgetConfig, ModelsConfig
from nexus.llm.budget import BudgetDegradedError, BudgetExceededError, BudgetGuard
from nexus.llm.cost import estimate_cost

logger = logging.getLogger(__name__)


def _resolve_provider(model_name: str) -> str:
    """Determine provider from model name prefix."""
    if model_name.startswith("ollama/"):
        return "ollama"
    elif model_name.startswith("gemini"):
        return "gemini"
    elif model_name.startswith("claude"):
        return "anthropic"
    elif model_name.startswith("deepseek"):
        return "deepseek"
    elif model_name.startswith(("gpt-", "o1", "o3", "o4")):
        return "openai"
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

    def cost_summary(self) -> dict:
        """Aggregate cost from tracked calls using the cost module."""
        from nexus.llm.cost import cost_summary as _cost_summary
        return _cost_summary(self._calls)


class LLMClient:
    def __init__(
        self,
        models_config: ModelsConfig,
        api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        deepseek_api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        ollama_base_url: Optional[str] = None,
        budget_config: Optional[BudgetConfig] = None,
    ):
        self._config = models_config
        self._gemini_client = None
        self._anthropic_client = None
        self._deepseek_client = None
        self._openai_client = None
        self._ollama_base_url = ollama_base_url or "http://localhost:11434"
        self.usage = UsageTracker()
        self._budget_guard: Optional[BudgetGuard] = (
            BudgetGuard(budget_config) if budget_config else None
        )

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

        # Lazy-init OpenAI (API key or OAuth token)
        if openai_api_key:
            from openai import AsyncOpenAI
            self._openai_client = AsyncOpenAI(api_key=openai_api_key)

    def set_openai_oauth_token(self, token: str) -> None:
        """Hot-swap OpenAI client to use an OAuth access token."""
        from openai import AsyncOpenAI
        self._openai_client = AsyncOpenAI(api_key=token)

    def resolve_model(self, config_key: str) -> str:
        """Look up model name from config key (e.g. 'filtering' -> 'gemini-3-flash-preview')."""
        if not hasattr(self._config, config_key):
            raise ValueError(f"Unknown config key: {config_key}")
        return getattr(self._config, config_key)

    @property
    def budget_status(self) -> str:
        """Return budget status: 'ok', 'warning', or 'over_limit'."""
        if not self._budget_guard:
            return "ok"
        spend = self._budget_guard.today_spend
        limit = self._budget_guard._config.daily_limit_usd
        warning = self._budget_guard._config.warning_threshold_usd
        if spend >= limit:
            return "over_limit"
        if spend >= warning:
            return "warning"
        return "ok"

    @property
    def today_spend(self) -> float:
        """Return current day's spend in USD."""
        if not self._budget_guard:
            return 0.0
        return self._budget_guard.today_spend

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

        # Budget check before making the call
        if self._budget_guard:
            status = self._budget_guard.check_budget(config_key)
            if status == "blocked":
                raise BudgetExceededError(
                    f"Daily budget limit reached (${self._budget_guard.today_spend:.4f}), all calls blocked"
                )
            if status == "degraded":
                raise BudgetDegradedError(
                    f"Daily budget limit reached, expensive operation '{config_key}' blocked"
                )

        t0 = time.monotonic()

        if provider == "gemini":
            text, in_tok, out_tok = await self._complete_gemini(
                model, system_prompt, user_prompt, json_response)
        elif provider == "anthropic":
            text, in_tok, out_tok = await self._complete_anthropic(
                model, system_prompt, user_prompt, json_response)
        elif provider == "openai":
            text, in_tok, out_tok = await self._complete_openai(
                model, system_prompt, user_prompt, json_response)
        elif provider == "deepseek":
            text, in_tok, out_tok = await self._complete_deepseek(
                model, system_prompt, user_prompt, json_response)
        elif provider == "ollama":
            text, in_tok, out_tok = await self._complete_ollama(
                model, system_prompt, user_prompt, json_response)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

        elapsed = time.monotonic() - t0
        self.usage.record(provider, model, config_key, in_tok, out_tok, elapsed)

        # Record cost to budget guard after successful call
        if self._budget_guard:
            cost = estimate_cost(model, in_tok, out_tok)
            self._budget_guard.record_cost(cost)

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

    async def _complete_ollama(
        self, model: str, system_prompt: str, user_prompt: str, json_response: bool
    ) -> tuple[str, int, int]:
        model_name = model.removeprefix("ollama/")
        url = f"{self._ollama_base_url}/api/chat"
        body: dict = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
        }
        if json_response:
            body["format"] = "json"

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(url, json=body, timeout=120.0)
                resp.raise_for_status()
        except httpx.ConnectError:
            raise RuntimeError(
                f"Ollama not running at {self._ollama_base_url}"
            )

        data = resp.json()
        text = data["message"]["content"]
        in_tok = data.get("prompt_eval_count", 0)
        out_tok = data.get("eval_count", 0)
        return text, in_tok, out_tok

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

    async def _complete_openai(
        self, model: str, system_prompt: str, user_prompt: str, json_response: bool
    ) -> tuple[str, int, int]:
        if not self._openai_client:
            raise RuntimeError("OpenAI client not initialized — set OPENAI_API_KEY")

        kwargs: dict = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        if json_response:
            kwargs["response_format"] = {"type": "json_object"}

        response = await self._openai_client.chat.completions.create(**kwargs)

        in_tok = getattr(getattr(response, 'usage', None), 'prompt_tokens', 0) or 0
        out_tok = getattr(getattr(response, 'usage', None), 'completion_tokens', 0) or 0
        return response.choices[0].message.content, in_tok, out_tok
