"""Provider-agnostic LLM client.

Supports Gemini, Anthropic, OpenAI, DeepSeek, Ollama, and LiteLLM.
"""

import asyncio
import logging
import os
import time
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

import httpx
from dotenv import dotenv_values

# ── Retry / Circuit Breaker constants ──
MAX_RETRIES = 3
RETRY_BASE_DELAY = 1.0  # seconds; doubles each attempt (1, 2, 4)
CIRCUIT_FAILURE_THRESHOLD = 5
CIRCUIT_RECOVERY_TIMEOUT = 60.0  # seconds

# Default timeouts per provider (seconds)
PROVIDER_TIMEOUTS: dict[str, float] = {
    "gemini": 60.0,
    "anthropic": 120.0,
    "openai": 60.0,
    "deepseek": 60.0,
    "litellm": 60.0,
    "ollama": 120.0,
}

from nexus.config.models import BudgetConfig, ModelsConfig
from nexus.llm.budget import BudgetDegradedError, BudgetExceededError, BudgetGuard
from nexus.llm.cost import estimate_cost

logger = logging.getLogger(__name__)

LITELLM_MODEL_ENV_MAP = {
    "gpt": "LITELLM_MODEL_GPT",
    "opus": "LITELLM_MODEL_OPUS",
    "sonnet": "LITELLM_MODEL_SONNET",
    "gemini": "LITELLM_MODEL_GEMINI",
}

# Best-effort normalization for usage/cost tracking when hosted env aliases are used.
LITELLM_CANONICAL_MODEL_MAP = {
    "gpt": "gpt-5.4",
    "opus": "claude-opus-4-6",
    "sonnet": "claude-sonnet-4-6",
    "gemini": "gemini-3.1-pro-preview",
    "opus-4.6": "claude-opus-4-6",
    "sonnet-4.6": "claude-sonnet-4-6",
}


def _env_first(*names: str) -> str:
    """Return the first non-empty env var value from the provided names."""
    for name in names:
        value = os.getenv(name, "").strip()
        if value:
            return value
    return ""


def _parse_explicit_expiry(value: str) -> datetime | None:
    """Parse an ISO8601-ish expiry string from env."""
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _litellm_env_file_path() -> Path | None:
    """Optional env file used by hosted runtimes to refresh rotated creds."""
    raw = os.getenv("NEXUS_ENV_FILE", "").strip()
    return Path(raw) if raw else None


def _is_auth_error(exc: Exception) -> bool:
    """Best-effort detection for provider auth failures."""
    status_code = getattr(exc, "status_code", None)
    response = getattr(exc, "response", None)
    if status_code is None and response is not None:
        status_code = getattr(response, "status_code", None)
    if status_code == 401:
        return True
    message = str(exc).lower()
    return "invalid vm proxy token" in message or "unauthorized" in message



def _is_retryable(exc: Exception) -> bool:
    """Check if an error is transient and worth retrying."""
    if isinstance(exc, (httpx.ConnectError, httpx.TimeoutException, ConnectionError, TimeoutError)):
        return True
    status = getattr(exc, "status_code", None)
    if status is None:
        response = getattr(exc, "response", None)
        if response is not None:
            status = getattr(response, "status_code", None)
    if status is not None:
        return status in (429, 500, 502, 503, 504)
    msg = str(exc).lower()
    return any(p in msg for p in ("rate limit", "too many requests", "overloaded", "service unavailable"))


class CircuitOpenError(Exception):
    """Raised when a provider's circuit breaker is open."""


class CircuitBreaker:
    """Per-provider circuit breaker.

    States: closed (normal) → open (blocking) → half_open (one test call).
    """

    def __init__(
        self,
        failure_threshold: int = CIRCUIT_FAILURE_THRESHOLD,
        recovery_timeout: float = CIRCUIT_RECOVERY_TIMEOUT,
    ):
        self._threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        # Per-provider state
        self._failures: dict[str, int] = defaultdict(int)
        self._opened_at: dict[str, float] = {}
        self._half_open: set[str] = set()

    def check(self, provider: str) -> None:
        """Raise CircuitOpenError if the provider circuit is open."""
        if provider not in self._opened_at:
            return  # closed
        elapsed = time.monotonic() - self._opened_at[provider]
        if elapsed >= self._recovery_timeout:
            # Transition to half-open: allow one test call
            self._half_open.add(provider)
            del self._opened_at[provider]
            return
        raise CircuitOpenError(f"Circuit open for provider '{provider}'")

    def record_success(self, provider: str) -> None:
        self._failures[provider] = 0
        self._half_open.discard(provider)
        self._opened_at.pop(provider, None)

    def record_failure(self, provider: str) -> None:
        self._failures[provider] += 1
        if provider in self._half_open:
            # Half-open test call failed → re-open
            self._half_open.discard(provider)
            self._opened_at[provider] = time.monotonic()
            return
        if self._failures[provider] >= self._threshold:
            self._opened_at[provider] = time.monotonic()


def _resolve_provider(model_name: str) -> str:
    """Determine provider from model name prefix."""
    if model_name.startswith("litellm/"):
        return "litellm"
    if model_name.startswith("ollama/"):
        return "ollama"
    if model_name.startswith("gemini"):
        return "gemini"
    if model_name.startswith(("claude", "opus-", "sonnet-", "haiku-")):
        return "anthropic"
    if model_name.startswith("deepseek"):
        return "deepseek"
    if model_name.startswith(("gpt-", "o1", "o3", "o4")):
        return "openai"
    raise ValueError(f"Unknown model provider for: {model_name}")


class UsageTracker:
    """Track token usage and timing across LLM calls."""

    def __init__(self):
        self.reset()

    def reset(self):
        self._calls: list[dict] = []

    def record(
        self,
        provider: str,
        model: str,
        config_key: str,
        input_tokens: int,
        output_tokens: int,
        elapsed_s: float,
    ):
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
        litellm_base_url: Optional[str] = None,
        litellm_api_key: Optional[str] = None,
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
        self._store = None  # KnowledgeStore, set via set_store()
        self._pending_usage_tasks: set[asyncio.Task] = set()
        self._circuit_breaker = CircuitBreaker()

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

        # LiteLLM proxy (OpenAI-compatible). The hosted environment may rotate
        # credentials, so we keep a signature and refresh on demand in
        # _get_litellm_client().
        self._litellm_client = None
        self._litellm_signature: tuple[str, str] | None = None
        if litellm_api_key and litellm_base_url:
            from openai import AsyncOpenAI
            base_url = litellm_base_url.rstrip("/")
            self._litellm_client = AsyncOpenAI(
                api_key=litellm_api_key,
                base_url=base_url,
            )
            self._litellm_signature = (base_url, litellm_api_key)

    async def set_store(self, store) -> None:
        """Attach a KnowledgeStore for persistent usage logging.

        Also seeds the BudgetGuard with today's existing spend from the store
        so that budget enforcement survives process restarts.
        """
        await self.flush_usage()  # Drain tasks targeting previous store
        self._store = store
        if self._budget_guard:
            await self._budget_guard.sync_from_store(store)

    async def flush_usage(self) -> None:
        """Wait for pending usage record writes. Call before store.close()."""
        if self._pending_usage_tasks:
            await asyncio.gather(*self._pending_usage_tasks, return_exceptions=True)
            self._pending_usage_tasks.clear()

    def set_openai_oauth_token(self, token: str) -> None:
        """Hot-swap OpenAI client to use an OAuth access token."""
        from openai import AsyncOpenAI
        self._openai_client = AsyncOpenAI(api_key=token)

    def resolve_model(self, config_key: str) -> str:
        """Look up configured model string for a pipeline stage."""
        if not hasattr(self._config, config_key):
            raise ValueError(f"Unknown config key: {config_key}")
        return getattr(self._config, config_key)

    def _resolve_runtime_model(self, configured_model: str) -> str:
        """Resolve hosted aliases and env references at request time."""
        if configured_model.startswith("litellm/"):
            alias = configured_model.split("/", 1)[1].strip()
            if alias in LITELLM_MODEL_ENV_MAP:
                env_key = LITELLM_MODEL_ENV_MAP[alias]
                runtime_model = _env_first(env_key)
                if not runtime_model:
                    raise RuntimeError(
                        f"LiteLLM model alias '{configured_model}' is not configured — set {env_key}"
                    )
                return runtime_model
            return alias

        if configured_model.startswith("env:"):
            env_key = configured_model[4:].strip()
            runtime_model = _env_first(env_key)
            if not runtime_model:
                raise RuntimeError(
                    f"Model env reference '{configured_model}' is not configured — set {env_key}"
                )
            return runtime_model

        return configured_model

    def _usage_model(self, configured_model: str, runtime_model: str) -> str:
        """Normalize model names for usage/cost tracking where possible."""
        if configured_model.startswith("litellm/"):
            alias = configured_model.split("/", 1)[1].strip()
            if alias in LITELLM_MODEL_ENV_MAP:
                return LITELLM_CANONICAL_MODEL_MAP.get(alias, runtime_model)
            return LITELLM_CANONICAL_MODEL_MAP.get(runtime_model, runtime_model)
        return LITELLM_CANONICAL_MODEL_MAP.get(runtime_model, runtime_model)

    def _litellm_token_expiry(self) -> datetime | None:
        """Return proxy token expiry when the hosted environment exposes it."""
        explicit = _parse_explicit_expiry(os.getenv("LITELLM_PROXY_TOKEN_EXPIRES_AT", ""))
        if explicit:
            return explicit
        return None

    def _refresh_hosted_env(self) -> bool:
        """Reload env vars from an optional hosted env file.

        This is intentionally generic so self-hosted or hosted deployments can
        refresh rotated credentials without wiring platform-specific logic into
        application code.
        """
        env_path = _litellm_env_file_path()
        if not env_path or not env_path.exists():
            return False
        try:
            values = dotenv_values(env_path)
            applied = False
            for key, value in values.items():
                if value is None:
                    continue
                if os.getenv(key) != value:
                    os.environ[key] = value
                    applied = True
            if applied:
                logger.info("Reloaded runtime env from %s", env_path)
            return applied
        except Exception:
            logger.warning("Failed to reload runtime env from %s", env_path, exc_info=True)
            return False

    def _get_litellm_client(self, force_refresh: bool = False):
        """Return an OpenAI-compatible LiteLLM client, refreshing creds from env."""
        expiry = self._litellm_token_expiry()
        if expiry and expiry <= datetime.now(timezone.utc):
            self._refresh_hosted_env()

        base_url = _env_first("LITELLM_BASE_URL", "LITELLM_PROXY_URL").rstrip("/")
        api_key = _env_first("LITELLM_API_KEY", "LITELLM_PROXY_API_KEY")
        if not base_url or not api_key:
            raise RuntimeError(
                "LiteLLM proxy not configured — set LITELLM_BASE_URL/LITELLM_API_KEY "
                "or LITELLM_PROXY_URL/LITELLM_PROXY_API_KEY"
            )

        signature = (base_url, api_key)
        if force_refresh:
            self._litellm_client = None
            self._litellm_signature = None
        if self._litellm_client is None or self._litellm_signature != signature:
            from openai import AsyncOpenAI
            self._litellm_client = AsyncOpenAI(api_key=api_key, base_url=base_url)
            self._litellm_signature = signature
        return self._litellm_client

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
        timeout_s: float | None = None,
    ) -> str:
        """Send a completion request through the configured model.

        Args:
            timeout_s: Per-call timeout in seconds. If None, uses provider default.
        """
        configured_model = self.resolve_model(config_key)
        provider = _resolve_provider(configured_model)
        runtime_model = self._resolve_runtime_model(configured_model)
        usage_model = self._usage_model(configured_model, runtime_model)

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

        # Circuit breaker check
        self._circuit_breaker.check(provider)

        effective_timeout = timeout_s if timeout_s is not None else PROVIDER_TIMEOUTS.get(provider, 60.0)

        t0 = time.monotonic()
        last_exc: Exception | None = None

        for attempt in range(MAX_RETRIES):
            try:
                if provider == "gemini":
                    text, in_tok, out_tok = await self._complete_gemini(
                        runtime_model, system_prompt, user_prompt, json_response, timeout_s=effective_timeout)
                elif provider == "anthropic":
                    text, in_tok, out_tok = await self._complete_anthropic(
                        runtime_model, system_prompt, user_prompt, json_response, timeout_s=effective_timeout)
                elif provider == "openai":
                    text, in_tok, out_tok = await self._complete_openai(
                        runtime_model, system_prompt, user_prompt, json_response, timeout_s=effective_timeout)
                elif provider == "deepseek":
                    text, in_tok, out_tok = await self._complete_deepseek(
                        runtime_model, system_prompt, user_prompt, json_response, timeout_s=effective_timeout)
                elif provider == "litellm":
                    text, in_tok, out_tok = await self._complete_litellm(
                        runtime_model, system_prompt, user_prompt, json_response, timeout_s=effective_timeout)
                elif provider == "ollama":
                    text, in_tok, out_tok = await self._complete_ollama(
                        runtime_model, system_prompt, user_prompt, json_response, timeout_s=effective_timeout)
                else:
                    raise ValueError(f"Unsupported provider: {provider}")

                # Success — record and return
                self._circuit_breaker.record_success(provider)
                break

            except Exception as exc:
                last_exc = exc
                self._circuit_breaker.record_failure(provider)

                if not _is_retryable(exc) or attempt == MAX_RETRIES - 1:
                    raise

                delay = RETRY_BASE_DELAY * (2 ** attempt)
                logger.warning(
                    "LLM call failed (attempt %d/%d, provider=%s): %s — retrying in %.1fs",
                    attempt + 1, MAX_RETRIES, provider, exc, delay,
                )
                await asyncio.sleep(delay)
        else:
            raise last_exc  # type: ignore[misc]

        elapsed = time.monotonic() - t0
        self.usage.record(provider, usage_model, config_key, in_tok, out_tok, elapsed)

        # Record cost to budget guard after successful call
        cost = estimate_cost(usage_model, in_tok, out_tok)
        if self._budget_guard:
            self._budget_guard.record_cost(cost)

        # Persist to SQLite (fire-and-forget — don't block LLM calls on DB writes)
        if self._store:
            try:
                task = asyncio.ensure_future(self._store.add_usage_record(
                    date.today().isoformat(), provider, usage_model, config_key,
                    in_tok, out_tok, cost,
                ))
                self._pending_usage_tasks.add(task)
                task.add_done_callback(self._pending_usage_tasks.discard)
            except Exception:
                logger.debug("Failed to persist usage record", exc_info=True)

        return text

    async def _complete_gemini(
        self, model: str, system_prompt: str, user_prompt: str, json_response: bool,
        *, timeout_s: float = 60.0,
    ) -> tuple[str, int, int]:
        if not self._gemini_client:
            raise RuntimeError("Gemini client not initialized — set GEMINI_API_KEY")

        from google.genai import types
        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            http_options=types.HttpOptions(timeout=int(timeout_s * 1000)),
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
        self, model: str, system_prompt: str, user_prompt: str, json_response: bool,
        *, timeout_s: float = 120.0,
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
            timeout=timeout_s,
        )

        in_tok = getattr(getattr(response, 'usage', None), 'input_tokens', 0) or 0
        out_tok = getattr(getattr(response, 'usage', None), 'output_tokens', 0) or 0
        return response.content[0].text, in_tok, out_tok

    async def _complete_ollama(
        self, model: str, system_prompt: str, user_prompt: str, json_response: bool,
        *, timeout_s: float = 120.0,
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
                resp = await client.post(url, json=body, timeout=timeout_s)
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
        self, model: str, system_prompt: str, user_prompt: str, json_response: bool,
        *, timeout_s: float = 60.0,
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
            "timeout": timeout_s,
        }
        if json_response:
            kwargs["response_format"] = {"type": "json_object"}

        response = await self._deepseek_client.chat.completions.create(**kwargs)

        in_tok = getattr(getattr(response, 'usage', None), 'prompt_tokens', 0) or 0
        out_tok = getattr(getattr(response, 'usage', None), 'completion_tokens', 0) or 0
        return response.choices[0].message.content, in_tok, out_tok

    async def _complete_litellm(
        self, model: str, system_prompt: str, user_prompt: str, json_response: bool,
        *, timeout_s: float = 60.0,
    ) -> tuple[str, int, int]:
        client = self._get_litellm_client()
        model_name = model.removeprefix("litellm/")

        effective_system = system_prompt
        if json_response:
            effective_system += "\n\nIMPORTANT: Respond with valid JSON only. No markdown, no explanation."

        kwargs: dict = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": effective_system},
                {"role": "user", "content": user_prompt},
            ],
            "timeout": timeout_s,
        }
        if json_response:
            kwargs["response_format"] = {"type": "json_object"}

        try:
            response = await client.chat.completions.create(**kwargs)
        except Exception as e:
            if _is_auth_error(e) and self._refresh_hosted_env():
                logger.warning("LiteLLM auth error detected; reloading env and retrying once")
                client = self._get_litellm_client(force_refresh=True)
                response = await client.chat.completions.create(**kwargs)
            else:
                raise

        in_tok = getattr(getattr(response, 'usage', None), 'prompt_tokens', 0) or 0
        out_tok = getattr(getattr(response, 'usage', None), 'completion_tokens', 0) or 0
        return response.choices[0].message.content, in_tok, out_tok

    async def _complete_openai(
        self, model: str, system_prompt: str, user_prompt: str, json_response: bool,
        *, timeout_s: float = 60.0,
    ) -> tuple[str, int, int]:
        if not self._openai_client:
            raise RuntimeError("OpenAI client not initialized — set OPENAI_API_KEY")

        kwargs: dict = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "timeout": timeout_s,
        }
        if json_response:
            kwargs["response_format"] = {"type": "json_object"}

        response = await self._openai_client.chat.completions.create(**kwargs)

        in_tok = getattr(getattr(response, 'usage', None), 'prompt_tokens', 0) or 0
        out_tok = getattr(getattr(response, 'usage', None), 'completion_tokens', 0) or 0
        return response.choices[0].message.content, in_tok, out_tok
