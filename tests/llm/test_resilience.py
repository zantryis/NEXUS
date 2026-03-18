"""Tests for LLM client resilience: retry, circuit breaker, timeouts."""

import time

import httpx
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from nexus.config.models import ModelsConfig
from nexus.llm.client import (
    CircuitBreaker,
    CircuitOpenError,
    LLMClient,
    _is_retryable,
)


# ── _is_retryable ──


class TestIsRetryable:
    def test_connection_error(self):
        assert _is_retryable(httpx.ConnectError("refused")) is True

    def test_timeout_exception(self):
        assert _is_retryable(httpx.TimeoutException("timed out")) is True

    def test_builtin_connection_error(self):
        assert _is_retryable(ConnectionError("reset")) is True

    def test_builtin_timeout_error(self):
        assert _is_retryable(TimeoutError("deadline")) is True

    def test_rate_limit_429(self):
        exc = Exception("rate limited")
        exc.status_code = 429
        assert _is_retryable(exc) is True

    def test_server_error_500(self):
        exc = Exception("internal")
        exc.status_code = 500
        assert _is_retryable(exc) is True

    def test_server_error_502(self):
        exc = Exception("bad gateway")
        exc.status_code = 502
        assert _is_retryable(exc) is True

    def test_server_error_503(self):
        exc = Exception("unavailable")
        exc.status_code = 503
        assert _is_retryable(exc) is True

    def test_server_error_on_response_attr(self):
        """Status code on exc.response.status_code."""
        resp = MagicMock(status_code=503)
        exc = Exception("unavailable")
        exc.response = resp
        assert _is_retryable(exc) is True

    def test_auth_error_not_retryable(self):
        exc = Exception("unauthorized")
        exc.status_code = 401
        assert _is_retryable(exc) is False

    def test_bad_request_not_retryable(self):
        exc = Exception("bad request")
        exc.status_code = 400
        assert _is_retryable(exc) is False

    def test_generic_value_error_not_retryable(self):
        assert _is_retryable(ValueError("bad config")) is False

    def test_rate_limit_message_pattern(self):
        assert _is_retryable(Exception("rate limit exceeded")) is True
        assert _is_retryable(Exception("Too Many Requests")) is True

    def test_overloaded_message_pattern(self):
        assert _is_retryable(Exception("model is overloaded")) is True

    def test_service_unavailable_message(self):
        assert _is_retryable(Exception("service unavailable, try later")) is True


# ── CircuitBreaker ──


class TestCircuitBreaker:
    def test_starts_closed(self):
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=60.0)
        cb.check("gemini")  # should not raise

    def test_opens_after_threshold_failures(self):
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=60.0)
        cb.record_failure("gemini")
        cb.record_failure("gemini")
        cb.check("gemini")  # still closed (2 < 3)
        cb.record_failure("gemini")
        with pytest.raises(CircuitOpenError, match="gemini"):
            cb.check("gemini")

    def test_success_resets_failure_count(self):
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=60.0)
        cb.record_failure("gemini")
        cb.record_failure("gemini")
        cb.record_success("gemini")
        cb.record_failure("gemini")
        cb.record_failure("gemini")
        cb.check("gemini")  # still closed — only 2 consecutive

    def test_independent_per_provider(self):
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=60.0)
        cb.record_failure("gemini")
        cb.record_failure("gemini")
        # gemini is open
        with pytest.raises(CircuitOpenError):
            cb.check("gemini")
        # anthropic is still closed
        cb.check("anthropic")

    def test_auto_resets_after_recovery_timeout(self):
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.1)
        cb.record_failure("gemini")
        cb.record_failure("gemini")
        with pytest.raises(CircuitOpenError):
            cb.check("gemini")
        time.sleep(0.15)
        # Should be half-open now — check doesn't raise
        cb.check("gemini")

    def test_half_open_success_closes(self):
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.1)
        cb.record_failure("gemini")
        cb.record_failure("gemini")
        time.sleep(0.15)
        cb.check("gemini")  # half-open, allows one call
        cb.record_success("gemini")
        cb.check("gemini")  # should be fully closed now

    def test_half_open_failure_reopens(self):
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.1)
        cb.record_failure("gemini")
        cb.record_failure("gemini")
        time.sleep(0.15)
        cb.check("gemini")  # half-open
        cb.record_failure("gemini")
        with pytest.raises(CircuitOpenError):
            cb.check("gemini")


# ── Retry in complete() ──


@pytest.fixture
def models_config():
    return ModelsConfig()


@pytest.fixture
def client(models_config):
    return LLMClient(models_config, api_key="test-key")


class TestRetry:
    @pytest.mark.asyncio
    async def test_retries_on_transient_error_then_succeeds(self, client):
        """Should retry on a transient error and return the successful response."""
        mock_response = MagicMock()
        mock_response.text = "success"
        mock_response.usage_metadata = MagicMock(
            prompt_token_count=10, candidates_token_count=5
        )

        call_count = 0

        async def flaky_generate(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise httpx.ConnectError("connection refused")
            return mock_response

        with patch.object(
            client._gemini_client.aio.models,
            "generate_content",
            side_effect=flaky_generate,
        ):
            result = await client.complete("filtering", "system", "user")

        assert result == "success"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_gives_up_after_max_retries(self, client):
        """Should raise after exhausting all retry attempts."""
        with patch.object(
            client._gemini_client.aio.models,
            "generate_content",
            new_callable=AsyncMock,
            side_effect=httpx.ConnectError("always fails"),
        ):
            with pytest.raises(httpx.ConnectError, match="always fails"):
                await client.complete("filtering", "system", "user")

    @pytest.mark.asyncio
    async def test_no_retry_on_non_retryable_error(self, client):
        """Should not retry auth errors or config errors."""
        call_count = 0

        async def auth_fail(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            exc = Exception("unauthorized")
            exc.status_code = 401
            raise exc

        with patch.object(
            client._gemini_client.aio.models,
            "generate_content",
            side_effect=auth_fail,
        ):
            with pytest.raises(Exception, match="unauthorized"):
                await client.complete("filtering", "system", "user")

        assert call_count == 1  # no retry

    @pytest.mark.asyncio
    async def test_retry_backoff_timing(self, client):
        """Verify exponential backoff delays between retries."""
        delays = []

        async def mock_sleep(seconds):
            delays.append(seconds)

        with patch.object(
            client._gemini_client.aio.models,
            "generate_content",
            new_callable=AsyncMock,
            side_effect=httpx.ConnectError("fail"),
        ), patch("nexus.llm.client.asyncio.sleep", side_effect=mock_sleep):
            with pytest.raises(httpx.ConnectError):
                await client.complete("filtering", "system", "user")

        # 3 attempts = 2 sleeps (between attempt 1→2 and 2→3)
        assert len(delays) == 2
        assert delays[0] == pytest.approx(1.0, abs=0.1)
        assert delays[1] == pytest.approx(2.0, abs=0.1)

    @pytest.mark.asyncio
    async def test_circuit_breaker_prevents_call(self, client):
        """When circuit is open, complete() raises without calling provider."""
        # Trip the circuit breaker for gemini
        for _ in range(5):
            client._circuit_breaker.record_failure("gemini")

        with pytest.raises(CircuitOpenError):
            await client.complete("filtering", "system", "user")

    @pytest.mark.asyncio
    async def test_retry_failures_feed_circuit_breaker(self, client):
        """Each failed retry attempt should count toward circuit breaker."""
        client._circuit_breaker = CircuitBreaker(
            failure_threshold=3, recovery_timeout=60.0
        )

        with patch.object(
            client._gemini_client.aio.models,
            "generate_content",
            new_callable=AsyncMock,
            side_effect=httpx.ConnectError("fail"),
        ), patch("nexus.llm.client.asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(httpx.ConnectError):
                await client.complete("filtering", "system", "user")

        # 3 attempts all failed → circuit should be open
        with pytest.raises(CircuitOpenError):
            client._circuit_breaker.check("gemini")


# ── Timeout ──


class TestTimeout:
    @pytest.mark.asyncio
    async def test_default_timeout_passed_to_provider(self, client):
        """complete() should pass the provider default timeout to _complete_gemini."""
        client._complete_gemini = AsyncMock(return_value=("ok", 5, 3))
        await client.complete("filtering", "system", "user")
        # Gemini default is 60.0
        call_kwargs = client._complete_gemini.call_args
        assert call_kwargs.kwargs["timeout_s"] == 60.0

    @pytest.mark.asyncio
    async def test_custom_timeout_overrides_default(self, client):
        """timeout_s parameter should override the provider default."""
        client._complete_gemini = AsyncMock(return_value=("ok", 5, 3))
        await client.complete("filtering", "system", "user", timeout_s=30.0)
        call_kwargs = client._complete_gemini.call_args
        assert call_kwargs.kwargs["timeout_s"] == 30.0

    @pytest.mark.asyncio
    async def test_timeout_triggers_retry(self, client):
        """A timeout should be treated as a retryable error."""
        mock_response = MagicMock()
        mock_response.text = "ok"
        mock_response.usage_metadata = MagicMock(
            prompt_token_count=5, candidates_token_count=3
        )

        call_count = 0

        async def timeout_then_ok(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise TimeoutError("deadline exceeded")
            return mock_response

        with patch.object(
            client._gemini_client.aio.models,
            "generate_content",
            side_effect=timeout_then_ok,
        ), patch("nexus.llm.client.asyncio.sleep", new_callable=AsyncMock):
            result = await client.complete("filtering", "system", "user")

        assert result == "ok"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_ollama_timeout_passed(self):
        """Ollama should use the timeout_s value for httpx."""
        client = LLMClient(
            ModelsConfig(filtering="ollama/qwen2"),
            ollama_base_url="http://localhost:11434",
        )

        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "message": {"content": "ok"},
            "prompt_eval_count": 5,
            "eval_count": 3,
        }
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as MockClient:
            mock_client_instance = AsyncMock()
            mock_client_instance.post.return_value = mock_resp
            mock_client_instance.__aenter__ = AsyncMock(
                return_value=mock_client_instance
            )
            mock_client_instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_client_instance

            await client._complete_ollama("ollama/qwen2", "system", "user", False, timeout_s=45.0)

            _, call_kwargs = mock_client_instance.post.call_args
            assert call_kwargs["timeout"] == 45.0
