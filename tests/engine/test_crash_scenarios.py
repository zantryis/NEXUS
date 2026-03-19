"""Crash scenario tests — verify graceful degradation under failure conditions.

Block 3C: Pipeline crash mid-topic, all providers down, TTS partial failure.
"""

import pytest
import time
from unittest.mock import AsyncMock, MagicMock, patch

from nexus.config.models import NexusConfig, UserConfig, TopicConfig, AudioConfig
from nexus.engine.pipeline import run_pipeline, TopicPipelineResult
from nexus.engine.audio.pipeline import run_audio_pipeline
from nexus.engine.audio.script import DialogueScript, DialogueTurn
from nexus.engine.synthesis.knowledge import TopicSynthesis, NarrativeThread
from nexus.llm.client import (
    CircuitBreaker, CircuitOpenError, UsageTracker,
)


# ── Helpers ──


def _make_config(topics: list[TopicConfig] | None = None, audio: bool = False) -> NexusConfig:
    return NexusConfig(
        user=UserConfig(name="Test", timezone="America/Denver"),
        topics=topics or [
            TopicConfig(name="AI Research", priority="high", subtopics=["agents"]),
            TopicConfig(name="Iran-US Relations", priority="high", subtopics=["sanctions"]),
        ],
        audio=AudioConfig(enabled=audio),
    )


def _mock_topic_pipeline_success(topic_name: str) -> TopicPipelineResult:
    return TopicPipelineResult(
        synthesis=TopicSynthesis(
            topic_name=topic_name,
            threads=[NarrativeThread(headline=f"{topic_name} thread", significance=7)],
        ),
    )


# ── Scenario 1: Pipeline crash mid-topic ──


@pytest.mark.asyncio
async def test_pipeline_continues_after_topic_failure(tmp_path):
    """If one topic's pipeline crashes, remaining topics should still complete."""
    config = _make_config()
    mock_llm = AsyncMock()
    mock_llm.usage = UsageTracker()
    data_dir = tmp_path / "data"

    call_count = 0

    async def _topic_side_effect(llm, topic, data_dir, sources, *args, **kwargs):
        nonlocal call_count
        call_count += 1
        if topic.name == "AI Research":
            raise RuntimeError("LLM exploded mid-extraction")
        return _mock_topic_pipeline_success(topic.name)

    with patch("nexus.engine.pipeline.load_source_registry") as mock_registry, \
         patch("nexus.engine.pipeline.run_topic_pipeline", new_callable=AsyncMock, side_effect=_topic_side_effect), \
         patch("nexus.engine.pipeline.render_text_briefing") as mock_render, \
         patch("nexus.engine.pipeline.run_projection_pass", new_callable=AsyncMock):

        mock_registry.return_value = [{"url": "https://feed.com/rss", "id": "test"}]
        mock_render.return_value = "# Briefing\n\n## Iran-US Relations\n\nContent."

        briefing_path = await run_pipeline(config, mock_llm, data_dir)

    # Both topics were attempted
    assert call_count == 2
    # Briefing was still rendered (from the surviving topic)
    assert briefing_path.exists()
    assert "Iran-US" in briefing_path.read_text()
    # Render was called with only the successful synthesis
    syntheses_arg = mock_render.call_args.args[2]
    assert len(syntheses_arg) == 1
    assert syntheses_arg[0].topic_name == "Iran-US Relations"


@pytest.mark.asyncio
async def test_pipeline_records_failure_when_all_topics_crash(tmp_path):
    """If ALL topics crash, pipeline should still complete (empty briefing)."""
    config = _make_config()
    mock_llm = AsyncMock()
    mock_llm.usage = UsageTracker()
    data_dir = tmp_path / "data"

    async def _always_fail(*args, **kwargs):
        raise RuntimeError("Everything is broken")

    with patch("nexus.engine.pipeline.load_source_registry") as mock_registry, \
         patch("nexus.engine.pipeline.run_topic_pipeline", new_callable=AsyncMock, side_effect=_always_fail), \
         patch("nexus.engine.pipeline.render_text_briefing") as mock_render:

        mock_registry.return_value = [{"url": "https://feed.com/rss", "id": "test"}]
        mock_render.return_value = "# Empty Briefing\n\nNo topics today."

        briefing_path = await run_pipeline(config, mock_llm, data_dir)

    assert briefing_path.exists()
    # Render was called with empty syntheses list
    syntheses_arg = mock_render.call_args.args[2]
    assert len(syntheses_arg) == 0


# ── Scenario 2: Circuit breaker / all providers down ──


def test_circuit_breaker_opens_after_threshold():
    """Circuit should open after N consecutive failures."""
    cb = CircuitBreaker(failure_threshold=3, recovery_timeout=60.0)

    for _ in range(3):
        cb.record_failure("gemini")

    with pytest.raises(CircuitOpenError, match="gemini"):
        cb.check("gemini")


def test_circuit_breaker_resets_on_success():
    """A single success should reset the failure count."""
    cb = CircuitBreaker(failure_threshold=3, recovery_timeout=60.0)

    cb.record_failure("gemini")
    cb.record_failure("gemini")
    cb.record_success("gemini")

    # Should not raise — counter was reset
    cb.check("gemini")

    # Need 3 more failures to open
    cb.record_failure("gemini")
    cb.check("gemini")  # still closed (only 1 failure)


def test_circuit_breaker_half_open_recovery(monkeypatch):
    """After recovery timeout, circuit should allow one test call."""
    cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.1)

    cb.record_failure("openai")
    cb.record_failure("openai")

    with pytest.raises(CircuitOpenError):
        cb.check("openai")

    # Wait for recovery
    time.sleep(0.15)

    # Should allow one test call (half-open)
    cb.check("openai")  # no exception

    # If that call succeeds, circuit closes
    cb.record_success("openai")
    cb.check("openai")  # still fine


def test_circuit_breaker_half_open_failure_reopens():
    """If the half-open test call fails, circuit should re-open."""
    cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.1)

    cb.record_failure("deepseek")
    cb.record_failure("deepseek")

    time.sleep(0.15)
    cb.check("deepseek")  # enters half-open

    cb.record_failure("deepseek")  # test call failed → re-open

    with pytest.raises(CircuitOpenError):
        cb.check("deepseek")


def test_circuit_breaker_per_provider_isolation():
    """Failures in one provider should not affect other providers."""
    cb = CircuitBreaker(failure_threshold=2, recovery_timeout=60.0)

    cb.record_failure("gemini")
    cb.record_failure("gemini")

    with pytest.raises(CircuitOpenError, match="gemini"):
        cb.check("gemini")

    # Other providers unaffected
    cb.check("openai")
    cb.check("anthropic")


# ── Scenario 3: TTS partial failure ──


@pytest.mark.asyncio
async def test_tts_partial_failure_produces_audio():
    """If some TTS turns fail but others succeed, audio should still be produced."""
    config = NexusConfig(
        user=UserConfig(name="Test"),
        audio=AudioConfig(enabled=True),
    )
    mock_llm = MagicMock()

    mock_script = DialogueScript(turns=[
        DialogueTurn(speaker="A", text="Hello"),
        DialogueTurn(speaker="B", text="Welcome"),
        DialogueTurn(speaker="A", text="Let's discuss"),
    ])

    call_count = 0

    async def _flaky_synthesize(turn):
        nonlocal call_count
        call_count += 1
        if turn.speaker == "B":
            raise RuntimeError("TTS service temporarily unavailable")
        return b"fake-audio-bytes"

    mock_tts = AsyncMock()
    mock_tts.synthesize = _flaky_synthesize

    output_file = MagicMock()

    with patch("nexus.engine.audio.pipeline.generate_dialogue_script", new_callable=AsyncMock, return_value=mock_script), \
         patch("nexus.engine.audio.pipeline.get_tts_backend", return_value=mock_tts), \
         patch("nexus.engine.audio.pipeline.concatenate_audio", new_callable=AsyncMock, return_value=output_file) as mock_concat:

        result = await run_audio_pipeline(mock_llm, config, [], MagicMock())

    # Audio was produced despite the B turn failing
    assert result == output_file
    # concatenate_audio received the 2 successful segments (A turns)
    segments = mock_concat.call_args.args[0]
    assert len(segments) == 2


@pytest.mark.asyncio
async def test_tts_retries_before_giving_up():
    """TTS should retry 3 times per turn before marking it as failed."""
    config = NexusConfig(
        user=UserConfig(name="Test"),
        audio=AudioConfig(enabled=True),
    )
    mock_llm = MagicMock()

    mock_script = DialogueScript(turns=[
        DialogueTurn(speaker="A", text="Hello"),
    ])

    attempt_count = 0

    async def _retry_then_succeed(turn):
        nonlocal attempt_count
        attempt_count += 1
        if attempt_count < 3:
            raise RuntimeError("Transient error")
        return b"success-audio"

    mock_tts = AsyncMock()
    mock_tts.synthesize = _retry_then_succeed

    output_file = MagicMock()

    with patch("nexus.engine.audio.pipeline.generate_dialogue_script", new_callable=AsyncMock, return_value=mock_script), \
         patch("nexus.engine.audio.pipeline.get_tts_backend", return_value=mock_tts), \
         patch("nexus.engine.audio.pipeline.concatenate_audio", new_callable=AsyncMock, return_value=output_file), \
         patch("asyncio.sleep", new_callable=AsyncMock):  # skip retry delays

        result = await run_audio_pipeline(mock_llm, config, [], MagicMock())

    assert result == output_file
    assert attempt_count == 3  # 2 failures + 1 success


@pytest.mark.asyncio
async def test_tts_all_turns_fail_returns_none():
    """If ALL TTS calls fail (after retries), pipeline returns None."""
    config = NexusConfig(
        user=UserConfig(name="Test"),
        audio=AudioConfig(enabled=True),
    )
    mock_llm = MagicMock()

    mock_script = DialogueScript(turns=[
        DialogueTurn(speaker="A", text="Hello"),
        DialogueTurn(speaker="B", text="Hi"),
    ])

    mock_tts = AsyncMock()
    mock_tts.synthesize = AsyncMock(side_effect=RuntimeError("TTS completely down"))

    with patch("nexus.engine.audio.pipeline.generate_dialogue_script", new_callable=AsyncMock, return_value=mock_script), \
         patch("nexus.engine.audio.pipeline.get_tts_backend", return_value=mock_tts), \
         patch("asyncio.sleep", new_callable=AsyncMock):

        result = await run_audio_pipeline(mock_llm, config, [], MagicMock())

    assert result is None


@pytest.mark.asyncio
async def test_audio_failure_doesnt_crash_main_pipeline(tmp_path):
    """If the entire audio pipeline throws, the main pipeline should still complete."""
    config = _make_config(audio=True)
    mock_llm = AsyncMock()
    mock_llm.usage = UsageTracker()
    data_dir = tmp_path / "data"

    with patch("nexus.engine.pipeline.load_source_registry") as mock_registry, \
         patch("nexus.engine.pipeline.run_topic_pipeline", new_callable=AsyncMock) as mock_topic, \
         patch("nexus.engine.pipeline.render_text_briefing") as mock_render, \
         patch("nexus.engine.pipeline.run_audio_pipeline", new_callable=AsyncMock, side_effect=RuntimeError("Audio crashed")), \
         patch("nexus.engine.pipeline.run_projection_pass", new_callable=AsyncMock):

        mock_registry.return_value = [{"url": "https://feed.com/rss", "id": "test"}]
        mock_topic.return_value = _mock_topic_pipeline_success("AI Research")
        mock_render.return_value = "# Briefing\n\nContent."

        briefing_path = await run_pipeline(config, mock_llm, data_dir)

    # Briefing still produced despite audio failure
    assert briefing_path.exists()
