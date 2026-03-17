"""Tests for pipeline artifact cleanup on crash."""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from nexus.engine.audio.pipeline import run_audio_pipeline
from nexus.config.models import NexusConfig, UserConfig, AudioConfig


@pytest.fixture
def audio_config():
    return NexusConfig(
        user=UserConfig(name="Test"),
        audio=AudioConfig(enabled=True),
    )


@pytest.mark.asyncio
async def test_audio_cleanup_on_concatenation_failure(audio_config, tmp_path):
    """If concatenation fails, the partial output file should be removed."""
    mock_llm = MagicMock()
    mock_script = MagicMock()
    mock_script.turns = [MagicMock(), MagicMock()]

    mock_tts = AsyncMock()
    mock_tts.synthesize.return_value = b"fake-audio-bytes"

    output_dir = tmp_path / "artifacts" / "audio"
    output_dir.mkdir(parents=True)

    with patch("nexus.engine.audio.pipeline.generate_dialogue_script", new_callable=AsyncMock, return_value=mock_script), \
         patch("nexus.engine.audio.pipeline.get_tts_backend", return_value=mock_tts), \
         patch("nexus.engine.audio.pipeline.concatenate_audio", new_callable=AsyncMock, side_effect=RuntimeError("ffmpeg failed")):

        with pytest.raises(RuntimeError, match="ffmpeg failed"):
            await run_audio_pipeline(mock_llm, audio_config, [], tmp_path)

    # Partial output file should not exist
    from datetime import date
    expected_path = output_dir / f"{date.today().isoformat()}.mp3"
    assert not expected_path.exists()


@pytest.mark.asyncio
async def test_audio_cleanup_leaves_no_partial_on_tts_total_failure(audio_config, tmp_path):
    """If ALL TTS calls fail, no partial output file should exist."""
    mock_llm = MagicMock()
    mock_script = MagicMock()
    mock_script.turns = [MagicMock()]

    mock_tts = AsyncMock()
    mock_tts.synthesize.side_effect = RuntimeError("TTS down")

    output_dir = tmp_path / "artifacts" / "audio"
    output_dir.mkdir(parents=True)

    with patch("nexus.engine.audio.pipeline.generate_dialogue_script", new_callable=AsyncMock, return_value=mock_script), \
         patch("nexus.engine.audio.pipeline.get_tts_backend", return_value=mock_tts):

        # Should return None (no segments), not crash
        result = await run_audio_pipeline(mock_llm, audio_config, [], tmp_path)

    assert result is None
    # No files created
    assert list(output_dir.iterdir()) == []
