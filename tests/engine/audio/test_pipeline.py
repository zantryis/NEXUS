"""Tests for audio pipeline orchestrator."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import date
from pathlib import Path

from nexus.config.models import NexusConfig, UserConfig, AudioConfig
from nexus.engine.audio.pipeline import run_audio_pipeline
from nexus.engine.audio.script import DialogueScript, DialogueTurn
from nexus.engine.synthesis.knowledge import TopicSynthesis, NarrativeThread
from nexus.engine.knowledge.events import Event


def _make_config(**audio_kw) -> NexusConfig:
    return NexusConfig(
        user=UserConfig(name="Tristan"),
        audio=AudioConfig(**audio_kw),
    )


def _make_syntheses() -> list[TopicSynthesis]:
    return [TopicSynthesis(
        topic_name="Test Topic",
        threads=[NarrativeThread(
            headline="Test Thread",
            events=[Event(date=date(2026, 3, 10), summary="Test event",
                          significance=7, entities=["A"],
                          sources=[{"outlet": "reuters"}])],
            significance=7,
        )],
    )]


async def test_run_audio_pipeline_disabled(tmp_path):
    """Pipeline should return None when audio is disabled."""
    config = _make_config(enabled=False)
    result = await run_audio_pipeline(
        llm=AsyncMock(), config=config,
        syntheses=_make_syntheses(), data_dir=tmp_path,
    )
    assert result is None


@patch("nexus.engine.audio.pipeline.generate_dialogue_script")
@patch("nexus.engine.audio.pipeline.get_tts_backend")
@patch("nexus.engine.audio.pipeline.concatenate_audio")
async def test_run_audio_pipeline_success(mock_concat, mock_get_tts, mock_gen_script, tmp_path):
    """Full pipeline: script gen → TTS → concat."""
    # Mock script generation
    mock_gen_script.return_value = DialogueScript(turns=[
        DialogueTurn(speaker="A", text="Hello"),
        DialogueTurn(speaker="B", text="Hi there"),
    ])

    # Mock TTS backend
    mock_tts = AsyncMock()
    mock_tts.synthesize = AsyncMock(return_value=b"fake-wav")
    mock_get_tts.return_value = mock_tts

    # Mock concatenation
    output_file = tmp_path / "audio" / "2026-03-10.mp3"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_bytes(b"fake-mp3")
    mock_concat.return_value = output_file

    config = _make_config()
    result = await run_audio_pipeline(
        llm=AsyncMock(), config=config,
        syntheses=_make_syntheses(), data_dir=tmp_path,
        gemini_api_key="fake-key",
    )

    assert result == output_file
    mock_gen_script.assert_called_once()
    assert mock_tts.synthesize.call_count == 2
    mock_concat.assert_called_once()


@patch("nexus.engine.audio.pipeline.generate_dialogue_script")
async def test_run_audio_pipeline_empty_script(mock_gen_script, tmp_path):
    """Pipeline should return None if script has no turns."""
    mock_gen_script.return_value = DialogueScript(turns=[])

    config = _make_config()
    result = await run_audio_pipeline(
        llm=AsyncMock(), config=config,
        syntheses=_make_syntheses(), data_dir=tmp_path,
    )
    assert result is None


@patch("nexus.engine.audio.pipeline.generate_dialogue_script")
@patch("nexus.engine.audio.pipeline.get_tts_backend")
@patch("nexus.engine.audio.pipeline.concatenate_audio")
async def test_run_audio_pipeline_lang_suffix(mock_concat, mock_get_tts, mock_gen_script, tmp_path):
    """Pipeline with lang_suffix produces language-suffixed output file."""
    mock_gen_script.return_value = DialogueScript(turns=[
        DialogueTurn(speaker="A", text="你好"),
    ])
    mock_tts = AsyncMock()
    mock_tts.synthesize = AsyncMock(return_value=b"fake-wav")
    mock_get_tts.return_value = mock_tts

    output_file = tmp_path / "audio" / "2026-03-10-zh.mp3"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_bytes(b"fake-mp3")
    mock_concat.return_value = output_file

    config = _make_config()
    result = await run_audio_pipeline(
        llm=AsyncMock(), config=config,
        syntheses=_make_syntheses(), data_dir=tmp_path,
        report_date=date(2026, 3, 10),
        lang_suffix="zh",
    )

    assert result == output_file
    # Verify the output path passed to concat has the language suffix
    concat_call_path = mock_concat.call_args.args[1]
    assert "-zh.mp3" in str(concat_call_path)
