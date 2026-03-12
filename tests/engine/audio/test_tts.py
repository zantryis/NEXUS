"""Tests for TTS backends."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from nexus.config.models import AudioConfig
from nexus.engine.audio.tts import GeminiTTS, ElevenLabsTTS, OpenAITTS, get_tts_backend
from nexus.engine.audio.script import DialogueTurn


def test_get_tts_backend_gemini():
    config = AudioConfig(tts_backend="gemini")
    backend = get_tts_backend(config, gemini_api_key="fake-key")
    assert isinstance(backend, GeminiTTS)


def test_get_tts_backend_unknown():
    config = AudioConfig(tts_backend="nonexistent")
    with pytest.raises(ValueError, match="Unsupported TTS backend"):
        get_tts_backend(config)


async def test_gemini_tts_synthesize():
    """Mock the google-genai client to test GeminiTTS."""
    # Build a mock response with audio data
    mock_audio_part = MagicMock()
    mock_audio_part.inline_data.data = b"fake-audio-bytes"
    mock_audio_part.inline_data.mime_type = "audio/wav"

    mock_response = MagicMock()
    mock_response.candidates = [MagicMock()]
    mock_response.candidates[0].content.parts = [mock_audio_part]

    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

    tts = GeminiTTS(
        client=mock_client,
        model="gemini-2.5-flash-preview-tts",
        voice_a="Kore",
        voice_b="Puck",
    )

    audio_bytes = await tts.synthesize(
        DialogueTurn(speaker="A", text="Hello world")
    )
    assert audio_bytes == b"fake-audio-bytes"

    # Verify the call used voice_a (Kore) for speaker A
    call_kwargs = mock_client.aio.models.generate_content.call_args.kwargs
    assert call_kwargs["model"] == "gemini-2.5-flash-preview-tts"


async def test_gemini_tts_speaker_b_voice():
    """Speaker B should use voice_b."""
    mock_audio_part = MagicMock()
    mock_audio_part.inline_data.data = b"speaker-b-audio"
    mock_audio_part.inline_data.mime_type = "audio/wav"

    mock_response = MagicMock()
    mock_response.candidates = [MagicMock()]
    mock_response.candidates[0].content.parts = [mock_audio_part]

    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

    tts = GeminiTTS(
        client=mock_client,
        model="gemini-2.5-flash-preview-tts",
        voice_a="Kore",
        voice_b="Puck",
    )

    audio_bytes = await tts.synthesize(
        DialogueTurn(speaker="B", text="Interesting point!")
    )
    assert audio_bytes == b"speaker-b-audio"


# ── ElevenLabs TTS ──


def test_get_tts_backend_elevenlabs():
    config = AudioConfig(tts_backend="elevenlabs")
    backend = get_tts_backend(config, elevenlabs_api_key="fake-key")
    assert isinstance(backend, ElevenLabsTTS)


def test_elevenlabs_requires_key():
    config = AudioConfig(tts_backend="elevenlabs")
    with pytest.raises(ValueError, match="ELEVENLABS_API_KEY"):
        get_tts_backend(config)


async def test_elevenlabs_synthesize():
    """Mock httpx to test ElevenLabs TTS."""
    tts = ElevenLabsTTS(api_key="fake-key", voice_a="Rachel", voice_b="Drew")

    mock_response = MagicMock()
    mock_response.content = b"elevenlabs-audio-bytes"
    mock_response.raise_for_status = MagicMock()

    with patch("nexus.engine.audio.tts.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

        audio = await tts.synthesize(DialogueTurn(speaker="A", text="Hello"))
        assert audio == b"elevenlabs-audio-bytes"


def test_elevenlabs_voice_mapping():
    """Voice names map to voice IDs."""
    tts = ElevenLabsTTS(api_key="key")
    assert "Rachel" in tts.DEFAULT_VOICES
    assert "Drew" in tts.DEFAULT_VOICES


# ── OpenAI TTS ──


def test_get_tts_backend_openai():
    config = AudioConfig(tts_backend="openai", tts_model="tts-1-hd")
    backend = get_tts_backend(config, openai_api_key="fake-key")
    assert isinstance(backend, OpenAITTS)


def test_openai_tts_requires_key():
    config = AudioConfig(tts_backend="openai")
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        get_tts_backend(config)
