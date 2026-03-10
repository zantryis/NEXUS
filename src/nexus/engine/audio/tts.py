"""TTS backends — convert dialogue turns to audio bytes."""

import logging
from abc import ABC, abstractmethod

from nexus.config.models import AudioConfig
from nexus.engine.audio.script import DialogueTurn

logger = logging.getLogger(__name__)


class TTSBackend(ABC):
    """Abstract TTS backend."""

    @abstractmethod
    async def synthesize(self, turn: DialogueTurn) -> bytes:
        """Synthesize a single dialogue turn to WAV audio bytes."""
        ...


class GeminiTTS(TTSBackend):
    """Gemini native TTS via google-genai SDK."""

    def __init__(self, client, model: str, voice_a: str, voice_b: str):
        self._client = client
        self._model = model
        self._voice_a = voice_a
        self._voice_b = voice_b

    async def synthesize(self, turn: DialogueTurn) -> bytes:
        from google.genai import types

        voice = self._voice_a if turn.speaker == "A" else self._voice_b

        response = await self._client.aio.models.generate_content(
            model=self._model,
            contents=turn.text,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name=voice,
                        )
                    )
                ),
            ),
        )

        # Extract audio bytes from response
        part = response.candidates[0].content.parts[0]
        return part.inline_data.data


def get_tts_backend(
    config: AudioConfig,
    gemini_api_key: str | None = None,
) -> TTSBackend:
    """Factory: create a TTS backend from config."""
    if config.tts_backend == "gemini":
        from google import genai
        client = genai.Client(api_key=gemini_api_key)
        return GeminiTTS(
            client=client,
            model=config.tts_model,
            voice_a=config.voice_host_a,
            voice_b=config.voice_host_b,
        )
    else:
        raise ValueError(f"Unsupported TTS backend: {config.tts_backend}")
