"""TTS backends — convert dialogue turns to audio bytes."""

import logging
from abc import ABC, abstractmethod

import httpx

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


class ElevenLabsTTS(TTSBackend):
    """ElevenLabs text-to-speech backend via REST API."""

    # Default voice IDs (ElevenLabs built-in voices)
    DEFAULT_VOICES = {
        "Rachel": "21m00Tcm4TlvDq8ikWAM",
        "Drew": "29vD33N1CtxCmqQRPOHJ",
        "Clyde": "2EiwWnXFnvU5JabPnv8n",
        "Domi": "AZnzlk1XvdvUeBnXmlld",
        "Bella": "EXAVITQu4vr4xnSDxMaL",
        "Josh": "TxGEqnHWrfWFTfGW9XjX",
        "Adam": "pNInz6obpgDQGcFmaJgB",
    }

    def __init__(
        self, api_key: str, voice_a: str = "Rachel", voice_b: str = "Drew",
        model: str = "eleven_multilingual_v2",
    ):
        self._api_key = api_key
        self._voice_a = voice_a
        self._voice_b = voice_b
        self._model = model

    async def synthesize(self, turn: DialogueTurn) -> bytes:
        voice_name = self._voice_a if turn.speaker == "A" else self._voice_b
        # Support both voice names and raw voice IDs
        voice_id = self.DEFAULT_VOICES.get(voice_name, voice_name)

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
                headers={
                    "xi-api-key": self._api_key,
                    "Content-Type": "application/json",
                },
                json={
                    "text": turn.text,
                    "model_id": self._model,
                    "voice_settings": {
                        "stability": 0.5,
                        "similarity_boost": 0.75,
                    },
                },
                timeout=60.0,
            )
            response.raise_for_status()
            return response.content  # MP3 bytes


class OpenAITTS(TTSBackend):
    """OpenAI TTS backend via openai SDK."""

    def __init__(self, api_key: str, voice_a: str = "nova", voice_b: str = "onyx",
                 model: str = "tts-1-hd"):
        self._api_key = api_key
        self._voice_a = voice_a
        self._voice_b = voice_b
        self._model = model

    async def synthesize(self, turn: DialogueTurn) -> bytes:
        from openai import AsyncOpenAI

        voice = self._voice_a if turn.speaker == "A" else self._voice_b
        client = AsyncOpenAI(api_key=self._api_key)
        response = await client.audio.speech.create(
            model=self._model,
            voice=voice,
            input=turn.text,
        )
        return response.content  # MP3 bytes


def get_tts_backend(
    config: AudioConfig,
    gemini_api_key: str | None = None,
    elevenlabs_api_key: str | None = None,
    openai_api_key: str | None = None,
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
    elif config.tts_backend == "elevenlabs":
        if not elevenlabs_api_key:
            raise ValueError("ELEVENLABS_API_KEY required for ElevenLabs TTS")
        return ElevenLabsTTS(
            api_key=elevenlabs_api_key,
            voice_a=config.voice_host_a,
            voice_b=config.voice_host_b,
            model=config.tts_model or "eleven_multilingual_v2",
        )
    elif config.tts_backend == "openai":
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY required for OpenAI TTS")
        return OpenAITTS(
            api_key=openai_api_key,
            voice_a=config.voice_host_a,
            voice_b=config.voice_host_b,
            model=config.tts_model or "tts-1-hd",
        )
    else:
        raise ValueError(f"Unsupported TTS backend: {config.tts_backend}")
