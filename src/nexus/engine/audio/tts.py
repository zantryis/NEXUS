"""TTS backends — convert dialogue turns to audio bytes."""

import logging
from abc import ABC, abstractmethod

import httpx

from nexus.config.models import AudioConfig
from nexus.engine.audio.script import DialogueTurn

logger = logging.getLogger(__name__)

# Per-backend defaults — used when configured voices/model belong to a different backend
BACKEND_DEFAULTS = {
    "gemini": {
        "model": "gemini-2.5-flash-preview-tts",
        "voice_a": "Kore",
        "voice_b": "Puck",
        "known_voices": {"Kore", "Puck", "Charon", "Fenrir", "Aoede", "Leda", "Orus", "Zephyr"},
    },
    "elevenlabs": {
        "model": "eleven_multilingual_v2",
        "voice_a": "Rachel",
        "voice_b": "Drew",
        "known_voices": {"Rachel", "Drew", "Clyde", "Domi", "Bella", "Josh", "Adam"},
    },
    "openai": {
        "model": "tts-1-hd",
        "voice_a": "nova",
        "voice_b": "onyx",
        "known_voices": {"alloy", "echo", "fable", "onyx", "nova", "shimmer"},
    },
}


def _resolve_voice(backend: str, configured: str, default_key: str) -> str:
    """Return configured voice if valid for backend, else the backend's default."""
    defaults = BACKEND_DEFAULTS.get(backend, {})
    known = defaults.get("known_voices", set())
    # Accept any voice the user explicitly set if it looks intentional for this backend
    if configured in known:
        return configured
    # Also accept raw ElevenLabs voice IDs (24-char alphanumeric)
    if backend == "elevenlabs" and len(configured) > 15:
        return configured
    # Fall back to backend default
    fallback = defaults.get(default_key, configured)
    if configured != fallback:
        logger.info(f"Voice '{configured}' not valid for {backend} TTS, using '{fallback}'")
    return fallback


def _resolve_model(backend: str, configured: str) -> str:
    """Return configured model if it belongs to this backend, else the backend's default."""
    defaults = BACKEND_DEFAULTS.get(backend, {})
    default_model = defaults.get("model", configured)
    # Detect cross-backend model strings
    other_backends = [b for b in BACKEND_DEFAULTS if b != backend]
    for other in other_backends:
        if BACKEND_DEFAULTS[other]["model"] in configured:
            logger.info(f"Model '{configured}' belongs to {other}, using '{default_model}' for {backend}")
            return default_model
    return configured


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
    """Factory: create a TTS backend from config.

    Resolves voice names and model IDs to backend-appropriate defaults when the
    configured values belong to a different backend (e.g., Gemini voices with
    ElevenLabs backend).
    """
    backend = config.tts_backend
    voice_a = _resolve_voice(backend, config.voice_host_a, "voice_a")
    voice_b = _resolve_voice(backend, config.voice_host_b, "voice_b")
    model = _resolve_model(backend, config.tts_model)

    if backend == "gemini":
        from google import genai
        client = genai.Client(api_key=gemini_api_key)
        return GeminiTTS(
            client=client,
            model=model,
            voice_a=voice_a,
            voice_b=voice_b,
        )
    elif backend == "elevenlabs":
        if not elevenlabs_api_key:
            raise ValueError("ELEVENLABS_API_KEY required for ElevenLabs TTS")
        return ElevenLabsTTS(
            api_key=elevenlabs_api_key,
            voice_a=voice_a,
            voice_b=voice_b,
            model=model,
        )
    elif backend == "openai":
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY required for OpenAI TTS")
        return OpenAITTS(
            api_key=openai_api_key,
            voice_a=voice_a,
            voice_b=voice_b,
            model=model,
        )
    else:
        raise ValueError(f"Unsupported TTS backend: {backend}")
