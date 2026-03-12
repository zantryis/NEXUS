"""Audio concatenation — join TTS segments into a single MP3."""

import asyncio
import io
import logging
import shutil
import struct
from pathlib import Path

from pydub import AudioSegment

logger = logging.getLogger(__name__)


def _make_wav_silence(duration_ms: int, sample_rate: int = 24000) -> bytes:
    """Generate silent WAV bytes for inter-turn gaps."""
    num_samples = int(sample_rate * duration_ms / 1000)
    data = b"\x00\x00" * num_samples  # 16-bit silence
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + len(data),
        b"WAVE",
        b"fmt ",
        16,
        1,            # PCM
        1,            # mono
        sample_rate,
        sample_rate * 2,
        2,
        16,
        b"data",
        len(data),
    )
    return header + data


async def concatenate_audio(
    segments: list[bytes],
    output_path: Path,
    silence_ms: int = 300,
) -> Path:
    """Concatenate WAV audio segments with silence gaps, export as MP3."""
    if not shutil.which("ffmpeg") and not shutil.which("avconv"):
        raise RuntimeError(
            "ffmpeg is required to export podcast audio. Install ffmpeg and retry."
        )

    def _concat():
        combined = AudioSegment.empty()
        silence = AudioSegment.silent(duration=silence_ms)

        for i, seg_bytes in enumerate(segments):
            try:
                # Try WAV first
                segment = AudioSegment.from_wav(io.BytesIO(seg_bytes))
            except Exception:
                try:
                    # Gemini TTS returns raw PCM (24kHz, 16-bit, mono)
                    segment = AudioSegment.from_raw(
                        io.BytesIO(seg_bytes),
                        sample_width=2, frame_rate=24000, channels=1,
                    )
                except Exception as e:
                    logger.warning(f"Failed to decode segment {i}: {e}")
                    continue
            if i > 0:
                combined += silence
            combined += segment

        output_path.parent.mkdir(parents=True, exist_ok=True)
        combined.export(str(output_path), format="mp3")
        return output_path

    # Run in thread to avoid blocking the event loop
    return await asyncio.to_thread(_concat)
