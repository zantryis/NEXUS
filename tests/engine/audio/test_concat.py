"""Tests for audio concatenation."""

import struct

from nexus.engine.audio.concat import concatenate_audio, _make_wav_silence


def _make_tiny_wav(duration_ms: int = 100, sample_rate: int = 24000) -> bytes:
    """Generate a minimal valid WAV file with silence."""
    num_samples = int(sample_rate * duration_ms / 1000)
    # 16-bit mono PCM
    data = b"\x00\x00" * num_samples
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + len(data),
        b"WAVE",
        b"fmt ",
        16,           # chunk size
        1,            # PCM format
        1,            # mono
        sample_rate,
        sample_rate * 2,  # byte rate (16-bit mono)
        2,            # block align
        16,           # bits per sample
        b"data",
        len(data),
    )
    return header + data


def test_make_wav_silence():
    """Silence generator produces valid WAV bytes."""
    silence = _make_wav_silence(300, sample_rate=24000)
    assert silence[:4] == b"RIFF"
    assert b"WAVE" in silence[:12]


async def test_concatenate_audio_single_segment(tmp_path):
    """Single segment should produce a valid output file."""
    segments = [_make_tiny_wav(100)]
    output_path = tmp_path / "output.mp3"

    result = await concatenate_audio(segments, output_path)

    assert result.exists()
    assert result.stat().st_size > 0


async def test_concatenate_audio_multiple_segments(tmp_path):
    """Multiple segments should be concatenated with silence gaps."""
    segments = [_make_tiny_wav(100), _make_tiny_wav(100), _make_tiny_wav(100)]
    output_path = tmp_path / "output.mp3"

    result = await concatenate_audio(segments, output_path, silence_ms=300)

    assert result.exists()
    assert result.stat().st_size > 0
