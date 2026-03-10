"""Audio pipeline orchestrator — script gen → TTS → concatenation."""

import logging
from datetime import date
from pathlib import Path

from nexus.config.models import NexusConfig
from nexus.engine.audio.concat import concatenate_audio
from nexus.engine.audio.script import generate_dialogue_script
from nexus.engine.audio.tts import get_tts_backend
from nexus.engine.synthesis.knowledge import TopicSynthesis
from nexus.llm.client import LLMClient

logger = logging.getLogger(__name__)


async def run_audio_pipeline(
    llm: LLMClient,
    config: NexusConfig,
    syntheses: list[TopicSynthesis],
    data_dir: Path,
    gemini_api_key: str | None = None,
    report_date: date | None = None,
) -> Path | None:
    """Run the full audio pipeline: script → TTS → MP3.

    Returns path to the output MP3 or None if disabled/failed.
    """
    if not config.audio.enabled:
        return None

    if report_date is None:
        report_date = date.today()

    # 1. Generate dialogue script
    logger.info("Generating dialogue script...")
    script = await generate_dialogue_script(llm, config, syntheses, report_date=report_date)

    if not script.turns:
        logger.warning("Dialogue script has no turns, skipping audio.")
        return None

    # 2. Synthesize each turn via TTS
    logger.info(f"Synthesizing {len(script.turns)} dialogue turns via TTS...")
    tts = get_tts_backend(config.audio, gemini_api_key=gemini_api_key)

    segments: list[bytes] = []
    for i, turn in enumerate(script.turns):
        try:
            audio_bytes = await tts.synthesize(turn)
            segments.append(audio_bytes)
        except Exception as e:
            logger.warning(f"TTS failed for turn {i}: {e}")
            continue

    if not segments:
        logger.warning("No audio segments produced, skipping concatenation.")
        return None

    # 3. Concatenate and export
    today = report_date.isoformat()
    output_path = data_dir / "artifacts" / "audio" / f"{today}.mp3"
    logger.info(f"Concatenating {len(segments)} segments → {output_path}")

    result = await concatenate_audio(segments, output_path)
    logger.info(f"Audio pipeline complete: {result}")
    return result
