"""Render Chinese briefing + audio from cached syntheses."""

import asyncio
import os
import sys
from datetime import date
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nexus.config.loader import load_config
from nexus.engine.synthesis.knowledge import TopicSynthesis
from nexus.engine.synthesis.renderers import render_text_briefing
from nexus.engine.audio.pipeline import run_audio_pipeline
from nexus.llm.client import LLMClient


async def main():
    data_dir = Path("data")
    config = load_config(data_dir / "config.yaml")

    # Override language to Chinese
    config.user.output_language = "zh"

    llm = LLMClient(
        config.models,
        api_key=os.getenv("GEMINI_API_KEY"),
    )

    # Load cached syntheses
    today = date.today().isoformat()
    synth_dir = data_dir / "artifacts" / "syntheses" / today
    syntheses = []
    for path in sorted(synth_dir.glob("*.yaml")):
        raw = yaml.safe_load(path.read_text())
        syntheses.append(TopicSynthesis(**raw))
        print(f"Loaded synthesis: {path.name}")

    # Render Chinese briefing
    print("\nRendering Chinese briefing...")
    text = await render_text_briefing(llm, config, syntheses)
    out_path = data_dir / "artifacts" / "briefings" / f"{today}-zh.md"
    out_path.write_text(text)
    print(f"Saved to {out_path} ({len(text)} chars, ~{len(text.split())} words)")

    # Generate Chinese audio
    print("\nGenerating Chinese audio...")
    gemini_key = os.getenv("GEMINI_API_KEY")
    audio_path = await run_audio_pipeline(
        llm, config, syntheses, data_dir,
        gemini_api_key=gemini_key,
        lang_suffix="zh",
    )
    if audio_path:
        size_kb = audio_path.stat().st_size / 1024
        print(f"Audio saved to {audio_path} ({size_kb:.0f} KB)")
    else:
        print("Audio generation failed or produced no output")


if __name__ == "__main__":
    asyncio.run(main())
