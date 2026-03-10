"""CLI entry point: python -m nexus <command>."""

import asyncio
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv


def run_engine():
    """Run the daily engine pipeline."""
    load_dotenv()
    from nexus.config.loader import load_config
    from nexus.config.models import ModelsConfig
    from nexus.engine.pipeline import run_pipeline
    from nexus.llm.client import LLMClient

    data_dir = Path("data")
    config_path = data_dir / "config.yaml"

    if not config_path.exists():
        print(f"Config not found at {config_path}. Copy data/config.example.yaml to get started.")
        sys.exit(1)

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY not set. Add it to .env")
        sys.exit(1)

    config = load_config(config_path)
    llm = LLMClient(config.models, api_key=api_key)

    briefing_path = asyncio.run(run_pipeline(config, llm, data_dir))
    print(f"Briefing generated: {briefing_path}")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    if len(sys.argv) < 2:
        print("Usage: python -m nexus <engine|setup>")
        sys.exit(1)

    command = sys.argv[1]

    if command == "engine":
        run_engine()
    elif command == "setup":
        print("Setup wizard not yet implemented.")
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
