"""Live test: run pipeline in --quick mode on a niche topic.

Usage: .venv/bin/python scripts/test_quick_mode.py
"""
import asyncio
import logging
import os
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("quick-test")


async def main():
    from nexus.config.models import (
        NexusConfig, AudioConfig, FutureProjectionConfig, ModelsConfig,
        SourcesConfig, BudgetConfig, BreakingNewsConfig, BriefingConfig,
        TelegramConfig, TopicConfig, UserConfig,
    )
    from nexus.engine.pipeline import run_pipeline
    from nexus.llm.client import LLMClient

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY not set")
        return

    # Niche topic — "International Rugby Union"
    config = NexusConfig(
        preset="balanced",
        user=UserConfig(name="Test", timezone="America/Denver"),
        topics=[
            TopicConfig(
                name="International Rugby Union",
                priority="medium",
                subtopics=["Six Nations", "Rugby World Cup", "Super Rugby"],
                source_languages=["en"],
                filter_threshold=5.0,
                scope="narrow",
            ),
        ],
        models=ModelsConfig(),
        audio=AudioConfig(enabled=False),  # --quick: off
        future_projection=FutureProjectionConfig(enabled=False),  # --quick: off
        sources=SourcesConfig(discover_new_sources=True),
        budget=BudgetConfig(),
        breaking_news=BreakingNewsConfig(enabled=False),
        briefing=BriefingConfig(style="analytical"),
        telegram=TelegramConfig(enabled=False),
    )

    llm = LLMClient(config.models, api_key=api_key, budget_config=config.budget)

    data_dir = Path("data")
    t0 = time.monotonic()

    logger.info("Starting quick pipeline for 'International Rugby Union'")
    logger.info("  max_ingest=20, audio=off, projections=off, discovery=on")

    try:
        briefing_path = await run_pipeline(
            config, llm, data_dir,
            gemini_api_key=api_key,
            max_ingest=20,  # --quick cap
            trigger="manual",
        )
        elapsed = time.monotonic() - t0
        logger.info(f"Done in {elapsed:.0f}s — briefing at {briefing_path}")

        # Print briefing preview
        if briefing_path and briefing_path.exists():
            text = briefing_path.read_text()
            print(f"\n{'='*60}")
            print(f"BRIEFING PREVIEW ({len(text)} chars, {elapsed:.0f}s)")
            print(f"{'='*60}")
            # Print first 1500 chars
            print(text[:1500])
            if len(text) > 1500:
                print(f"\n... ({len(text) - 1500} more chars)")
        else:
            print(f"\nNo briefing generated (path={briefing_path})")

        # Print cost
        cost = llm.usage.cost_summary()
        print(f"\nCost: ${cost.get('total_cost', 0):.4f}")
        print(f"LLM calls: {cost.get('total_calls', 0)}")

    except Exception as e:
        elapsed = time.monotonic() - t0
        logger.error(f"Pipeline failed after {elapsed:.0f}s: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())
