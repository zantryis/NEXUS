"""Run actor engine predictions against existing Kalshi-matched markets.

Uses real knowledge store data (entities, events, relationships) to generate
actor-based predictions for previously matched Kalshi markets.
Saves results as a new forecast_run with engine='actor'.
"""

import asyncio
import json
import logging
import os
import sys
from datetime import date, timedelta
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from nexus.config.models import ModelsConfig
from nexus.engine.knowledge.store import KnowledgeStore
from nexus.engine.projection.actor_engine import predict
from nexus.engine.projection.forecasting import _clip_probability
from nexus.engine.projection.models import ForecastQuestion, ForecastRun
from nexus.llm.client import LLMClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Curated markets — one per unique question, with implied probabilities
MARKETS = [
    {
        "ticker": "KXABRAHAMSA-29-JAN20",
        "question": "Will Israel and Saudi Arabia normalize relations during Trump's term?",
        "implied": 0.12,
    },
    {
        "ticker": "KXABRAHAMSY-29-JAN20",
        "question": "Will Israel and Syria normalize relations during Trump's term?",
        "implied": 0.08,
    },
    {
        "ticker": "KXABRAHAMQ-29-JAN20",
        "question": "Will Israel and Qatar normalize relations during Trump's term?",
        "implied": 0.06,
    },
    {
        "ticker": "KXOAIANTH-40-OAI",
        "question": "Will OpenAI IPO before Anthropic?",
        "implied": 0.58,
    },
    {
        "ticker": "KXOAIANTH-40-ANTH",
        "question": "Will Anthropic IPO before OpenAI?",
        "implied": 0.42,
    },
    {
        "ticker": "KXPRIMEENGCONSUMPTION-30-GAS",
        "question": "Will natural gas be the largest source of global primary energy consumption in 2030?",
        "implied": 0.35,
    },
    {
        "ticker": "KXPRIMEENGCONSUMPTION-30-COAL",
        "question": "Will coal be the largest source of global primary energy consumption in 2030?",
        "implied": 0.30,
    },
    {
        "ticker": "KXGREENTERRITORY-29-26APR",
        "question": "Will the US take control of any part of Greenland before April 26?",
        "implied": 0.04,
    },
]


async def main():
    db_path = Path("data/knowledge.db")
    if not db_path.exists():
        logger.error("Knowledge database not found at %s", db_path)
        return

    store = KnowledgeStore(db_path)
    await store.initialize()

    models = ModelsConfig()
    llm = LLMClient(
        models,
        api_key=os.environ.get("GEMINI_API_KEY"),
        deepseek_api_key=os.environ.get("DEEPSEEK_API_KEY"),
    )
    llm.set_store(store)

    run_date = date.today()
    questions: list[ForecastQuestion] = []

    for market in MARKETS:
        ticker = market["ticker"]
        question_text = market["question"]
        implied = market["implied"]

        logger.info("Predicting: %s (market implied: %.0f%%)", ticker, implied * 100)

        try:
            prediction = await predict(
                store,
                llm,
                question_text,
                run_date=run_date,
                market_prob=implied,
                max_actors=4,
            )
            our_prob = _clip_probability(prediction.calibrated_probability)
            reasoning = prediction.reasoning_chain[:500]
            actors_used = [a.actor for a in prediction.actors]
            logger.info(
                "  Result: %.0f%% (actors: %s)",
                our_prob * 100,
                ", ".join(actors_used),
            )
        except Exception as exc:
            logger.warning("  Failed: %s — using market implied", exc)
            our_prob = implied
            reasoning = f"Fallback to market implied due to: {exc}"
            actors_used = []

        questions.append(
            ForecastQuestion(
                question=question_text,
                forecast_type="binary",
                target_variable="kalshi_aligned",
                probability=our_prob,
                base_rate=implied,
                resolution_criteria=f"Kalshi market {ticker} resolution",
                resolution_date=run_date + timedelta(days=14),
                horizon_days=14,
                signpost=f"Kalshi market: {ticker}",
                external_ref=ticker,
                target_metadata={
                    "kalshi_ticker": ticker,
                    "kalshi_implied": implied,
                    "actors": actors_used,
                },
                signals_cited=[
                    f"kalshi:implied={implied:.3f}",
                    f"kalshi:our={our_prob:.3f}",
                    f"kalshi:gap={abs(our_prob - implied):.3f}",
                    f"engine:actor",
                ],
            )
        )

    # Save as a forecast run
    if questions:
        run = ForecastRun(
            topic_slug="kalshi-aligned",
            topic_name="Kalshi Market Alignment",
            engine="actor",
            generated_for=run_date,
            summary=f"Actor-engine Kalshi-aligned forecasts: {len(questions)} markets.",
            questions=questions,
            metadata={"kalshi_aligned": True, "markets_matched": len(questions), "engine": "actor"},
        )
        run_id = await store.save_forecast_run(run)
        logger.info("Saved forecast run %d with %d questions", run_id, len(questions))

        # Print summary
        print("\n" + "=" * 70)
        print("ACTOR ENGINE KALSHI PREDICTIONS")
        print("=" * 70)
        for q in questions:
            meta = q.target_metadata or {}
            gap = abs(q.probability - (meta.get("kalshi_implied") or 0))
            print(
                f"  {meta.get('kalshi_ticker', '?'):30s}  "
                f"Our: {q.probability:5.1%}  Market: {meta.get('kalshi_implied', 0):5.1%}  "
                f"Gap: {gap * 100:4.1f}pp"
            )
        print("=" * 70)

    await store.close()


if __name__ == "__main__":
    asyncio.run(main())
