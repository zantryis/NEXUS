"""Naked LLM baseline engine — no knowledge context, pure model world knowledge."""

from __future__ import annotations

import json
import logging
from datetime import date

from nexus.engine.projection.forecasting import ForecastEngineInput, _clip_probability
from nexus.engine.projection.models import ForecastQuestion, ForecastRun
from nexus.engine.projection.swarm import extremize

logger = logging.getLogger(__name__)

NAKED_SYSTEM_PROMPT = (
    "You are a calibrated probabilistic forecaster. You assess the probability of "
    "real-world outcomes based solely on your general knowledge. Be well-calibrated: "
    "when you say 60%, events like that should happen about 60% of the time. "
    "Avoid overconfidence. Return JSON only."
)

NAKED_USER_PROMPT = (
    "Today's date is {as_of}.\n\n"
    "Question: {question}\n\n"
    "What is the probability that the answer is YES?\n\n"
    'Return JSON: {{"probability": <float between 0.05 and 0.95>}}'
)


class NakedBenchmarkEngine:
    """Benchmark engine: just ask the LLM with zero knowledge context."""

    engine_name = "naked"

    async def predict_probability(
        self,
        question: str,
        *,
        llm=None,
        store=None,
        market_prob: float | None = None,
        as_of: date | None = None,
    ) -> float:
        if llm is None:
            return 0.50

        as_of_str = (as_of or date.today()).isoformat()
        user_prompt = NAKED_USER_PROMPT.format(as_of=as_of_str, question=question)

        try:
            response = await llm.complete(
                config_key="knowledge_summary",
                system_prompt=NAKED_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                json_response=True,
            )
            data = json.loads(response)
            raw_prob = float(data.get("probability", 0.50))
        except Exception as exc:
            logger.warning("Naked engine LLM failed: %s", exc)
            return 0.50

        # Compress overconfidence
        calibrated = extremize(raw_prob, gamma=0.8)
        return _clip_probability(calibrated)


class NakedForecastEngine:
    """Adapter for the full ForecastEngine protocol (topic-based runs)."""

    engine_name = "naked"

    async def generate(
        self,
        llm,
        payload: ForecastEngineInput,
        *,
        critic_pass: bool = True,
        max_questions: int = 4,
        calibration_data: list[dict] | None = None,
    ) -> ForecastRun:
        benchmark = NakedBenchmarkEngine()
        questions: list[ForecastQuestion] = []

        for thread in payload.threads[:max_questions]:
            question_text = f"Will the narrative '{thread.headline}' see significant developments in the next 14 days?"
            prob = await benchmark.predict_probability(
                question_text, llm=llm, as_of=payload.run_date
            )
            questions.append(
                ForecastQuestion(
                    question=question_text,
                    forecast_type="binary",
                    target_variable="thread_development",
                    probability=prob,
                    base_rate=0.50,
                    resolution_criteria=f"Significant events in thread '{thread.headline}'",
                    resolution_date=payload.run_date + __import__("datetime").timedelta(days=14),
                    horizon_days=14,
                    signpost=f"Watch for events in: {thread.headline}",
                    signals_cited=["engine:naked"],
                )
            )

        return ForecastRun(
            topic_slug=payload.topic_slug,
            topic_name=payload.topic_name,
            engine="naked",
            generated_for=payload.run_date,
            summary=f"Naked LLM baseline: {len(questions)} questions.",
            questions=questions,
            metadata={"engine": "naked"},
        )
