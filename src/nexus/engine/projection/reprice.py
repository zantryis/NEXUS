"""Forecast repricing — daily re-evaluation of open predictions with a gate."""

import logging
from datetime import date, datetime, timezone

from nexus.config.models import NexusConfig
from nexus.engine.knowledge.store import KnowledgeStore
from nexus.llm.client import LLMClient

logger = logging.getLogger(__name__)

# Gate thresholds
DEFAULT_MAX_AGE_HOURS = 24
MIN_DAYS_BEFORE_RESOLUTION = 1
EXTREME_PROBABILITY_THRESHOLD = 0.05  # Skip if < 0.05 or > 0.95


async def should_reprice(
    question: dict,
    store: KnowledgeStore,
    *,
    max_age_hours: int = DEFAULT_MAX_AGE_HOURS,
    run_date: date | None = None,
) -> bool:
    """Gate: decide whether a forecast question should be repriced.

    Returns True if:
    - Last update (or creation) > max_age_hours ago
    - New events ingested for topic since last update
    Skips if:
    - Resolution within MIN_DAYS_BEFORE_RESOLUTION days
    - Extreme probability and no new events
    """
    today = run_date or date.today()
    resolution_date = date.fromisoformat(question["resolution_date"])
    days_until = (resolution_date - today).days

    # Skip if resolving very soon
    if days_until < MIN_DAYS_BEFORE_RESOLUTION:
        return False

    # Check staleness
    updated_at = question.get("updated_at")
    if updated_at:
        last_update = datetime.fromisoformat(updated_at)
        if last_update.tzinfo is None:
            last_update = last_update.replace(tzinfo=timezone.utc)
        age_hours = (datetime.now(timezone.utc) - last_update).total_seconds() / 3600
    else:
        # Never updated — use generated_for date as proxy (always stale)
        age_hours = max_age_hours + 1

    is_stale = age_hours > max_age_hours

    # Check for new events since last update
    topic_slug = question["topic_slug"]
    since_date = date.fromisoformat(updated_at[:10]) if updated_at else date.fromisoformat(question["generated_for"])
    new_events = await store.get_recent_events(
        topic_slug, days=max((today - since_date).days, 1),
        limit=5, reference_date=today,
    )
    has_new_events = len(new_events) > 0

    # Skip extreme probabilities without new evidence
    prob = question["probability"]
    is_extreme = prob < EXTREME_PROBABILITY_THRESHOLD or prob > (1 - EXTREME_PROBABILITY_THRESHOLD)
    if is_extreme and not has_new_events:
        return False

    return is_stale or has_new_events


async def reprice_forecast(
    question: dict,
    store: KnowledgeStore,
    llm: LLMClient,
    *,
    as_of: date | None = None,
) -> float:
    """Re-run the structural engine for a single forecast question.

    Returns the new probability.
    """
    from nexus.engine.projection.evidence import assemble_evidence_package
    from nexus.engine.projection.structural_engine import predict_structural

    run_date = as_of or date.today()
    question_text = question["question"]

    evidence = await assemble_evidence_package(store, question_text, as_of=run_date)
    assessment = await predict_structural(llm, evidence)

    return assessment.implied_probability


async def run_reprice_pass(
    store: KnowledgeStore,
    llm: LLMClient,
    config: NexusConfig,
    *,
    run_date: date | None = None,
) -> dict:
    """Top-level reprice pass: check all open forecasts, reprice where needed.

    Returns {total_open, repriced, skipped, errors}.
    """
    today = run_date or date.today()
    open_forecasts = await store.get_open_forecasts()

    stats = {"total_open": len(open_forecasts), "repriced": 0, "skipped": 0, "errors": 0}

    for question in open_forecasts:
        qid = question["forecast_question_id"]
        try:
            if not await should_reprice(question, store, run_date=today):
                stats["skipped"] += 1
                continue

            old_prob = question["probability"]
            new_prob = await reprice_forecast(question, store, llm, as_of=today)

            # Only update if meaningfully different (> 2pp)
            if abs(new_prob - old_prob) > 0.02:
                await store.update_forecast_probability(
                    qid, new_prob, source="daily_reprice",
                    market_probability=question.get("base_rate"),
                )
                logger.info(
                    f"Repriced Q{qid} ({question['topic_slug']}): "
                    f"{old_prob:.2f} → {new_prob:.2f}"
                )
                stats["repriced"] += 1
            else:
                # Touch updated_at to avoid re-checking tomorrow
                await store.update_forecast_probability(
                    qid, old_prob, source="daily_reprice_unchanged",
                    market_probability=question.get("base_rate"),
                )
                stats["skipped"] += 1

        except Exception as e:
            logger.warning(f"Reprice failed for Q{qid}: {e}", exc_info=True)
            stats["errors"] += 1

    logger.info(
        f"Reprice pass complete: {stats['repriced']} repriced, "
        f"{stats['skipped']} skipped, {stats['errors']} errors "
        f"(of {stats['total_open']} open)"
    )
    return stats
