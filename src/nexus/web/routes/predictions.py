"""Predictions results page — showcases prediction outcomes and Brier scores."""

from datetime import date, timedelta

from fastapi import APIRouter, Request

from nexus.web.app import get_store, get_templates

router = APIRouter()


@router.get("/predictions")
async def predictions_page(request: Request):
    """Predictions results page."""
    store = get_store(request)
    templates = get_templates(request)

    # Get all Kalshi-aligned forecasts with resolution status
    end = date.today()
    start = end - timedelta(days=90)
    questions = await store.get_forecast_questions_between(start=start, end=end)

    kalshi_questions = [
        q for q in questions
        if q.get("target_variable") == "kalshi_aligned"
    ]

    # Also get actor-engine forecasts
    actor_questions = [
        q for q in questions
        if q.get("engine") == "actor" and q.get("target_variable") != "kalshi_aligned"
    ]

    all_forecasts = kalshi_questions + actor_questions

    # Compute summary stats
    resolved = [q for q in all_forecasts if q.get("outcome_status") == "resolved"]
    pending = [q for q in all_forecasts if q.get("outcome_status") in (None, "pending")]

    brier_scores = [q["brier_score"] for q in resolved if q.get("brier_score") is not None]
    mean_brier = round(sum(brier_scores) / len(brier_scores), 4) if brier_scores else None

    # Market comparison for Kalshi questions
    market_briers = []
    for q in resolved:
        meta = q.get("target_metadata") or {}
        market_prob = meta.get("kalshi_implied")
        if market_prob is not None and q.get("resolved_bool") is not None:
            outcome = 1.0 if q["resolved_bool"] else 0.0
            market_briers.append(round((float(market_prob) - outcome) ** 2, 4))

    market_mean_brier = round(sum(market_briers) / len(market_briers), 4) if market_briers else None

    # Accuracy: within 10pp of outcome
    accurate = 0
    for q in resolved:
        if q.get("resolved_bool") is not None:
            outcome = 1.0 if q["resolved_bool"] else 0.0
            if abs(q["probability"] - outcome) <= 0.10:
                accurate += 1
    accuracy = round(accurate / len(resolved) * 100, 1) if resolved else None

    summary = {
        "total": len(all_forecasts),
        "resolved": len(resolved),
        "pending": len(pending),
        "mean_brier": mean_brier,
        "market_mean_brier": market_mean_brier,
        "accuracy": accuracy,
    }

    # Enrich forecasts with display fields
    for q in all_forecasts:
        meta = q.get("target_metadata") or {}
        q["ticker"] = meta.get("kalshi_ticker", "")
        q["market_prob"] = meta.get("kalshi_implied")
        if q.get("market_prob") is not None and q.get("probability") is not None:
            q["gap_pp"] = round(abs(q["probability"] - q["market_prob"]) * 100, 1)
        else:
            q["gap_pp"] = None
        if q.get("outcome_status") == "resolved":
            q["status_label"] = "hit" if q.get("brier_score", 1) < 0.25 else "miss"
        else:
            q["status_label"] = "pending"

    return templates.TemplateResponse(request, "predictions.html", {
        "forecasts": all_forecasts,
        "summary": summary,
    })
