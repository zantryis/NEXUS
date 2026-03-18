"""Topic detail view."""

from datetime import date

from fastapi import APIRouter, Request

from nexus.web.app import get_store, get_templates
from nexus.web.routes.predictions import _compute_interest_score, _enrich_forecast

router = APIRouter(prefix="/topics")


@router.get("/{slug}")
async def topic_detail(request: Request, slug: str):
    store = get_store(request)
    templates = get_templates(request)
    today = date.today()

    active_threads = await store.get_active_threads(slug)
    recent_events = await store.get_recent_events(slug, days=7, reference_date=today)
    filter_stats = await store.get_filter_stats(slug, today)

    # Check for backstory page
    backstory = await store.get_page(f"backstory:{slug}")
    projection_page = await store.get_page(f"projection:{slug}")
    latest_projection = await store.get_latest_projection(slug)
    cross_topic_signals = await store.get_cross_topic_signals(slug, limit=5)

    # Featured predictions for this topic
    featured_predictions = []
    try:
        raw_predictions = await store.get_featured_predictions(slug, limit=5)
        for p in raw_predictions:
            _enrich_forecast(p, today)
            _compute_interest_score(p)
        # Fetch trend for each
        for p in raw_predictions:
            ext = p.get("external_ref")
            if ext:
                try:
                    history = await store.get_prediction_history(
                        ext, lookback_days=14,
                        engine=p.get("engine"),
                        run_label=p.get("run_label"),
                    )
                    if len(history) >= 2:
                        delta = history[-1]["probability"] - history[0]["probability"]
                        p["trend_delta"] = round(delta * 100, 1)
                        p["trend_direction"] = "up" if delta > 0.02 else "down" if delta < -0.02 else "stable"
                except Exception:
                    pass
        featured_predictions = sorted(raw_predictions, key=lambda p: p.get("interest_score", 0), reverse=True)
    except Exception:
        pass

    return templates.TemplateResponse(request, "topic.html", {
        "slug": slug,
        "active_threads": active_threads,
        "recent_events": recent_events,
        "filter_stats": filter_stats,
        "backstory": backstory,
        "projection_page": projection_page,
        "latest_projection": latest_projection,
        "cross_topic_signals": cross_topic_signals,
        "featured_predictions": featured_predictions,
        "today": today,
    })
