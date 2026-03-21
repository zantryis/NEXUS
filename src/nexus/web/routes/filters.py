"""Filter log views — index, per-topic, and detail pages."""

from datetime import date

from fastapi import APIRouter, Request

from nexus.web.app import get_store, get_templates

router = APIRouter(prefix="/filters")


@router.get("/")
async def filter_index(request: Request):
    """Filter log overview — all topics, recent dates, stats."""
    store = get_store(request)
    templates = get_templates(request)

    topics = await store.get_filter_log_topics()
    today = date.today()

    # Build per-topic summary for recent dates
    topic_summaries = []
    for slug in topics:
        dates = await store.get_filter_log_dates(slug)
        if not dates:
            continue
        latest_date = dates[0]
        stats = await store.get_filter_stats(slug, date.fromisoformat(latest_date))
        topic_summaries.append({
            "slug": slug,
            "latest_date": latest_date,
            "dates": dates[:14],  # last 14 runs
            "stats": stats,
        })

    # Adjacent signals for today
    adjacent = await store.get_adjacent_signals(today)

    return templates.TemplateResponse(request, "filter_index.html", {
        "topic_summaries": topic_summaries,
        "adjacent_signals": adjacent,
        "today": today.isoformat(),
    })


@router.get("/{topic_slug}/{run_date}")
async def filter_log(request: Request, topic_slug: str, run_date: str):
    store = get_store(request)
    templates = get_templates(request)

    try:
        d = date.fromisoformat(run_date)
    except ValueError:
        return templates.TemplateResponse(request, "404.html", {"message": "Invalid date format"}, status_code=404)
    log = await store.get_filter_log(topic_slug, d)
    stats = await store.get_filter_stats(topic_slug, d)

    # Get available dates for navigation
    dates = await store.get_filter_log_dates(topic_slug)

    return templates.TemplateResponse(request, "filter_log.html", {
        "topic_slug": topic_slug,
        "run_date": run_date,
        "log": log,
        "stats": stats,
        "available_dates": dates[:30],
    })
