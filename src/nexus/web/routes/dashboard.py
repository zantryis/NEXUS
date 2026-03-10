"""Dashboard landing page — today's feed."""

from datetime import date

from fastapi import APIRouter, Request

from nexus.web.app import get_store, get_templates

router = APIRouter()


@router.get("/")
async def dashboard(request: Request):
    store = get_store(request)
    templates = get_templates(request)
    today = date.today()

    topic_stats = await store.get_topic_stats()
    filter_stats = {}
    for ts in topic_stats:
        slug = ts["topic_slug"]
        fs = await store.get_filter_stats(slug, today)
        if fs.get("total", 0) > 0:
            filter_stats[slug] = fs

    # Active threads across all topics
    active_threads = await store.get_active_threads()

    # Source balance across all topics
    source_stats = await store.get_source_stats()
    balance = {}
    for s in source_stats:
        affil = s["affiliation"] or "unknown"
        balance[affil] = balance.get(affil, 0) + s["event_count"]

    return templates.TemplateResponse(request, "dashboard.html", {
        "today": today,
        "topic_stats": topic_stats,
        "filter_stats": filter_stats,
        "active_threads": active_threads,
        "source_balance": balance,
    })
