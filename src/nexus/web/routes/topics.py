"""Topic detail view."""

from datetime import date

from fastapi import APIRouter, Request

from nexus.web.app import get_store, get_templates

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

    return templates.TemplateResponse(request, "topic.html", {
        "slug": slug,
        "active_threads": active_threads,
        "recent_events": recent_events,
        "filter_stats": filter_stats,
        "backstory": backstory,
        "today": today,
    })
