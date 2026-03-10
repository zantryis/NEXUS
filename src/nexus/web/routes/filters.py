"""Filter log views."""

from datetime import date

from fastapi import APIRouter, Request

from nexus.web.app import get_store, get_templates

router = APIRouter(prefix="/filters")


@router.get("/{topic_slug}/{run_date}")
async def filter_log(request: Request, topic_slug: str, run_date: str):
    store = get_store(request)
    templates = get_templates(request)

    d = date.fromisoformat(run_date)
    log = await store.get_filter_log(topic_slug, d)
    stats = await store.get_filter_stats(topic_slug, d)

    return templates.TemplateResponse(request, "filter_log.html", {
        "topic_slug": topic_slug,
        "run_date": run_date,
        "log": log,
        "stats": stats,
    })
