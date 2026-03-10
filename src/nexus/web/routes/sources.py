"""Source analytics view."""

from fastapi import APIRouter, Request

from nexus.web.app import get_store, get_templates

router = APIRouter(prefix="/sources")


@router.get("/")
async def source_stats(request: Request):
    store = get_store(request)
    templates = get_templates(request)

    topic = request.query_params.get("topic")
    stats = await store.get_source_stats(topic_slug=topic)

    return templates.TemplateResponse(request, "source_stats.html", {
        "stats": stats,
        "filter_topic": topic,
    })
