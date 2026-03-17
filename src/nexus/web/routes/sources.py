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
    feed_health = await store.get_all_feed_health()

    # Build lookup by URL for template
    health_by_url: dict[str, dict] = {}
    for h in feed_health:
        health_by_url[h["source_url"]] = h

    dead_feeds = [h for h in feed_health if h["status"] == "dead"]

    return templates.TemplateResponse(request, "source_stats.html", {
        "stats": stats,
        "filter_topic": topic,
        "feed_health": feed_health,
        "health_by_url": health_by_url,
        "dead_feeds": dead_feeds,
    })
