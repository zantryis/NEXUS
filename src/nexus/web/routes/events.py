"""Event views — list and detail."""

from fastapi import APIRouter, Request

from nexus.web.app import get_store, get_templates

router = APIRouter(prefix="/events")


@router.get("/")
async def event_list(request: Request):
    store = get_store(request)
    templates = get_templates(request)

    topic = request.query_params.get("topic")
    events = await store.get_all_events(topic_slug=topic)

    return templates.TemplateResponse(request, "event_list.html", {
        "events": events,
        "filter_topic": topic,
    })


@router.get("/{event_id:int}")
async def event_detail(request: Request, event_id: int):
    store = get_store(request)
    templates = get_templates(request)

    event = await store.get_event_by_id(event_id)
    if not event:
        return templates.TemplateResponse(request, "404.html", {
            "what": "Event", "id": str(event_id),
        }, status_code=404)

    return templates.TemplateResponse(request, "event.html", {
        "event": event,
        "event_id": event_id,
    })
