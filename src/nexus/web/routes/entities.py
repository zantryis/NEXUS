"""Entity views — list, search, detail."""

from fastapi import APIRouter, Request

from nexus.web.app import get_store, get_templates

router = APIRouter(prefix="/entities")


@router.get("/")
async def entity_list(request: Request):
    store = get_store(request)
    templates = get_templates(request)

    topic = request.query_params.get("topic")
    q = request.query_params.get("q")

    if q:
        entities = await store.search_entities(q)
    else:
        entities = await store.get_all_entities(topic_slug=topic)

    return templates.TemplateResponse(request, "entity_list.html", {
        "entities": entities,
        "filter_topic": topic,
        "search_query": q,
    })


@router.get("/{entity_id:int}")
async def entity_detail(request: Request, entity_id: int):
    store = get_store(request)
    templates = get_templates(request)

    entity = await store.find_entity_by_id(entity_id)
    if not entity:
        return templates.TemplateResponse(request, "404.html", {
            "what": "Entity", "id": str(entity_id),
        }, status_code=404)

    events = await store.get_events_for_entity(entity_id)
    threads = await store.get_threads_for_entity(entity_id)
    related = await store.get_related_entities(entity_id)

    # Check for entity profile page
    page = await store.get_page(f"entity:{entity_id}")

    return templates.TemplateResponse(request, "entity.html", {
        "entity": entity,
        "events": events,
        "threads": threads,
        "related": related,
        "page": page,
    })
