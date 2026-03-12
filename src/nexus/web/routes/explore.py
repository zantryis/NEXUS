"""Knowledge graph explorer — entity grid and detail views."""

from fastapi import APIRouter, Request

from nexus.web.app import get_store, get_templates
from nexus.web.graph import render_entity_network_svg

router = APIRouter(prefix="/explore")


@router.get("/")
async def explore_index(request: Request):
    """Entity grid with search and type filtering."""
    store = get_store(request)
    templates = get_templates(request)

    q = request.query_params.get("q", "").strip()
    entity_type = request.query_params.get("type", "")

    if q:
        entities = await store.search_entities(q, limit=50)
    else:
        entities = await store.get_all_entities()

    # Filter by type if specified
    if entity_type:
        entities = [e for e in entities if e.get("entity_type") == entity_type]

    # Sort by event count (computed from last_seen - first_seen spread + entity mentions)
    # For now, sort alphabetically
    entities.sort(key=lambda e: e.get("canonical_name", "").lower())

    return templates.TemplateResponse(request, "explore.html", {
        "entities": entities,
        "search_query": q,
        "filter_type": entity_type,
    })


@router.get("/entities/")
async def explore_entities_search(request: Request):
    """Redirect search to explore index."""
    q = request.query_params.get("q", "")
    store = get_store(request)
    templates = get_templates(request)

    if q:
        entities = await store.search_entities(q, limit=50)
    else:
        entities = await store.get_all_entities()

    return templates.TemplateResponse(request, "explore.html", {
        "entities": entities,
        "search_query": q,
        "filter_type": "",
    })


@router.get("/entities/{entity_id:int}")
async def explore_entity_detail(request: Request, entity_id: int):
    """Enhanced entity detail with SVG network visualization."""
    store = get_store(request)
    templates = get_templates(request)

    entity = await store.find_entity_by_id(entity_id)
    if not entity:
        return templates.TemplateResponse(request, "404.html", {
            "what": "Entity", "id": entity_id,
        }, status_code=404)

    events = await store.get_events_for_entity(entity_id)
    threads = await store.get_threads_for_entity(entity_id)
    related = await store.get_related_entities(entity_id)
    page = await store.get_page(f"entity:{entity_id}")

    # Generate SVG network
    graph_svg = render_entity_network_svg(
        center_name=entity["canonical_name"],
        related=related,
    )

    return templates.TemplateResponse(request, "explore_entity.html", {
        "entity": entity,
        "events": events,
        "threads": threads,
        "related": related,
        "page": page,
        "graph_svg": graph_svg,
    })
