"""Interactive knowledge graph — D3 force-directed visualization."""

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from nexus.web.app import get_store, get_templates

router = APIRouter()


@router.get("/explore/graph")
async def graph_page(request: Request):
    """Full-page interactive knowledge graph."""
    templates = get_templates(request)
    focus = request.query_params.get("focus", "")
    return templates.TemplateResponse(request, "graph.html", {
        "focus_entity_id": focus,
    })


@router.get("/api/graph-data")
async def graph_data(request: Request):
    """JSON API: nodes + links for D3 force graph."""
    store = get_store(request)
    min_events = int(request.query_params.get("min_events", "3"))
    min_co = int(request.query_params.get("min_co", "2"))
    data = await store.get_graph_data(min_events=min_events, min_co=min_co)
    return JSONResponse(data)


@router.get("/api/entity-panel/{entity_id:int}")
async def entity_panel(request: Request, entity_id: int):
    """HTMX partial: entity details for the graph side panel."""
    store = get_store(request)
    templates = get_templates(request)

    entity = await store.find_entity_by_id(entity_id)
    if not entity:
        return templates.TemplateResponse(request, "partials/entity_panel.html", {
            "entity": None,
        })

    events = await store.get_events_for_entity(entity_id)
    events = sorted(events, key=lambda e: str(e.date), reverse=True)[:5]

    threads = await store.get_threads_for_entity(entity_id)
    threads = threads[:5]

    return templates.TemplateResponse(request, "partials/entity_panel.html", {
        "entity": entity,
        "events": events,
        "threads": threads,
    })
