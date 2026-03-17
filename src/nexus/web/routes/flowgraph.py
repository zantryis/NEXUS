"""Pipeline flowgraph — interactive visualization of the Nexus intelligence pipeline."""

from fastapi import APIRouter, Request

from nexus.web.app import get_templates

router = APIRouter()


@router.get("/flowgraph")
async def flowgraph_page(request: Request):
    """Full-page pipeline flowgraph visualization."""
    templates = get_templates(request)
    return templates.TemplateResponse(request, "flowgraph.html", {})
