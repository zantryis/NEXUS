"""Cached narrative page viewer."""

from datetime import datetime

from fastapi import APIRouter, Request

from nexus.web.app import get_store, get_templates
from nexus.web.sanitize import safe_markdown

router = APIRouter(prefix="/pages")


@router.get("/{slug:path}")
async def page_view(request: Request, slug: str):
    store = get_store(request)
    templates = get_templates(request)

    page = await store.get_page(slug)
    if not page:
        return templates.TemplateResponse(request, "404.html", {
            "what": "Page", "id": slug,
        }, status_code=404)

    rendered = safe_markdown(page["content_md"])

    # Determine if page is stale
    is_stale = False
    if page.get("stale_after"):
        try:
            stale_dt = datetime.fromisoformat(page["stale_after"])
            is_stale = datetime.now() > stale_dt
        except (ValueError, TypeError):
            pass

    return templates.TemplateResponse(request, "page.html", {
        "page": page,
        "rendered_html": rendered,
        "is_stale": is_stale,
    })
