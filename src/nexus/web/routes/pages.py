"""Cached narrative page viewer."""

import markdown

from fastapi import APIRouter, Request

from nexus.web.app import get_store, get_templates

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

    rendered = markdown.markdown(page["content_md"], extensions=["tables", "fenced_code"])

    return templates.TemplateResponse(request, "page.html", {
        "page": page,
        "rendered_html": rendered,
    })
