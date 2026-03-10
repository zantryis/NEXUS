"""Thread views — list and detail."""

from fastapi import APIRouter, Request

from nexus.web.app import get_store, get_templates

router = APIRouter(prefix="/threads")


@router.get("/")
async def thread_list(request: Request):
    store = get_store(request)
    templates = get_templates(request)

    status = request.query_params.get("status")
    topic = request.query_params.get("topic")
    threads = await store.get_all_threads(topic_slug=topic, status=status)

    return templates.TemplateResponse(request, "thread_list.html", {
        "threads": threads,
        "filter_status": status,
        "filter_topic": topic,
    })


@router.get("/{slug}")
async def thread_detail(request: Request, slug: str):
    store = get_store(request)
    templates = get_templates(request)

    thread = await store.get_thread(slug)
    if not thread:
        return templates.TemplateResponse(request, "404.html", {
            "what": "Thread", "id": slug,
        }, status_code=404)

    events = await store.get_events_for_thread(thread["id"])
    convergence = await store.get_convergence_for_thread(thread["id"])
    divergence = await store.get_divergence_for_thread(thread["id"])

    # Check for deep-dive page
    page = await store.get_page(f"thread:{slug}")

    return templates.TemplateResponse(request, "thread.html", {
        "thread": thread,
        "events": events,
        "convergence": convergence,
        "divergence": divergence,
        "page": page,
    })
