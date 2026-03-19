"""Thread views — list and tracker detail."""

from fastapi import APIRouter, Request

from nexus.web.app import get_store, get_templates
from nexus.web.clustering import cluster_threads

router = APIRouter(prefix="/threads")


@router.get("/")
async def thread_list(request: Request):
    store = get_store(request)
    templates = get_templates(request)

    status = request.query_params.get("status", "active")
    if status == "all":
        status = None
    topic = request.query_params.get("topic")
    threads = await store.get_all_threads(topic_slug=topic, status=status)
    # Sort by most recently updated first
    threads.sort(key=lambda t: t.get("updated_at", ""), reverse=True)

    clusters = cluster_threads(threads)

    # Fetch available topics with thread counts for the filter bar
    cursor = await store.db.execute(
        "SELECT tt.topic_slug, COUNT(DISTINCT tt.thread_id) "
        "FROM thread_topics tt "
        "JOIN threads t ON tt.thread_id = t.id "
        "WHERE t.status NOT IN ('resolved') "
        "GROUP BY tt.topic_slug ORDER BY COUNT(DISTINCT tt.thread_id) DESC",
    )
    topic_counts = await cursor.fetchall()

    return templates.TemplateResponse(request, "thread_list.html", {
        "threads": threads,
        "clusters": clusters,
        "filter_status": status,
        "filter_topic": topic,
        "topic_counts": topic_counts,
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
    causal_links = await store.get_causal_links_for_thread(thread["id"])
    topics = await store.get_topics_for_thread(thread["id"])
    projection_items = await store.get_projection_items_for_thread(thread["id"])

    # Build full events with sources for the tracker
    events_with_sources = []
    for ev in events:
        events_with_sources.append({
            "date": ev.date,
            "summary": ev.summary,
            "significance": ev.significance,
            "sources": ev.sources,
            "entities": ev.entities,
        })

    page = await store.get_page(f"thread:{slug}")

    return templates.TemplateResponse(request, "thread.html", {
        "thread": thread,
        "events": events_with_sources,
        "convergence": convergence,
        "divergence": divergence,
        "causal_links": causal_links,
        "topics": topics,
        "page": page,
        "projection_items": projection_items,
    })
