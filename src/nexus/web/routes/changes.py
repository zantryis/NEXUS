"""What Changed — diff view between synthesis snapshots."""

from datetime import date

from fastapi import APIRouter, Request

from nexus.engine.synthesis.diff import diff_syntheses, is_empty_diff
from nexus.engine.synthesis.knowledge import TopicSynthesis
from nexus.web.app import get_store, get_templates

router = APIRouter(prefix="/changes")


@router.get("/")
async def changes_index(request: Request):
    """Show changes overview — diffs for all topics for a given date."""
    store = get_store(request)
    templates = get_templates(request)

    target = request.query_params.get("date")
    today = date.today()
    try:
        target_date = date.fromisoformat(target) if target else today
    except ValueError:
        target_date = today

    # Get all topic slugs that have syntheses
    all_dates = await store.get_all_synthesis_dates()
    topic_stats = await store.get_topic_stats()
    topic_slugs = [s["topic_slug"] for s in topic_stats]

    diffs = []
    for slug in topic_slugs:
        current_data = await store.get_synthesis(slug, target_date)
        if not current_data:
            continue

        previous_data = await store.get_previous_synthesis(slug, target_date)
        current = TopicSynthesis(**current_data)

        if previous_data:
            previous = TopicSynthesis(**previous_data)
            diff = diff_syntheses(current, previous)
            diff["topic_slug"] = slug
            diff["topic_name"] = current.topic_name
            diff["has_previous"] = True
            diff["is_empty"] = is_empty_diff(diff)
        else:
            # First synthesis — everything is new
            diff = {
                "topic_slug": slug,
                "topic_name": current.topic_name,
                "has_previous": False,
                "is_empty": False,
                "new_threads": current.threads,
                "updated_threads": [],
                "resolved_threads": [],
                "new_convergence": [],
                "new_divergence": [],
                "new_entities": [],
                "source_balance_shift": {},
            }
        diffs.append(diff)

    # Get adjacent signals for this date
    adjacent = await store.get_adjacent_signals(target_date)

    return templates.TemplateResponse(request, "changes.html", {
        "target_date": target_date.isoformat(),
        "diffs": diffs,
        "adjacent_signals": adjacent,
        "available_dates": all_dates[:30],  # Last 30 dates
        "today": today.isoformat(),
    })
