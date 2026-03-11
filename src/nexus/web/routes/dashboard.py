"""Dashboard landing page — today's feed."""

from datetime import date
from pathlib import Path

from fastapi import APIRouter, Request

from nexus.web.app import get_store, get_templates

router = APIRouter()

# Topic emoji mapping
TOPIC_EMOJI = {
    "iran": "&#127758;", "energy": "&#9889;", "ai": "&#129302;",
    "formula": "&#127950;&#65039;", "f1": "&#127950;&#65039;",
    "climate": "&#127793;", "china": "&#127464;&#127475;",
    "crypto": "&#129689;", "tech": "&#128187;", "finance": "&#128200;",
    "health": "&#129657;", "space": "&#128640;", "defense": "&#128737;&#65039;",
}


def _topic_emoji(slug: str) -> str:
    for key, emoji in TOPIC_EMOJI.items():
        if key in slug.lower():
            return emoji
    return "&#128196;"


@router.get("/")
async def dashboard(request: Request):
    store = get_store(request)
    templates = get_templates(request)
    today = date.today()

    topic_stats = await store.get_topic_stats()

    # Active threads across all topics
    active_threads = await store.get_active_threads()

    # Recent events (last 3 days, all topics) for timeline
    recent_events = []
    for ts in topic_stats:
        slug = ts["topic_slug"]
        events = await store.get_recent_events(slug, days=3, limit=8)
        for e in events:
            recent_events.append({"topic": slug, "event": e})
    # Sort by date desc, limit to 15
    recent_events.sort(key=lambda x: x["event"].date, reverse=True)
    recent_events = recent_events[:15]

    # Source balance
    source_stats = await store.get_source_stats()
    balance = {}
    for s in source_stats:
        affil = s["affiliation"] or "unknown"
        balance[affil] = balance.get(affil, 0) + s["event_count"]

    # Today's cost
    today_cost = await store.get_daily_cost(today.isoformat())

    # Briefing availability
    data_dir = getattr(request.app.state, "data_dir", Path("data"))
    briefing_path = data_dir / "artifacts" / "briefings" / f"{today.isoformat()}.md"
    audio_path = data_dir / "artifacts" / "audio" / f"{today.isoformat()}.mp3"

    # Aggregate counts
    total_events = sum(ts["event_count"] for ts in topic_stats)
    total_entities = sum(ts["entity_count"] for ts in topic_stats)

    return templates.TemplateResponse(request, "dashboard.html", {
        "today": today,
        "topic_stats": topic_stats,
        "active_threads": active_threads,
        "recent_events": recent_events,
        "source_balance": balance,
        "today_cost": today_cost,
        "briefing_available": briefing_path.exists(),
        "audio_available": audio_path.exists(),
        "total_events": total_events,
        "total_entities": total_entities,
        "topic_emoji": _topic_emoji,
    })
