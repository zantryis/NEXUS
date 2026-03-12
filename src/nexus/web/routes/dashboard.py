"""Dashboard — structured intelligence briefing homepage."""

from datetime import date, timedelta
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, FileResponse

from nexus.web.app import get_store, get_templates

router = APIRouter()

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


def _find_audio(data_dir: Path, target_date: date) -> dict:
    """Find audio files for a given date. Returns dict with available languages."""
    audio_dir = data_dir / "artifacts" / "audio"
    result = {"available": False, "primary": None, "languages": []}

    primary = audio_dir / f"{target_date.isoformat()}.mp3"
    if primary.exists():
        result["available"] = True
        result["primary"] = target_date.isoformat()
        result["languages"].append({"code": "en", "label": "English", "file": f"{target_date.isoformat()}.mp3"})

    # Check for language variants (e.g., 2026-03-11-zh.mp3)
    if audio_dir.exists():
        for f in sorted(audio_dir.glob(f"{target_date.isoformat()}-*.mp3")):
            lang_code = f.stem.split("-", 3)[-1]  # "2026-03-11-zh" → "zh"
            lang_labels = {
                "zh": "Chinese", "es": "Spanish", "fr": "French",
                "de": "German", "ar": "Arabic", "fa": "Persian",
                "ja": "Japanese", "ko": "Korean", "pt": "Portuguese",
                "ru": "Russian", "hi": "Hindi", "it": "Italian",
            }
            result["available"] = True
            result["languages"].append({
                "code": lang_code,
                "label": lang_labels.get(lang_code, lang_code.upper()),
                "file": f.name,
            })

    return result


def _find_briefing(data_dir: Path, target_date: date) -> tuple[date, str] | None:
    """Find briefing markdown for target_date, falling back to most recent."""
    path = data_dir / "artifacts" / "briefings" / f"{target_date.isoformat()}.md"
    if path.exists():
        return target_date, path.read_text()

    briefings_dir = data_dir / "artifacts" / "briefings"
    if not briefings_dir.exists():
        return None

    files = sorted(briefings_dir.glob("????-??-??.md"), reverse=True)
    for f in files:
        try:
            d = date.fromisoformat(f.stem)
            return d, f.read_text()
        except ValueError:
            continue
    return None


async def _build_topics_data(store, max_threads: int = 3, max_events: int = 2):
    """Build structured topic data for the web briefing."""
    topic_stats = await store.get_topic_stats()
    topics_data = []

    for ts in topic_stats:
        slug = ts["topic_slug"]
        threads_raw = await store.get_active_threads(slug)
        # Top threads by significance
        threads_raw.sort(key=lambda t: t.get("significance", 0), reverse=True)
        threads_raw = threads_raw[:max_threads]

        topic_threads = []
        for t in threads_raw:
            tid = t["id"]
            # Get latest events for this thread
            events = await store.get_events_for_thread(tid)
            # Most recent first, limited
            events.reverse()
            latest_events = []
            for ev in events[:max_events]:
                latest_events.append({
                    "date": ev.date,
                    "summary": ev.summary,
                    "significance": ev.significance,
                    "sources": ev.sources,
                    "entities": ev.entities,
                })

            # Get top convergence/divergence
            convergence = await store.get_convergence_for_thread(tid)
            divergence = await store.get_divergence_for_thread(tid)

            topic_threads.append({
                "slug": t["slug"],
                "headline": t["headline"],
                "significance": t["significance"],
                "status": t["status"],
                "key_entities": t.get("key_entities", [])[:5],
                "updated_at": t.get("updated_at", ""),
                "events": latest_events,
                "convergence": convergence[:2] if convergence else [],
                "divergence": divergence[:1] if divergence else [],
            })

        topics_data.append({
            "slug": slug,
            "emoji": _topic_emoji(slug),
            "event_count": ts["event_count"],
            "thread_count": ts.get("thread_count", len(threads_raw)),
            "threads": topic_threads,
        })

    return topics_data


@router.get("/")
async def homepage(request: Request):
    store = get_store(request)
    templates = get_templates(request)
    today = date.today()
    data_dir = getattr(request.app.state, "data_dir", Path("data"))

    # Build structured briefing from DB
    topics_data = await _build_topics_data(store)

    # Check if classic briefing exists
    result = _find_briefing(data_dir, today)
    has_classic_briefing = result is not None
    briefing_date = result[0] if result else today

    # Previous date navigation
    prev_date = briefing_date - timedelta(days=1)
    prev_path = data_dir / "artifacts" / "briefings" / f"{prev_date.isoformat()}.md"
    has_prev = prev_path.exists()

    # Sidebar data
    active_threads = await store.get_active_threads()
    active_threads.sort(key=lambda t: t.get("significance", 0), reverse=True)
    active_threads = active_threads[:8]

    topic_stats = await store.get_topic_stats()
    today_cost = await store.get_daily_cost(today.isoformat())
    total_events = sum(ts["event_count"] for ts in topic_stats)
    source_stats = await store.get_source_stats()

    # Breaking alerts (last 24h, grouped by topic)
    raw_alerts = await store.get_recent_breaking_alerts(hours=24)
    breaking_alerts: dict[str, list] = {}
    for a in raw_alerts:
        slug = a["topic_slug"] or "general"
        breaking_alerts.setdefault(slug, []).append(a)

    # Audio data
    audio_info = _find_audio(data_dir, briefing_date)

    return templates.TemplateResponse(request, "dashboard.html", {
        "today": today,
        "briefing_date": briefing_date,
        "topics_data": topics_data,
        "has_classic_briefing": has_classic_briefing,
        "has_prev": has_prev,
        "prev_date": prev_date,
        "active_threads": active_threads,
        "topic_stats": topic_stats,
        "topic_emoji": _topic_emoji,
        "today_cost": today_cost,
        "total_events": total_events,
        "total_sources": len(source_stats),
        "breaking_alerts": breaking_alerts,
        "audio": audio_info,
    })


@router.get("/briefings/{briefing_date}")
async def briefing_detail(request: Request, briefing_date: str):
    """Classic markdown briefing view (archive)."""
    import markdown
    store = get_store(request)
    templates = get_templates(request)
    data_dir = getattr(request.app.state, "data_dir", Path("data"))

    try:
        target = date.fromisoformat(briefing_date)
    except ValueError:
        return templates.TemplateResponse(request, "404.html", {
            "what": "briefing", "id": briefing_date,
        }, status_code=404)

    result = _find_briefing(data_dir, target)
    if not result:
        return templates.TemplateResponse(request, "404.html", {
            "what": "briefing", "id": briefing_date,
        }, status_code=404)

    actual_date, md_text = result
    briefing_html = markdown.markdown(md_text, extensions=["tables", "fenced_code"])

    prev_date = actual_date - timedelta(days=1)
    next_date = actual_date + timedelta(days=1)
    prev_path = data_dir / "artifacts" / "briefings" / f"{prev_date.isoformat()}.md"
    next_path = data_dir / "artifacts" / "briefings" / f"{next_date.isoformat()}.md"

    active_threads = await store.get_active_threads()
    active_threads.sort(key=lambda t: t.get("significance", 0), reverse=True)
    active_threads = active_threads[:8]

    topic_stats = await store.get_topic_stats()
    today_cost = await store.get_daily_cost(date.today().isoformat())
    total_events = sum(ts["event_count"] for ts in topic_stats)
    source_stats = await store.get_source_stats()

    audio_info = _find_audio(data_dir, actual_date)

    return templates.TemplateResponse(request, "briefing_classic.html", {
        "today": date.today(),
        "briefing_date": actual_date,
        "briefing_html": briefing_html,
        "has_prev": prev_path.exists(),
        "prev_date": prev_date,
        "has_next": next_path.exists(),
        "next_date": next_date,
        "active_threads": active_threads,
        "topic_stats": topic_stats,
        "topic_emoji": _topic_emoji,
        "today_cost": today_cost,
        "total_events": total_events,
        "total_sources": len(source_stats),
        "audio": audio_info,
    })


@router.get("/audio/{filename}")
async def serve_audio(request: Request, filename: str):
    """Serve audio briefing files."""
    data_dir = getattr(request.app.state, "data_dir", Path("data"))
    audio_path = data_dir / "artifacts" / "audio" / filename

    if not audio_path.exists() or not filename.endswith(".mp3"):
        return JSONResponse({"error": "Not found"}, status_code=404)

    return FileResponse(
        audio_path,
        media_type="audio/mpeg",
        filename=filename,
    )
