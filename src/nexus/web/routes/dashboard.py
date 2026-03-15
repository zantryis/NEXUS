"""Dashboard — structured intelligence briefing homepage."""

import asyncio
import logging
import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse

from nexus.config.loader import load_config
from nexus.utils.health import build_health_snapshot
from nexus.web.app import get_store, get_templates

logger = logging.getLogger(__name__)

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


def _is_within(parent: Path, child: Path) -> bool:
    """Return True when child resolves within parent."""
    try:
        child.relative_to(parent)
    except ValueError:
        return False
    return True


def _config_error_health(message: str) -> dict:
    """Fallback snapshot used when config is missing or invalid."""
    return {
        "status": "critical",
        "issues": [{"severity": "critical", "message": message}],
        "pipeline": {
            "running": False,
            "last_run": None,
            "configured_topics": [],
            "configured_topic_count": 0,
            "last_run_topic_count": 0,
            "missing_topics": [],
        },
        "deliverables": {"briefing_today": False, "audio_today": False},
        "telegram": {"enabled": False, "chat_id_configured": False},
        "litellm": {
            "used": False,
            "configured": False,
            "base_url_present": False,
            "api_key_present": False,
            "steps": {},
            "alias_targets": {},
            "missing_aliases": {},
            "token_expires_at": None,
            "token_ttl_minutes": None,
        },
    }


def _load_runtime_config(config_path: Path):
    """Load config.yaml safely for the dashboard health checks."""
    try:
        return load_config(config_path)
    except Exception as e:
        logger.warning("Failed to load config for runtime health: %s", e, exc_info=True)
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

    config_path = data_dir / "config.yaml"
    health_snapshot = None
    if config_path.exists():
        config = _load_runtime_config(config_path)
        if config is not None:
            health_snapshot = await build_health_snapshot(config, data_dir, store)
        else:
            health_snapshot = _config_error_health("Config validation failed. Re-run setup or fix data/config.yaml.")

    # Pipeline run info
    last_run = await store.get_last_pipeline_run()
    pipeline_running = await store.is_pipeline_running()
    cooldown_active = False
    cooldown_remaining = ""
    if last_run and last_run["status"] == "completed" and last_run["completed_at"]:
        completed = datetime.fromisoformat(last_run["completed_at"])
        if completed.tzinfo is None:
            completed = completed.replace(tzinfo=timezone.utc)
        since = datetime.now(timezone.utc) - completed
        if since < timedelta(minutes=30):
            cooldown_active = True
            remaining = timedelta(minutes=30) - since
            cooldown_remaining = f"{int(remaining.total_seconds() // 60)}m"

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
        "setup_complete": request.query_params.get("setup") == "complete",
        "pipeline_status": getattr(request.app.state, "pipeline_status", None),
        "last_run": last_run,
        "pipeline_running": pipeline_running,
        "cooldown_active": cooldown_active,
        "cooldown_remaining": cooldown_remaining,
        "health": health_snapshot,
    })


@router.get("/briefings/{briefing_date}")
async def briefing_detail(request: Request, briefing_date: str):
    """Classic markdown briefing view (archive)."""
    from nexus.web.sanitize import safe_markdown
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
    briefing_html = safe_markdown(md_text)

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


@router.post("/api/pipeline/run")
async def trigger_pipeline(request: Request):
    """Trigger pipeline manually from dashboard."""
    store = get_store(request)
    data_dir = getattr(request.app.state, "data_dir", Path("data"))

    # Guard 1: already running?
    if await store.is_pipeline_running():
        return HTMLResponse(
            '<div id="pipeline-status" class="pipeline-controls-status status-running"'
            ' hx-get="/api/pipeline/status" hx-trigger="every 3s" hx-swap="outerHTML">'
            '<div class="pipeline-spinner"></div> Pipeline is already running...</div>'
        )

    # Guard 2: cooldown — last completed run < 30 min ago?
    last_run = await store.get_last_pipeline_run()
    if last_run and last_run["status"] == "completed" and last_run["completed_at"]:
        completed = datetime.fromisoformat(last_run["completed_at"])
        if completed.tzinfo is None:
            completed = completed.replace(tzinfo=timezone.utc)
        since = datetime.now(timezone.utc) - completed
        if since < timedelta(minutes=30):
            remaining = int((timedelta(minutes=30) - since).total_seconds() // 60)
            return HTMLResponse(
                f'<div id="pipeline-status" class="pipeline-controls-status status-cooldown">'
                f'Cooldown: available in {remaining}m</div>'
            )

    # Guard 3: config exists?
    config_path = data_dir / "config.yaml"
    if not config_path.exists():
        return HTMLResponse(
            '<div id="pipeline-status" class="pipeline-controls-status status-error">'
            'No config found. Complete setup first.</div>'
        )

    # Launch pipeline in background
    async def _run():
        try:
            from dotenv import load_dotenv
            from telegram import Bot
            from nexus.agent.delivery import deliver_briefing
            from nexus.config.loader import load_config
            from nexus.llm.client import LLMClient
            from nexus.engine.pipeline import run_pipeline

            load_dotenv(data_dir.parent / ".env")
            config = load_config(config_path)

            api_key = os.getenv("GEMINI_API_KEY")
            openai_api_key = os.getenv("OPENAI_API_KEY")
            elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")

            llm = LLMClient(
                config.models,
                api_key=api_key,
                anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
                deepseek_api_key=os.getenv("DEEPSEEK_API_KEY"),
                openai_api_key=openai_api_key,
                budget_config=config.budget,
            )

            smoke_cap = int(os.getenv("NEXUS_SMOKE_MODE", "0"))
            briefing_path = await run_pipeline(
                config, llm, data_dir,
                gemini_api_key=api_key,
                openai_api_key=openai_api_key,
                elevenlabs_api_key=elevenlabs_api_key,
                max_ingest=smoke_cap or None,
                trigger="manual",
            )

            if config.telegram.enabled and config.telegram.chat_id:
                token = os.getenv("TELEGRAM_BOT_TOKEN")
                if token:
                    today = date.today().isoformat()
                    audio_path = data_dir / "artifacts" / "audio" / f"{today}.mp3"
                    await deliver_briefing(
                        Bot(token=token),
                        config.telegram.chat_id,
                        briefing_path.read_text(),
                        audio_path if audio_path.exists() else None,
                    )
                else:
                    logger.warning("Manual pipeline completed but TELEGRAM_BOT_TOKEN is not set; skipping delivery")
        except Exception as e:
            logger.error(f"Manual pipeline run failed: {e}", exc_info=True)

    asyncio.create_task(_run())

    return HTMLResponse(
        '<div id="pipeline-status" class="pipeline-controls-status status-running"'
        ' hx-get="/api/pipeline/status" hx-trigger="every 3s" hx-swap="outerHTML">'
        '<div class="pipeline-spinner"></div> Pipeline started...</div>'
    )


@router.get("/api/health")
async def api_health(request: Request):
    """Machine-readable health snapshot for dashboard and ops checks."""
    data_dir = getattr(request.app.state, "data_dir", Path("data"))
    config_path = data_dir / "config.yaml"
    if not config_path.exists():
        return JSONResponse(_config_error_health("No config.yaml found."), status_code=503)

    config = _load_runtime_config(config_path)
    if config is None:
        return JSONResponse(
            _config_error_health("Config validation failed. Re-run setup or fix data/config.yaml."),
            status_code=503,
        )

    store = get_store(request)
    snapshot = await build_health_snapshot(config, data_dir, store)
    status_code = 503 if snapshot["status"] == "critical" else 200
    return JSONResponse(snapshot, status_code=status_code)


@router.get("/api/pipeline/status")
async def pipeline_status(request: Request):
    """HTMX poll endpoint — returns current pipeline status fragment."""
    store = get_store(request)

    running = await store.is_pipeline_running()
    if running:
        return HTMLResponse(
            '<div id="pipeline-status" class="pipeline-controls-status status-running"'
            ' hx-get="/api/pipeline/status" hx-trigger="every 3s" hx-swap="outerHTML">'
            '<div class="pipeline-spinner"></div> Pipeline running...</div>'
        )

    # Not running — show last run summary
    last_run = await store.get_last_pipeline_run()
    if not last_run:
        return HTMLResponse(
            '<div id="pipeline-status" class="pipeline-controls-status">'
            'No pipeline runs yet</div>'
        )

    if last_run["status"] == "failed":
        from html import escape
        error = escape(last_run.get("error") or "Unknown error")
        return HTMLResponse(
            f'<div id="pipeline-status" class="pipeline-controls-status status-error">'
            f'Last run failed: {error}</div>'
        )

    # Completed
    events = last_run.get("event_count", 0)
    cost = last_run.get("cost_usd", 0.0)
    return HTMLResponse(
        f'<div id="pipeline-status" class="pipeline-controls-status status-done">'
        f'Complete &mdash; {events} events, ${cost:.2f}'
        f' <a href="/" onclick="location.reload()">Refresh</a></div>'
    )


@router.get("/audio/{filename}")
async def serve_audio(request: Request, filename: str):
    """Serve audio briefing files."""
    data_dir = getattr(request.app.state, "data_dir", Path("data"))
    audio_dir = (data_dir / "artifacts" / "audio").resolve()
    audio_path = (audio_dir / filename).resolve()

    # Prevent path traversal: resolved path must stay within audio directory.
    if (
        not _is_within(audio_dir, audio_path)
        or not audio_path.exists()
        or not filename.endswith(".mp3")
    ):
        return JSONResponse({"error": "Not found"}, status_code=404)

    return FileResponse(
        audio_path,
        media_type="audio/mpeg",
        filename=filename,
    )
