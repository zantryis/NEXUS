"""Web-based setup wizard — multi-step form for first-run configuration."""

import asyncio
import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, Request, Form
from fastapi.responses import RedirectResponse, HTMLResponse

from nexus.cli.setup import PRESET_INFO, TOPIC_CHOICES
from nexus.config.writer import write_config, write_env
from nexus.web.app import get_templates

logger = logging.getLogger(__name__)

router = APIRouter()

# In-memory session storage for wizard state (keyed by cookie UUID)
# Each session: {"step": int, "preset": str, "keys": {}, "topics": [], ...}


def _get_session(request: Request) -> tuple[str, dict]:
    """Get or create a wizard session from cookie."""
    sessions = getattr(request.app.state, "setup_sessions", None)
    if sessions is None:
        request.app.state.setup_sessions = {}
        sessions = request.app.state.setup_sessions

    session_id = request.cookies.get("nexus_setup")
    if session_id and session_id in sessions:
        return session_id, sessions[session_id]

    session_id = str(uuid.uuid4())
    sessions[session_id] = {"step": 1}
    return session_id, sessions[session_id]


def _set_cookie(response, session_id: str):
    """Set the session cookie on a response."""
    response.set_cookie("nexus_setup", session_id, httponly=True, max_age=3600, samesite="lax")
    return response


def _data_dir(request: Request) -> Path:
    """Get data directory from app state."""
    return getattr(request.app.state, "data_dir", Path("data"))


@router.get("/setup")
async def setup_start(request: Request):
    """Step 1: Welcome + preset selection."""
    templates = get_templates(request)
    session_id, session = _get_session(request)
    response = templates.TemplateResponse(request, "setup/step1_welcome.html", {
        "presets": PRESET_INFO,
        "selected_preset": session.get("preset"),
        "step": 1,
        "total_steps": 6,
    })
    return _set_cookie(response, session_id)


@router.post("/setup/step/1")
async def setup_step1(request: Request, preset: str = Form(...)):
    """Save preset, advance to step 2."""
    session_id, session = _get_session(request)
    session["preset"] = preset

    # Find required key for this preset
    required_key = None
    for name, _, _, key in PRESET_INFO:
        if name == preset:
            required_key = key
            break
    session["required_key"] = required_key

    # If free preset, skip API key step
    if not required_key:
        session["keys"] = {}
        response = RedirectResponse(url="/setup/step/3", status_code=303)
    else:
        response = RedirectResponse(url="/setup/step/2", status_code=303)
    return _set_cookie(response, session_id)


@router.get("/setup/step/2")
async def setup_step2_get(request: Request):
    """Step 2: Required API key."""
    templates = get_templates(request)
    session_id, session = _get_session(request)
    response = templates.TemplateResponse(request, "setup/step2_keys.html", {
        "required_key": session.get("required_key", ""),
        "preset": session.get("preset", ""),
        "error": None,
        "step": 2,
        "total_steps": 6,
    })
    return _set_cookie(response, session_id)


@router.post("/setup/step/2")
async def setup_step2_post(request: Request, api_key: str = Form(...)):
    """Validate and save required API key."""
    session_id, session = _get_session(request)
    api_key = api_key.strip()
    if not api_key:
        templates = get_templates(request)
        response = templates.TemplateResponse(request, "setup/step2_keys.html", {
            "required_key": session.get("required_key", ""),
            "preset": session.get("preset", ""),
            "error": "API key cannot be empty.",
            "step": 2,
            "total_steps": 6,
        })
        return _set_cookie(response, session_id)

    session["keys"] = {session["required_key"]: api_key}
    response = RedirectResponse(url="/setup/step/3", status_code=303)
    return _set_cookie(response, session_id)


@router.get("/setup/step/3")
async def setup_step3_get(request: Request):
    """Step 3: Optional keys (Telegram, TTS)."""
    templates = get_templates(request)
    session_id, session = _get_session(request)
    response = templates.TemplateResponse(request, "setup/step3_optional.html", {
        "preset": session.get("preset", ""),
        "step": 3,
        "total_steps": 6,
    })
    return _set_cookie(response, session_id)


@router.post("/setup/step/3")
async def setup_step3_post(request: Request):
    """Save optional keys."""
    session_id, session = _get_session(request)
    form = await request.form()

    telegram_token = (form.get("telegram_token") or "").strip()
    elevenlabs_key = (form.get("elevenlabs_key") or "").strip()

    keys = session.get("keys", {})
    if telegram_token:
        keys["TELEGRAM_BOT_TOKEN"] = telegram_token
    if elevenlabs_key:
        keys["ELEVENLABS_API_KEY"] = elevenlabs_key
    session["keys"] = keys

    session["telegram_enabled"] = bool(telegram_token)
    session["elevenlabs_enabled"] = bool(elevenlabs_key)

    response = RedirectResponse(url="/setup/step/4", status_code=303)
    return _set_cookie(response, session_id)


@router.get("/setup/step/4")
async def setup_step4_get(request: Request):
    """Step 4: Topic selection."""
    templates = get_templates(request)
    session_id, session = _get_session(request)
    response = templates.TemplateResponse(request, "setup/step4_topics.html", {
        "topic_choices": TOPIC_CHOICES,
        "selected_topics": session.get("topics", []),
        "step": 4,
        "total_steps": 6,
    })
    return _set_cookie(response, session_id)


@router.post("/setup/step/4")
async def setup_step4_post(request: Request):
    """Save selected topics."""
    session_id, session = _get_session(request)
    form = await request.form()

    selected = form.getlist("topics")
    custom_topic = (form.get("custom_topic") or "").strip()

    topics = []
    for slug in selected:
        for s, name in TOPIC_CHOICES:
            if s == slug:
                topics.append({"name": name, "priority": "high"})
                break

    if custom_topic:
        topics.append({"name": custom_topic, "priority": "medium"})

    if not topics:
        # Default to first topic
        topics = [{"name": TOPIC_CHOICES[0][1], "priority": "high"}]

    session["topics"] = topics

    response = RedirectResponse(url="/setup/step/5", status_code=303)
    return _set_cookie(response, session_id)


@router.get("/setup/step/5")
async def setup_step5_get(request: Request):
    """Step 5: User preferences."""
    templates = get_templates(request)
    session_id, session = _get_session(request)
    response = templates.TemplateResponse(request, "setup/step5_preferences.html", {
        "user_name": session.get("user_name", ""),
        "timezone": session.get("timezone", ""),
        "schedule": session.get("schedule", "06:00"),
        "style": session.get("style", "analytical"),
        "step": 5,
        "total_steps": 6,
    })
    return _set_cookie(response, session_id)


@router.post("/setup/step/5")
async def setup_step5_post(
    request: Request,
    user_name: str = Form(""),
    timezone: str = Form("UTC"),
    schedule: str = Form("06:00"),
    style: str = Form("analytical"),
):
    """Save user preferences."""
    session_id, session = _get_session(request)
    session["user_name"] = user_name.strip() or "User"
    session["timezone"] = timezone.strip() or "UTC"
    session["schedule"] = schedule.strip() or "06:00"
    session["style"] = style.strip()

    response = RedirectResponse(url="/setup/step/6", status_code=303)
    return _set_cookie(response, session_id)


@router.get("/setup/step/6")
async def setup_step6_get(request: Request):
    """Step 6: Review all choices."""
    templates = get_templates(request)
    session_id, session = _get_session(request)

    # Find preset display name
    preset_label = session.get("preset", "")
    for name, label, desc, _ in PRESET_INFO:
        if name == preset_label:
            preset_label = label
            break

    response = templates.TemplateResponse(request, "setup/step6_review.html", {
        "session": session,
        "preset_label": preset_label,
        "step": 6,
        "total_steps": 6,
    })
    return _set_cookie(response, session_id)


@router.post("/setup/complete")
async def setup_complete(request: Request):
    """Write config.yaml + .env, redirect to dashboard."""
    session_id, session = _get_session(request)
    data_dir = _data_dir(request)

    preset = session.get("preset", "balanced")
    is_free = preset == "free"

    # Determine audio config
    audio_config = {"enabled": not is_free}
    if session.get("elevenlabs_enabled"):
        audio_config["tts_backend"] = "elevenlabs"
        audio_config["enabled"] = True

    config_dict = {
        "preset": preset,
        "user": {
            "name": session.get("user_name", "User"),
            "timezone": session.get("timezone", "UTC"),
            "output_language": "en",
        },
        "topics": session.get("topics", [{"name": "AI/ML Research", "priority": "high"}]),
        "briefing": {
            "schedule": session.get("schedule", "06:00"),
            "style": session.get("style", "analytical"),
        },
        "audio": audio_config,
        "breaking_news": {"enabled": True, "threshold": 7},
        "telegram": {"enabled": session.get("telegram_enabled", False)},
    }

    write_config(data_dir, config_dict)

    # Write API keys to .env
    keys = session.get("keys", {})
    if keys:
        write_env(data_dir.parent, keys)

    # Clean up session
    sessions = getattr(request.app.state, "setup_sessions", {})
    sessions.pop(session_id, None)

    response = RedirectResponse(url="/", status_code=303)
    response.delete_cookie("nexus_setup")
    return response


@router.post("/setup/launch")
async def setup_launch(request: Request):
    """Start the pipeline in a background task."""
    import os
    from dotenv import load_dotenv

    data_dir = _data_dir(request)
    load_dotenv(data_dir.parent / ".env")

    # Set status
    request.app.state.pipeline_status = {"stage": "starting", "done": False}

    async def _run_pipeline():
        try:
            from nexus.config.loader import load_config
            from nexus.llm.client import LLMClient
            from nexus.engine.pipeline import run_pipeline

            status = request.app.state.pipeline_status

            config_path = data_dir / "config.yaml"
            config = load_config(config_path)

            api_key = os.getenv("GEMINI_API_KEY")
            anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
            deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
            openai_api_key = os.getenv("OPENAI_API_KEY")
            elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")

            llm = LLMClient(
                config.models,
                api_key=api_key,
                anthropic_api_key=anthropic_api_key,
                deepseek_api_key=deepseek_api_key,
                openai_api_key=openai_api_key,
                budget_config=config.budget,
            )

            status["stage"] = "running"
            await run_pipeline(
                config, llm, data_dir,
                gemini_api_key=api_key,
                openai_api_key=openai_api_key,
                elevenlabs_api_key=elevenlabs_api_key,
            )
            status["stage"] = "complete"
            status["done"] = True
        except Exception as e:
            logger.error(f"Background pipeline failed: {e}", exc_info=True)
            request.app.state.pipeline_status = {
                "stage": "error",
                "done": True,
                "error": str(e),
            }

    asyncio.create_task(_run_pipeline())
    return HTMLResponse('<div class="pipeline-status" id="pipeline-bar">'
                        'Pipeline started...</div>')


@router.get("/setup/status")
async def setup_status(request: Request):
    """Return pipeline progress as HTML fragment for HTMX polling."""
    status = getattr(request.app.state, "pipeline_status", None)
    if not status:
        return HTMLResponse("")

    stage = status.get("stage", "unknown")
    done = status.get("done", False)
    error = status.get("error")

    if error:
        return HTMLResponse(
            f'<div class="pipeline-status pipeline-error" id="pipeline-bar">'
            f'Pipeline error: {error}</div>'
        )
    if done:
        return HTMLResponse(
            '<div class="pipeline-status pipeline-done" id="pipeline-bar">'
            'Pipeline complete! <a href="/" onclick="location.reload()">View your first briefing</a>'
            '</div>'
        )

    stage_labels = {
        "starting": "Initializing pipeline...",
        "running": "Running pipeline (this may take a few minutes)...",
    }
    label = stage_labels.get(stage, f"Pipeline: {stage}...")

    return HTMLResponse(
        f'<div class="pipeline-status pipeline-running" id="pipeline-bar" '
        f'hx-get="/setup/status" hx-trigger="every 3s" hx-swap="outerHTML">'
        f'<div class="pipeline-spinner"></div> {label}</div>'
    )
