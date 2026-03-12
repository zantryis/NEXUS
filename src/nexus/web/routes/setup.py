"""Web-based setup wizard — multi-step form for first-run configuration."""

import asyncio
import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, Request, Form
from fastapi.responses import RedirectResponse, HTMLResponse

from nexus.cli.setup import PROVIDER_INFO, PROVIDER_TIERS, TOPIC_CHOICES
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
    """Step 1: Welcome + provider selection."""
    import json
    templates = get_templates(request)
    session_id, session = _get_session(request)
    response = templates.TemplateResponse(request, "setup/step1_welcome.html", {
        "providers": PROVIDER_INFO,
        "tiers_json": json.dumps(PROVIDER_TIERS),
        "selected_provider": session.get("provider"),
        "selected_preset": session.get("preset"),
        "error": None,
        "step": 1,
        "total_steps": 6,
    })
    return _set_cookie(response, session_id)


@router.post("/setup/step/1")
async def setup_step1(request: Request):
    """Save provider + preset, advance to step 2."""
    import json
    session_id, session = _get_session(request)
    form = await request.form()
    provider = (form.get("provider") or "").strip()
    preset = (form.get("preset") or "").strip()

    valid_providers = {p[0] for p in PROVIDER_INFO}
    if provider not in valid_providers:
        templates = get_templates(request)
        response = templates.TemplateResponse(
            request,
            "setup/step1_welcome.html",
            {
                "providers": PROVIDER_INFO,
                "tiers_json": json.dumps(PROVIDER_TIERS),
                "selected_provider": session.get("provider"),
                "selected_preset": session.get("preset"),
                "error": "Please choose a provider.",
                "step": 1,
                "total_steps": 6,
            },
            status_code=400,
        )
        return _set_cookie(response, session_id)

    session["provider"] = provider

    # Resolve preset from tier selection or default to first tier
    tiers = PROVIDER_TIERS.get(provider, [])
    valid_presets = {t[0] for t in tiers}
    if preset not in valid_presets:
        preset = tiers[0][0] if tiers else "balanced"
    session["preset"] = preset

    # Find required key for this provider
    required_key = None
    for pid, _, _, key in PROVIDER_INFO:
        if pid == provider:
            required_key = key
            break
    session["required_key"] = required_key

    # If no API key needed (ollama), skip key step
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
    required_key = session.get("required_key", "")
    key_already_set = required_key and required_key in session.get("keys", {})

    response = templates.TemplateResponse(request, "setup/step2_keys.html", {
        "required_key": required_key,
        "provider": session.get("provider", ""),
        "key_already_set": key_already_set,
        "error": None,
        "step": 2,
        "total_steps": 6,
    })
    return _set_cookie(response, session_id)


@router.post("/setup/step/2")
async def setup_step2_post(request: Request):
    """Validate and save required API key."""
    session_id, session = _get_session(request)
    form = await request.form()
    api_key = (form.get("api_key") or "").strip()
    required_key = session.get("required_key", "")

    # If key is empty but already set in session, keep it and proceed
    if not api_key and required_key in session.get("keys", {}):
        response = RedirectResponse(url="/setup/step/3", status_code=303)
        return _set_cookie(response, session_id)

    if not api_key:
        templates = get_templates(request)
        response = templates.TemplateResponse(request, "setup/step2_keys.html", {
            "required_key": required_key,
            "provider": session.get("provider", ""),
            "key_already_set": False,
            "error": "API key cannot be empty.",
            "step": 2,
            "total_steps": 6,
        })
        return _set_cookie(response, session_id)

    session.setdefault("keys", {})[required_key] = api_key
    response = RedirectResponse(url="/setup/step/3", status_code=303)
    return _set_cookie(response, session_id)


@router.get("/setup/step/3")
async def setup_step3_get(request: Request):
    """Step 3: Optional keys (Telegram, TTS)."""
    templates = get_templates(request)
    session_id, session = _get_session(request)
    keys = session.get("keys", {})
    response = templates.TemplateResponse(request, "setup/step3_optional.html", {
        "preset": session.get("preset", ""),
        "telegram_set": "TELEGRAM_BOT_TOKEN" in keys,
        "elevenlabs_set": "ELEVENLABS_API_KEY" in keys,
        "has_key_step": bool(session.get("required_key")),
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

    session["telegram_enabled"] = bool(telegram_token) or "TELEGRAM_BOT_TOKEN" in keys
    session["elevenlabs_enabled"] = bool(elevenlabs_key) or "ELEVENLABS_API_KEY" in keys

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
    custom_topics = [t.strip() for t in form.getlist("custom_topics") if t.strip()]

    topics = []
    for slug in selected:
        for s, name in TOPIC_CHOICES:
            if s == slug:
                topics.append({"name": name, "priority": "high"})
                break

    for ct in custom_topics:
        topics.append({"name": ct, "priority": "medium"})

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

    # Find provider and tier display names
    provider_label = session.get("provider", "")
    for pid, label, _, _ in PROVIDER_INFO:
        if pid == provider_label:
            provider_label = label
            break

    tier_label = ""
    provider_id = session.get("provider", "")
    preset_name = session.get("preset", "")
    for tier_preset, tier_lbl, tier_cost in PROVIDER_TIERS.get(provider_id, []):
        if tier_preset == preset_name:
            tier_label = f"{tier_lbl} ({tier_cost})"
            break

    response = templates.TemplateResponse(request, "setup/step6_review.html", {
        "session": session,
        "provider_label": provider_label,
        "tier_label": tier_label,
        "step": 6,
        "total_steps": 6,
    })
    return _set_cookie(response, session_id)


@router.post("/setup/complete")
async def setup_complete(request: Request):
    """Write config.yaml + .env, redirect to dashboard with restart guidance."""
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

    response = RedirectResponse(url="/?setup=complete", status_code=303)
    response.delete_cookie("nexus_setup")
    return response


@router.post("/setup/launch")
async def setup_launch(request: Request):
    """Start the pipeline in a background task."""
    import os
    from dotenv import load_dotenv

    data_dir = _data_dir(request)

    # Guard: already running?
    existing = getattr(request.app.state, "pipeline_status", None)
    if existing and not existing.get("done", True):
        return HTMLResponse(
            '<div class="pipeline-status pipeline-running" id="pipeline-bar" '
            'hx-get="/setup/status" hx-trigger="every 3s" hx-swap="outerHTML">'
            '<div class="pipeline-spinner"></div> Pipeline is already running...</div>'
        )

    # Validate config exists
    config_path = data_dir / "config.yaml"
    if not config_path.exists():
        return HTMLResponse(
            '<div class="pipeline-status pipeline-error" id="pipeline-bar">'
            'Error: No config found. Please complete setup first.</div>'
        )

    load_dotenv(data_dir.parent / ".env")

    # Validate required API key before launching
    from nexus.config.loader import load_config as _load_config
    try:
        _cfg = _load_config(config_path)
    except Exception as e:
        return HTMLResponse(
            f'<div class="pipeline-status pipeline-error" id="pipeline-bar">'
            f'Error loading config: {str(e)}</div>'
        )

    # Determine required API key from preset
    preset_name = getattr(_cfg, "preset", None) or "balanced"
    _preset_key_map = {
        "free": None,
        "cheap": "DEEPSEEK_API_KEY",
        "balanced": "GEMINI_API_KEY",
        "quality": "GEMINI_API_KEY",
        "openai-cheap": "OPENAI_API_KEY",
        "openai-balanced": "OPENAI_API_KEY",
        "openai-quality": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
    }
    required_key = _preset_key_map.get(preset_name)
    if required_key and not os.getenv(required_key):
        return HTMLResponse(
            f'<div class="pipeline-status pipeline-error" id="pipeline-bar">'
            f'Missing API key: <strong>{required_key}</strong> is required for the '
            f'"{preset_name}" preset. Set it in Settings or your .env file, then try again.</div>'
        )

    # Set status
    request.app.state.pipeline_status = {"stage": "starting", "done": False}

    async def _run_pipeline():
        try:
            from nexus.config.loader import load_config
            from nexus.llm.client import LLMClient
            from nexus.engine.pipeline import run_pipeline

            status = request.app.state.pipeline_status

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

            # Auto-discover sources for topics without registries
            import yaml as _yaml
            from nexus.engine.sources.discovery import discover_sources
            for topic_cfg in config.topics:
                slug = topic_cfg.name.lower().replace(" ", "-").replace("/", "-")
                registry_path = data_dir / "sources" / slug / "registry.yaml"
                if not registry_path.exists():
                    status["stage"] = f"discovering:{topic_cfg.name}"
                    try:
                        result = await discover_sources(
                            llm, topic_cfg.name,
                            subtopics=topic_cfg.subtopics,
                            data_dir=data_dir,
                        )
                        if result.feeds:
                            registry_path.parent.mkdir(parents=True, exist_ok=True)
                            registry_path.write_text(
                                _yaml.dump({"sources": result.feeds}, default_flow_style=False)
                            )
                            logger.info(f"Discovered {len(result.feeds)} sources for {topic_cfg.name}")
                    except Exception as disc_err:
                        logger.warning(f"Source discovery failed for {topic_cfg.name}: {disc_err}")

            status["stage"] = "running"
            # NEXUS_SMOKE_MODE caps ingestion for fast testing
            smoke_cap = int(os.getenv("NEXUS_SMOKE_MODE", "0"))
            briefing_path = await run_pipeline(
                config, llm, data_dir,
                gemini_api_key=api_key,
                openai_api_key=openai_api_key,
                elevenlabs_api_key=elevenlabs_api_key,
                max_ingest=smoke_cap or None,
            )
            status["stage"] = "complete"
            status["done"] = True
            # Mark that data now exists
            request.app.state.has_data = True

            # Deliver via Telegram if token and chat_id are available
            telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
            if (
                config.telegram.enabled
                and telegram_token
                and config.telegram.chat_id
                and briefing_path
                and briefing_path.exists()
            ):
                try:
                    from telegram import Bot
                    from nexus.agent.delivery import deliver_briefing
                    from datetime import date as _date

                    tg_bot = Bot(token=telegram_token)
                    today_str = _date.today().isoformat()
                    text = briefing_path.read_text()
                    audio_path = data_dir / "artifacts" / "audio" / f"{today_str}.mp3"
                    audio = audio_path if audio_path.exists() else None

                    async with tg_bot:
                        await deliver_briefing(tg_bot, config.telegram.chat_id, text, audio)
                    logger.info("Briefing delivered via Telegram")
                except Exception as tg_err:
                    logger.warning(f"Telegram delivery failed (non-blocking): {tg_err}")
        except Exception as e:
            logger.error(f"Background pipeline failed: {e}", exc_info=True)
            request.app.state.pipeline_status = {
                "stage": "error",
                "done": True,
                "error": str(e),
            }

    asyncio.create_task(_run_pipeline())
    return HTMLResponse(
        '<div class="pipeline-status pipeline-running" id="pipeline-bar" '
        'hx-get="/setup/status" hx-trigger="every 3s" hx-swap="outerHTML">'
        '<div class="pipeline-spinner"></div> Pipeline started...</div>'
    )


@router.get("/setup/status")
async def setup_status(request: Request):
    """Return pipeline progress as HTML fragment for HTMX polling."""
    status = getattr(request.app.state, "pipeline_status", None)
    if not status:
        return HTMLResponse('<div id="pipeline-bar" style="margin-top: 1rem;"></div>')

    stage = status.get("stage", "unknown")
    done = status.get("done", False)
    error = status.get("error")

    if error:
        from html import escape
        return HTMLResponse(
            f'<div class="pipeline-status pipeline-error" id="pipeline-bar">'
            f'Pipeline error: {escape(str(error))}</div>'
        )
    if done:
        return HTMLResponse(
            '<div class="pipeline-status pipeline-done" id="pipeline-bar">'
            'Pipeline complete! <a href="/" onclick="location.reload()">View your first briefing</a>'
            '</div>'
        )

    # Build label from stage
    if stage.startswith("discovering:"):
        topic_name = stage.split(":", 1)[1]
        label = f"Discovering sources for {topic_name}..."
    else:
        stage_labels = {
            "starting": "Initializing pipeline...",
            "discovering": "Discovering sources for new topics...",
            "running": "Running pipeline (this may take a few minutes)...",
        }
        label = stage_labels.get(stage, f"Pipeline: {stage}...")

    return HTMLResponse(
        f'<div class="pipeline-status pipeline-running" id="pipeline-bar" '
        f'hx-get="/setup/status" hx-trigger="every 3s" hx-swap="outerHTML">'
        f'<div class="pipeline-spinner"></div> {label}</div>'
    )
