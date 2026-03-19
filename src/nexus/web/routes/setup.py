"""Web-based setup wizard — streamlined 3-step form for first-run configuration.

Steps:
  1. Provider + API key (combined)
  2. Topics
  3. Review → auto-launches pipeline on submit
"""

import asyncio
import logging
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import RedirectResponse, HTMLResponse

from nexus.cli.setup import PROVIDER_INFO, PROVIDER_TIERS, TOPIC_CHOICES
from nexus.config.writer import build_initial_config, write_config, write_env
from nexus.utils.runtime_env import load_runtime_env, runtime_env_path
from nexus.web.app import get_templates

logger = logging.getLogger(__name__)

router = APIRouter()

TOTAL_STEPS = 3


def _track_background_task(request: Request, task: asyncio.Task) -> asyncio.Task:
    """Keep a strong reference to background tasks until they finish."""
    tasks = getattr(request.app.state, "background_tasks", None)
    if tasks is None:
        request.app.state.background_tasks = set()
        tasks = request.app.state.background_tasks
    tasks.add(task)
    task.add_done_callback(tasks.discard)
    return task


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


# ── Step 1: Provider + API Key (combined) ──


@router.get("/setup")
async def setup_start(request: Request):
    """Step 1: Provider selection + API key input (combined)."""
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
        "total_steps": TOTAL_STEPS,
    })
    return _set_cookie(response, session_id)


@router.post("/setup/step/1")
async def setup_step1(request: Request):
    """Save provider + preset + API key, advance to step 2 (topics)."""
    import json
    session_id, session = _get_session(request)
    form = await request.form()
    provider = (form.get("provider") or "").strip()
    preset = (form.get("preset") or "").strip()
    api_key = (form.get("api_key") or "").strip()

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
                "total_steps": TOTAL_STEPS,
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

    # Handle API key
    if required_key:
        # If key is empty but already set in session, keep it
        if not api_key and required_key in session.get("keys", {}):
            pass  # keep existing key
        elif not api_key:
            templates = get_templates(request)
            response = templates.TemplateResponse(
                request,
                "setup/step1_welcome.html",
                {
                    "providers": PROVIDER_INFO,
                    "tiers_json": json.dumps(PROVIDER_TIERS),
                    "selected_provider": provider,
                    "selected_preset": preset,
                    "error": "API key is required for this provider.",
                    "step": 1,
                    "total_steps": TOTAL_STEPS,
                },
                status_code=400,
            )
            return _set_cookie(response, session_id)
        else:
            session.setdefault("keys", {})[required_key] = api_key
    else:
        session["keys"] = {}

    # Skip straight to topics (step 2)
    response = RedirectResponse(url="/setup/step/2", status_code=303)
    return _set_cookie(response, session_id)


# ── Step 2: Topics ──


@router.get("/setup/step/2")
async def setup_step2_get(request: Request):
    """Step 2: Topic selection."""
    templates = get_templates(request)
    session_id, session = _get_session(request)
    error = session.pop("_error", None)
    response = templates.TemplateResponse(request, "setup/step2_topics.html", {
        "topic_choices": TOPIC_CHOICES,
        "selected_topics": session.get("topics", []),
        "step": 2,
        "total_steps": TOTAL_STEPS,
        "error": error,
    })
    return _set_cookie(response, session_id)


@router.post("/setup/step/2")
async def setup_step2_post(request: Request):
    """Save selected topics, advance to review."""
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
        session["_error"] = "Please select at least one topic."
        response = RedirectResponse(url="/setup/step/2", status_code=303)
        return _set_cookie(response, session_id)

    session["topics"] = topics

    response = RedirectResponse(url="/setup/step/3", status_code=303)
    return _set_cookie(response, session_id)


# ── Step 3: Review + auto-launch ──


@router.get("/setup/step/3")
async def setup_step3_get(request: Request):
    """Step 3: Review all choices."""
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

    response = templates.TemplateResponse(request, "setup/step3_review.html", {
        "session": session,
        "provider_label": provider_label,
        "tier_label": tier_label,
        "step": 3,
        "total_steps": TOTAL_STEPS,
    })
    return _set_cookie(response, session_id)


@router.post("/setup/complete")
async def setup_complete(request: Request):
    """Write config + .env, auto-launch pipeline, redirect to dashboard."""
    session_id, session = _get_session(request)
    data_dir = _data_dir(request)

    preset = session.get("preset", "balanced")
    is_free = preset == "free"

    # Determine audio config
    audio_config = {"enabled": not is_free}

    config_dict = build_initial_config(
        preset=preset,
        topics=session.get("topics", [{"name": "AI/ML Research", "priority": "high"}]),
    )
    config_dict["audio"] = audio_config

    write_config(data_dir, config_dict)

    # Write API keys to .env
    keys = session.get("keys", {})
    if keys:
        env_path = write_env(runtime_env_path(data_dir).parent, keys)
        load_runtime_env(env_path)

    # Clean up session
    sessions = getattr(request.app.state, "setup_sessions", {})
    sessions.pop(session_id, None)

    # Auto-launch pipeline in background
    _auto_launch_pipeline(request, data_dir)

    response = RedirectResponse(url="/?setup=complete", status_code=303)
    response.delete_cookie("nexus_setup")
    return response


def _auto_launch_pipeline(request: Request, data_dir: Path):
    """Kick off the pipeline immediately after setup completes."""
    import os

    # Testing gate: skip real pipeline in E2E tests
    if os.environ.get("NEXUS_SKIP_AUTO_PIPELINE"):
        request.app.state.pipeline_status = {"stage": "complete", "done": True}
        return

    existing = getattr(request.app.state, "pipeline_status", None)
    if existing and not existing.get("done", True):
        return  # already running

    config_path = data_dir / "config.yaml"
    if not config_path.exists():
        return

    env_path = runtime_env_path(data_dir)
    load_runtime_env(env_path)

    # Set status with start time for elapsed display
    request.app.state.pipeline_status = {
        "stage": "starting",
        "done": False,
        "started_at": time.time(),
        "topic_index": 0,
        "topic_count": 0,
    }

    async def _run_pipeline():
        try:
            from nexus.config.loader import load_config
            from nexus.llm.client import LLMClient
            from nexus.engine.pipeline import run_pipeline, run_backfill

            status = request.app.state.pipeline_status

            config = load_config(config_path)
            status["topic_count"] = len(config.topics)

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
            for i, topic_cfg in enumerate(config.topics):
                slug = topic_cfg.name.lower().replace(" ", "-").replace("/", "-")
                registry_path = data_dir / "sources" / slug / "registry.yaml"
                if not registry_path.exists():
                    status["stage"] = f"discovering:{topic_cfg.name}"
                    status["topic_index"] = i + 1
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
            # Pass status dict so pipeline can update per-topic progress
            smoke_cap = int(os.getenv("NEXUS_SMOKE_MODE", "0"))
            briefing_path = await run_pipeline(
                config, llm, data_dir,
                gemini_api_key=api_key,
                openai_api_key=openai_api_key,
                elevenlabs_api_key=elevenlabs_api_key,
                max_ingest=smoke_cap or None,
                trigger="setup",
                progress_status=status,
            )
            status["stage"] = "complete"
            status["done"] = True
            request.app.state.has_data = True

            # Deliver via Telegram if configured
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
                    status["telegram_delivered"] = True
                except Exception as tg_err:
                    logger.warning(f"Telegram delivery failed (non-blocking): {tg_err}")
                    status["telegram_delivered"] = False
                    status["telegram_error"] = str(tg_err)
            elif config.telegram.enabled and not config.telegram.chat_id:
                status["telegram_delivered"] = False
                status["telegram_error"] = "No chat_id — send /start to your bot"

            # Phase 2: Backfill historical events (7 days) in the background.
            # The user already has their quick briefing — now silently enrich
            # the knowledge store so subsequent runs have richer context.
            status["stage"] = "backfill"
            try:
                backfill_count = await run_backfill(
                    config, llm, data_dir,
                    max_age_hours=168,  # 7 days
                    progress_status=status,
                )
                status["backfill_events"] = backfill_count
                logger.info(f"Post-setup backfill added {backfill_count} historical events")
            except Exception as bf_err:
                logger.warning(f"Post-setup backfill failed (non-blocking): {bf_err}")
            finally:
                status["stage"] = "complete"
        except Exception as e:
            logger.error(f"Background pipeline failed: {e}", exc_info=True)
            request.app.state.pipeline_status = {
                "stage": "error",
                "done": True,
                "error": str(e),
                "started_at": status.get("started_at"),
            }

    _track_background_task(request, asyncio.create_task(_run_pipeline()))


# ── Pipeline launch (manual trigger from dashboard empty state) ──


@router.post("/setup/launch")
async def setup_launch(request: Request):
    """Start the pipeline in a background task (manual trigger)."""
    import os

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

    load_runtime_env(runtime_env_path(data_dir))

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

    _auto_launch_pipeline(request, data_dir)

    return HTMLResponse(
        '<div class="pipeline-status pipeline-running" id="pipeline-bar" '
        'hx-get="/setup/status" hx-trigger="every 3s" hx-swap="outerHTML">'
        '<div class="pipeline-spinner"></div> Pipeline started...</div>'
    )


# ── Pipeline status (HTMX polling) ──


@router.get("/setup/status")
async def setup_status(request: Request):
    """Return pipeline progress as HTML fragment for HTMX polling."""
    status = getattr(request.app.state, "pipeline_status", None)
    if not status:
        return HTMLResponse('<div id="pipeline-bar" style="margin-top: 1rem;"></div>')

    stage = status.get("stage", "unknown")
    done = status.get("done", False)
    error = status.get("error")

    # Elapsed time
    started_at = status.get("started_at")
    elapsed_html = ""
    if started_at:
        elapsed_s = int(time.time() - started_at)
        mins, secs = divmod(elapsed_s, 60)
        elapsed_html = f' <span class="pipeline-elapsed">{mins}:{secs:02d}</span>'

    if error:
        from html import escape
        return HTMLResponse(
            f'<div class="pipeline-status pipeline-error" id="pipeline-bar">'
            f'Pipeline error: {escape(str(error))}'
            f'{elapsed_html}</div>'
        )
    if done:
        tg_note = ""
        if status.get("telegram_delivered"):
            tg_note = " Briefing delivered to Telegram."
        elif "telegram_error" in status:
            from html import escape as _esc
            tg_note = f' <span style="color:var(--warning)">Telegram: {_esc(status["telegram_error"])}</span>'
        return HTMLResponse(
            f'<div class="pipeline-status pipeline-done" id="pipeline-bar">'
            f'Pipeline complete!{tg_note}{elapsed_html} '
            f'<a href="/" onclick="location.reload()">View your first briefing</a>'
            f'</div>'
        )

    # Build label from stage with per-topic granularity
    topic_index = status.get("topic_index", 0)
    topic_count = status.get("topic_count", 0)
    topic_progress = f" ({topic_index}/{topic_count})" if topic_count else ""

    if stage.startswith("discovering:"):
        topic_name = stage.split(":", 1)[1]
        label = f"Discovering sources for {topic_name}{topic_progress}..."
    elif stage.startswith("topic:"):
        # Per-topic pipeline stage: "topic:TopicName:stage_name"
        parts = stage.split(":", 2)
        topic_name = parts[1] if len(parts) > 1 else "?"
        sub_stage = parts[2] if len(parts) > 2 else "processing"
        sub_labels = {
            "polling": "Polling feeds",
            "filtering": "Filtering articles",
            "events": "Extracting events",
            "entities": "Resolving entities",
            "synthesis": "Synthesizing intelligence",
        }
        sub_label = sub_labels.get(sub_stage, sub_stage.title())
        label = f"{sub_label} &mdash; {topic_name}{topic_progress}..."
    elif stage == "rendering":
        label = "Rendering briefing..."
    elif stage == "audio":
        label = "Generating audio podcast..."
    else:
        stage_labels = {
            "starting": "Initializing pipeline...",
            "discovering": "Discovering sources for new topics...",
            "running": "Running pipeline...",
            "projections": "Computing predictions...",
        }
        label = stage_labels.get(stage, f"Pipeline: {stage}...")

    return HTMLResponse(
        f'<div class="pipeline-status pipeline-running" id="pipeline-bar" '
        f'hx-get="/setup/status" hx-trigger="every 3s" hx-swap="outerHTML">'
        f'<div class="pipeline-spinner"></div> {label}{elapsed_html}</div>'
    )


# ── Telegram validation (still available but moved out of setup flow) ──


@router.post("/setup/telegram/validate")
async def setup_telegram_validate(request: Request):
    """Validate Telegram bot token and show bot info (HTMX fragment)."""
    from nexus.agent.telegram_utils import validate_token
    from html import escape

    session_id, session = _get_session(request)
    form = await request.form()
    token = (form.get("telegram_token") or "").strip()

    if not token:
        token = session.get("keys", {}).get("TELEGRAM_BOT_TOKEN", "")

    if not token:
        resp = HTMLResponse(
            '<div id="telegram-status" class="form-hint" style="color:var(--danger)">'
            'Enter a bot token first.</div>'
        )
        return _set_cookie(resp, session_id)

    bot_info = await validate_token(token)
    if bot_info:
        username = escape(bot_info.get("username", ""))
        session.setdefault("keys", {})["TELEGRAM_BOT_TOKEN"] = token
        session["telegram_enabled"] = True
        session["telegram_bot_username"] = username
        resp = HTMLResponse(
            f'<div id="telegram-status">'
            f'<div class="form-hint" style="color:var(--success)">Token valid! Bot: @{username}</div>'
            f'<div style="margin-top:0.5rem">'
            f'<span class="form-hint">Now open Telegram and send <code>/start</code> to '
            f'<a href="https://t.me/{username}" target="_blank">@{username}</a></span>'
            f'</div>'
            f'<div id="telegram-poll" hx-get="/setup/telegram/poll" hx-trigger="every 3s" '
            f'hx-swap="outerHTML" style="margin-top:0.5rem">'
            f'<span class="form-hint">Waiting for /start...</span>'
            f'</div></div>'
        )
    else:
        resp = HTMLResponse(
            '<div id="telegram-status" class="form-hint" style="color:var(--danger)">'
            'Invalid token. Check with <a href="https://t.me/BotFather" target="_blank">@BotFather</a>.'
            '</div>'
        )
    return _set_cookie(resp, session_id)


@router.get("/setup/telegram/poll")
async def setup_telegram_poll(request: Request):
    """Poll for /start message to capture chat_id (HTMX fragment)."""
    from nexus.agent.telegram_utils import poll_for_chat_id

    session_id, session = _get_session(request)
    token = session.get("keys", {}).get("TELEGRAM_BOT_TOKEN", "")

    if not token:
        resp = HTMLResponse('<div id="telegram-poll"></div>')
        return _set_cookie(resp, session_id)

    if session.get("telegram_chat_id"):
        chat_id = session["telegram_chat_id"]
        resp = HTMLResponse(
            f'<div id="telegram-poll" class="form-hint" style="color:var(--success)">'
            f'Connected! Chat ID: {chat_id}</div>'
        )
        return _set_cookie(resp, session_id)

    chat_id = await poll_for_chat_id(token, timeout=3.0)
    if chat_id:
        session["telegram_chat_id"] = chat_id
        resp = HTMLResponse(
            f'<div id="telegram-poll" class="form-hint" style="color:var(--success)">'
            f'Connected! Chat ID: {chat_id}</div>'
        )
    else:
        resp = HTMLResponse(
            '<div id="telegram-poll" hx-get="/setup/telegram/poll" hx-trigger="every 3s" '
            'hx-swap="outerHTML">'
            '<span class="form-hint">Waiting for /start...</span>'
            '</div>'
        )
    return _set_cookie(resp, session_id)
