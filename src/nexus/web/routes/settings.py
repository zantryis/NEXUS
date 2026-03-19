"""Settings page — view and modify API keys, config, and feature toggles."""

import json
import os
import sys
import logging
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import RedirectResponse
from starlette.background import BackgroundTask

from nexus.config.presets import preset_names
from nexus.config.writer import write_config, write_env
from nexus.utils.runtime_env import build_runtime_env, load_runtime_env, runtime_env_path
from nexus.web.app import get_templates

logger = logging.getLogger(__name__)

router = APIRouter()

TTS_VOICES = {
    "gemini": ["Kore", "Puck", "Charon", "Fenrir", "Aoede", "Leda", "Orus", "Zephyr"],
    "elevenlabs": [
        "Sarah", "Laura", "Charlie", "George", "Callum",
        "River", "Lily", "Bill", "Will", "Jessica", "Aria",
        "Rachel", "Drew", "Clyde", "Domi", "Bella", "Josh", "Adam",
    ],
    "openai": ["alloy", "echo", "fable", "onyx", "nova", "shimmer"],
}

PROVIDERS = [
    {"key": "GEMINI_API_KEY", "name": "Gemini (Google)", "prefix": "gemini"},
    {"key": "OPENAI_API_KEY", "name": "OpenAI", "prefix": "openai"},
    {"key": "ANTHROPIC_API_KEY", "name": "Anthropic (Claude)", "prefix": "claude"},
    {"key": "DEEPSEEK_API_KEY", "name": "DeepSeek", "prefix": "deepseek"},
    {"key": "ELEVENLABS_API_KEY", "name": "ElevenLabs (TTS)", "prefix": "elevenlabs"},
    {"key": "TELEGRAM_BOT_TOKEN", "name": "Telegram Bot", "prefix": "telegram"},
]


RESTART_REQUIRED_FIELDS = {"preset", "tts_backend", "telegram_enabled"}

PODCAST_STYLES = ["conversational", "analytical", "energetic", "formal"]


def validate_settings(timezone: str = "", schedule: str = "") -> dict[str, str]:
    """Validate settings values. Returns {field: error_message} for invalid fields."""
    errors: dict[str, str] = {}

    # Timezone
    if timezone:
        from zoneinfo import available_timezones
        if timezone not in available_timezones():
            errors["timezone"] = f'Unknown timezone "{timezone}". Use a valid IANA timezone.'

    # Schedule format: HH:MM
    if schedule:
        import re
        m = re.match(r"^(\d{1,2}):(\d{2})$", schedule)
        if not m:
            errors["schedule"] = 'Schedule must be in HH:MM format (e.g., "06:00").'
        else:
            h, mn = int(m.group(1)), int(m.group(2))
            if h > 23 or mn > 59:
                errors["schedule"] = f'Invalid time "{schedule}". Hours 0-23, minutes 0-59.'

    return errors


def _data_dir(request: Request) -> Path:
    return getattr(request.app.state, "data_dir", Path("data"))


def _get_config_dict(request: Request) -> dict:
    """Load current config as raw dict for form editing."""
    import yaml
    config_path = _data_dir(request) / "config.yaml"
    if config_path.exists():
        return yaml.safe_load(config_path.read_text()) or {}
    return {}


def _provider_status() -> list[dict]:
    """Build provider status list with masked keys."""
    status = []
    for p in PROVIDERS:
        val = os.getenv(p["key"], "")
        status.append({
            "key": p["key"],
            "name": p["name"],
            "is_set": bool(val),
            "masked": f"...{val[-4:]}" if len(val) > 4 else ("***" if val else "not set"),
        })
    return status


@router.get("/settings")
async def settings_page(request: Request):
    """Settings page showing all config sections."""
    templates = get_templates(request)

    config = getattr(request.app.state, "config", None)
    raw = _get_config_dict(request)

    saved = request.query_params.get("saved")
    error = request.query_params.get("error")

    # Experimental OpenAI OAuth status
    from nexus.llm.oauth import OpenAIOAuthManager
    data_dir = _data_dir(request)
    oauth_mgr = OpenAIOAuthManager(token_path=data_dir / ".oauth-tokens.json")
    oauth_tokens = oauth_mgr.load_tokens()
    oauth_connected = oauth_tokens is not None and not oauth_mgr.is_expired(oauth_tokens)

    from nexus.config.presets import MODEL_CHOICES, PIPELINE_STAGES, PRESET_INFO

    # Current models (from config or raw yaml)
    models_raw = raw.get("models", {})

    # Pipeline running status
    pipeline_status = getattr(request.app.state, "pipeline_status", None)
    pipeline_running = pipeline_status is not None and not pipeline_status.get("done", True)

    validation_fields = request.query_params.get("fields", "").split(",") if error == "validation" else []

    return templates.TemplateResponse(request, "settings.html", {
        "providers": _provider_status(),
        "config": config,
        "raw": raw,
        "preset": getattr(config, "preset", raw.get("preset")),
        "saved": saved,
        "error": error,
        "validation_fields": validation_fields,
        "oauth_connected": oauth_connected,
        "model_choices": MODEL_CHOICES,
        "pipeline_stages": PIPELINE_STAGES,
        "preset_info": PRESET_INFO,
        "models_raw": models_raw,
        "tts_voices": TTS_VOICES,
        "tts_voices_json": json.dumps(TTS_VOICES),
        "pipeline_running": pipeline_running,
        "podcast_styles": PODCAST_STYLES,
    })


# ── Unified save ──────────────────────────────────────────

@router.post("/settings/save")
async def settings_save_all(request: Request):
    """Unified save: writes all config sections + API keys in one operation."""
    form = await request.form()
    data_dir = _data_dir(request)
    raw = _get_config_dict(request)

    # ─ Validate before writing ─
    tz = (form.get("timezone") or "").strip()
    sched = (form.get("schedule") or "").strip()
    errors = validate_settings(timezone=tz, schedule=sched)
    if errors:
        fields = ",".join(errors.keys())
        return RedirectResponse(url=f"/settings?error=validation&fields={fields}", status_code=303)

    # ─ API Keys ─
    keys = {}
    for p in PROVIDERS:
        val = (form.get(p["key"]) or "").strip()
        if val:
            keys[p["key"]] = val
    if keys:
        env_path = write_env(runtime_env_path(data_dir).parent, keys)
        load_runtime_env(env_path)

    # ─ User ─
    raw.setdefault("user", {})
    raw["user"]["name"] = (form.get("name") or "User").strip()
    raw["user"]["timezone"] = (form.get("timezone") or "UTC").strip()
    raw["user"]["output_language"] = (form.get("output_language") or "en").strip()

    # ─ Briefing ─
    raw.setdefault("briefing", {})
    raw["briefing"]["schedule"] = (form.get("schedule") or "06:00").strip()
    raw["briefing"]["style"] = (form.get("style") or "analytical").strip()
    raw["briefing"]["depth"] = (form.get("depth") or "detailed").strip()

    # ─ Audio ─
    raw.setdefault("audio", {})
    raw["audio"]["enabled"] = form.get("audio_enabled") == "on"
    raw["audio"]["tts_backend"] = (form.get("tts_backend") or "gemini").strip()
    raw["audio"]["voice_host_a"] = (form.get("voice_host_a") or "Kore").strip()
    raw["audio"]["voice_host_b"] = (form.get("voice_host_b") or "Puck").strip()
    # ElevenLabs voice tuning
    if raw["audio"]["tts_backend"] == "elevenlabs":
        try:
            raw["audio"]["elevenlabs_stability"] = float(form.get("elevenlabs_stability", "0.7"))
        except ValueError:
            raw["audio"]["elevenlabs_stability"] = 0.7
        try:
            raw["audio"]["elevenlabs_similarity_boost"] = float(form.get("elevenlabs_similarity_boost", "0.8"))
        except ValueError:
            raw["audio"]["elevenlabs_similarity_boost"] = 0.8
        try:
            raw["audio"]["elevenlabs_style"] = float(form.get("elevenlabs_style", "0.35"))
        except ValueError:
            raw["audio"]["elevenlabs_style"] = 0.35
        raw["audio"]["elevenlabs_speaker_boost"] = form.get("elevenlabs_speaker_boost") == "on"
    # Podcast style preset
    podcast_style = (form.get("podcast_style") or "").strip()
    if podcast_style in PODCAST_STYLES:
        raw["audio"]["podcast_style"] = podcast_style

    # ─ Telegram ─
    raw.setdefault("telegram", {})
    raw["telegram"]["enabled"] = form.get("telegram_enabled") == "on"

    # ─ Budget ─
    raw.setdefault("budget", {})
    try:
        raw["budget"]["daily_limit_usd"] = float(form.get("daily_limit_usd", "1.00"))
    except ValueError:
        raw["budget"]["daily_limit_usd"] = 1.00
    try:
        raw["budget"]["warning_threshold_usd"] = float(form.get("warning_threshold_usd", "0.50"))
    except ValueError:
        raw["budget"]["warning_threshold_usd"] = 0.50
    raw["budget"]["degradation_strategy"] = (form.get("degradation_strategy") or "skip_expensive").strip()

    # ─ Preset / Models ─
    preset = (form.get("preset") or raw.get("preset", "balanced")).strip()
    if preset in preset_names() or preset == "custom":
        raw["preset"] = preset
        if preset == "custom":
            from nexus.config.presets import PIPELINE_STAGES
            raw.setdefault("models", {})
            for stage_key, _, _ in PIPELINE_STAGES:
                val = (form.get(f"model_{stage_key}") or "").strip()
                if val:
                    raw["models"][stage_key] = val
        else:
            raw.pop("models", None)

    write_config(data_dir, raw)
    return RedirectResponse(url="/settings?saved=all", status_code=303)


# ── Restart ───────────────────────────────────────────────

@router.post("/settings/restart")
async def settings_restart(request: Request):
    """Restart the Nexus process."""
    env_path = runtime_env_path(_data_dir(request))

    def _restart():
        argv = [sys.executable, "-m", "nexus"] + sys.argv[1:]
        os.execve(sys.executable, argv, build_runtime_env(env_path))

    return RedirectResponse(
        url="/settings?saved=restarting",
        status_code=303,
        background=BackgroundTask(_restart),
    )


# ── Topics (stay as independent interactions) ─────────────

@router.post("/settings/topics")
async def settings_update_topics(request: Request):
    """Add a new topic."""
    form = await request.form()
    data_dir = _data_dir(request)
    raw = _get_config_dict(request)

    new_topic = (form.get("new_topic") or "").strip()
    if new_topic:
        raw.setdefault("topics", [])
        raw["topics"].append({"name": new_topic, "priority": "medium"})
        write_config(data_dir, raw)

    return RedirectResponse(url="/settings?saved=topics", status_code=303)


@router.post("/settings/topics/remove")
async def settings_remove_topic(request: Request):
    """Remove a topic by index."""
    form = await request.form()
    data_dir = _data_dir(request)
    raw = _get_config_dict(request)

    try:
        idx = int(form.get("index", "-1"))
        topics = raw.get("topics", [])
        if 0 <= idx < len(topics) and len(topics) > 1:
            topics.pop(idx)
            write_config(data_dir, raw)
    except (ValueError, IndexError):
        pass

    return RedirectResponse(url="/settings?saved=topics", status_code=303)


@router.post("/settings/topics/subtopics")
async def settings_update_subtopics(request: Request):
    """Add a subtopic to a topic."""
    form = await request.form()
    data_dir = _data_dir(request)
    raw = _get_config_dict(request)

    try:
        idx = int(form.get("topic_index", "-1"))
        subtopic = (form.get("subtopic") or "").strip()
        topics = raw.get("topics", [])
        if 0 <= idx < len(topics) and subtopic:
            topics[idx].setdefault("subtopics", [])
            if subtopic not in topics[idx]["subtopics"]:
                topics[idx]["subtopics"].append(subtopic)
                write_config(data_dir, raw)
    except (ValueError, IndexError):
        pass

    return RedirectResponse(url="/settings?saved=topics", status_code=303)


@router.post("/settings/topics/subtopics/remove")
async def settings_remove_subtopic(request: Request):
    """Remove a subtopic from a topic."""
    form = await request.form()
    data_dir = _data_dir(request)
    raw = _get_config_dict(request)

    try:
        topic_idx = int(form.get("topic_index", "-1"))
        sub_idx = int(form.get("subtopic_index", "-1"))
        topics = raw.get("topics", [])
        if 0 <= topic_idx < len(topics):
            subs = topics[topic_idx].get("subtopics", [])
            if 0 <= sub_idx < len(subs):
                subs.pop(sub_idx)
                write_config(data_dir, raw)
    except (ValueError, IndexError):
        pass

    return RedirectResponse(url="/settings?saved=topics", status_code=303)


# ── Legacy per-section endpoints (kept for backward compat) ─

@router.post("/settings/keys")
async def settings_update_keys(request: Request):
    """Update API keys in .env file."""
    form = await request.form()
    data_dir = _data_dir(request)

    keys = {}
    for p in PROVIDERS:
        val = (form.get(p["key"]) or "").strip()
        if val:
            keys[p["key"]] = val

    if keys:
        env_path = write_env(runtime_env_path(data_dir).parent, keys)
        load_runtime_env(env_path)

    return RedirectResponse(url="/settings?saved=keys", status_code=303)


@router.post("/settings/user")
async def settings_update_user(request: Request):
    """Update user config section."""
    form = await request.form()
    data_dir = _data_dir(request)
    raw = _get_config_dict(request)

    raw.setdefault("user", {})
    raw["user"]["name"] = (form.get("name") or "User").strip()
    raw["user"]["timezone"] = (form.get("timezone") or "UTC").strip()
    raw["user"]["output_language"] = (form.get("output_language") or "en").strip()

    write_config(data_dir, raw)
    return RedirectResponse(url="/settings?saved=user", status_code=303)


@router.post("/settings/briefing")
async def settings_update_briefing(request: Request):
    """Update briefing config section."""
    form = await request.form()
    data_dir = _data_dir(request)
    raw = _get_config_dict(request)

    raw.setdefault("briefing", {})
    raw["briefing"]["schedule"] = (form.get("schedule") or "06:00").strip()
    raw["briefing"]["style"] = (form.get("style") or "analytical").strip()
    raw["briefing"]["depth"] = (form.get("depth") or "detailed").strip()

    write_config(data_dir, raw)
    return RedirectResponse(url="/settings?saved=briefing", status_code=303)


@router.post("/settings/audio")
async def settings_update_audio(request: Request):
    """Update audio config section."""
    form = await request.form()
    data_dir = _data_dir(request)
    raw = _get_config_dict(request)

    raw.setdefault("audio", {})
    raw["audio"]["enabled"] = form.get("audio_enabled") == "on"
    raw["audio"]["tts_backend"] = (form.get("tts_backend") or "gemini").strip()
    raw["audio"]["voice_host_a"] = (form.get("voice_host_a") or "Kore").strip()
    raw["audio"]["voice_host_b"] = (form.get("voice_host_b") or "Puck").strip()
    if raw["audio"]["tts_backend"] == "elevenlabs":
        try:
            raw["audio"]["elevenlabs_stability"] = float(form.get("elevenlabs_stability", "0.7"))
        except ValueError:
            raw["audio"]["elevenlabs_stability"] = 0.7
        try:
            raw["audio"]["elevenlabs_similarity_boost"] = float(form.get("elevenlabs_similarity_boost", "0.8"))
        except ValueError:
            raw["audio"]["elevenlabs_similarity_boost"] = 0.8
        try:
            raw["audio"]["elevenlabs_style"] = float(form.get("elevenlabs_style", "0.35"))
        except ValueError:
            raw["audio"]["elevenlabs_style"] = 0.35
        raw["audio"]["elevenlabs_speaker_boost"] = form.get("elevenlabs_speaker_boost") == "on"

    write_config(data_dir, raw)
    return RedirectResponse(url="/settings?saved=audio", status_code=303)


@router.post("/settings/telegram")
async def settings_update_telegram(request: Request):
    """Update telegram config section."""
    form = await request.form()
    data_dir = _data_dir(request)
    raw = _get_config_dict(request)

    raw.setdefault("telegram", {})
    raw["telegram"]["enabled"] = form.get("telegram_enabled") == "on"

    write_config(data_dir, raw)
    return RedirectResponse(url="/settings?saved=telegram", status_code=303)


@router.post("/settings/budget")
async def settings_update_budget(request: Request):
    """Update budget config section."""
    form = await request.form()
    data_dir = _data_dir(request)
    raw = _get_config_dict(request)

    raw.setdefault("budget", {})
    try:
        raw["budget"]["daily_limit_usd"] = float(form.get("daily_limit_usd", "1.00"))
    except ValueError:
        raw["budget"]["daily_limit_usd"] = 1.00
    try:
        raw["budget"]["warning_threshold_usd"] = float(form.get("warning_threshold_usd", "0.50"))
    except ValueError:
        raw["budget"]["warning_threshold_usd"] = 0.50
    raw["budget"]["degradation_strategy"] = (form.get("degradation_strategy") or "skip_expensive").strip()

    write_config(data_dir, raw)
    return RedirectResponse(url="/settings?saved=budget", status_code=303)


@router.post("/settings/preset")
async def settings_update_preset(request: Request):
    """Switch model preset or save custom model selection."""
    form = await request.form()
    data_dir = _data_dir(request)
    raw = _get_config_dict(request)

    preset = (form.get("preset") or "balanced").strip()
    if preset not in preset_names():
        return RedirectResponse(url="/settings?error=invalid-preset", status_code=303)
    raw["preset"] = preset

    if preset == "custom":
        from nexus.config.presets import PIPELINE_STAGES
        raw.setdefault("models", {})
        for stage_key, _, _ in PIPELINE_STAGES:
            val = (form.get(f"model_{stage_key}") or "").strip()
            if val:
                raw["models"][stage_key] = val
    else:
        raw.pop("models", None)

    write_config(data_dir, raw)
    return RedirectResponse(url="/settings?saved=preset", status_code=303)


# ── ElevenLabs voice fetch ─────────────────────────────────

@router.get("/settings/elevenlabs/voices")
async def settings_elevenlabs_voices(request: Request):
    """Fetch available voices from ElevenLabs API."""
    import httpx
    from fastapi.responses import JSONResponse

    api_key = os.getenv("ELEVENLABS_API_KEY", "")
    if not api_key:
        return JSONResponse({"error": "ELEVENLABS_API_KEY not set", "voices": []}, status_code=400)

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                "https://api.elevenlabs.io/v1/voices",
                headers={"xi-api-key": api_key},
                timeout=10.0,
            )
        if resp.status_code != 200:
            return JSONResponse({"error": "ElevenLabs API error", "voices": []}, status_code=502)

        data = resp.json()
        voices = [
            {
                "voice_id": v["voice_id"],
                "name": v["name"],
                "category": v.get("category", ""),
            }
            for v in data.get("voices", [])
        ]
        return JSONResponse({"voices": voices})
    except Exception as e:
        logger.warning(f"ElevenLabs voice fetch failed: {e}")
        return JSONResponse({"error": str(e), "voices": []}, status_code=502)


# ── Telegram validation ────────────────────────────────────

@router.post("/settings/telegram/validate")
async def settings_telegram_validate(request: Request):
    """Validate Telegram bot token from settings (HTMX fragment)."""
    from nexus.agent.telegram_utils import validate_token
    from html import escape
    from fastapi.responses import HTMLResponse

    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    if not token:
        return HTMLResponse(
            '<div id="telegram-settings-status" class="form-hint" style="color:var(--danger)">'
            'No TELEGRAM_BOT_TOKEN in API keys.</div>'
        )

    bot_info = await validate_token(token)
    if bot_info:
        username = escape(bot_info.get("username", ""))
        config = getattr(request.app.state, "config", None)
        chat_id = getattr(config.telegram, "chat_id", None) if config else None

        if chat_id:
            return HTMLResponse(
                f'<div id="telegram-settings-status" class="form-hint" style="color:var(--success)">'
                f'Connected to @{username} (chat_id: {chat_id})</div>'
            )
        else:
            return HTMLResponse(
                f'<div id="telegram-settings-status">'
                f'<div class="form-hint" style="color:var(--success)">Bot: @{username}</div>'
                f'<div class="form-hint" style="color:var(--warning);margin-top:0.25rem">'
                f'No chat_id — send <code>/start</code> to '
                f'<a href="https://t.me/{username}" target="_blank">@{username}</a></div>'
                f'<div id="telegram-settings-poll" hx-get="/settings/telegram/poll" '
                f'hx-trigger="every 3s" hx-swap="outerHTML" style="margin-top:0.25rem">'
                f'<span class="form-hint">Waiting for /start...</span></div>'
                f'</div>'
            )
    else:
        return HTMLResponse(
            '<div id="telegram-settings-status" class="form-hint" style="color:var(--danger)">'
            'Invalid token. Check with @BotFather.</div>'
        )


@router.get("/settings/telegram/poll")
async def settings_telegram_poll(request: Request):
    """Poll for /start to capture chat_id from settings (HTMX fragment)."""
    from nexus.agent.telegram_utils import poll_for_chat_id
    from fastapi.responses import HTMLResponse

    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    if not token:
        return HTMLResponse('<div id="telegram-settings-poll"></div>')

    chat_id = await poll_for_chat_id(token, timeout=3.0)
    if chat_id:
        # Persist to config.yaml
        data_dir = _data_dir(request)
        raw = _get_config_dict(request)
        raw.setdefault("telegram", {})["chat_id"] = chat_id
        write_config(data_dir, raw)

        # Update in-memory config
        config = getattr(request.app.state, "config", None)
        if config:
            config.telegram.chat_id = chat_id

        return HTMLResponse(
            f'<div id="telegram-settings-poll" class="form-hint" style="color:var(--success)">'
            f'Connected! Chat ID: {chat_id}</div>'
        )
    else:
        return HTMLResponse(
            '<div id="telegram-settings-poll" hx-get="/settings/telegram/poll" '
            'hx-trigger="every 3s" hx-swap="outerHTML">'
            '<span class="form-hint">Waiting for /start...</span></div>'
        )
