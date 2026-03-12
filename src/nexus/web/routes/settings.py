"""Settings page — view and modify API keys, config, and feature toggles."""

import os
import logging
from pathlib import Path

from fastapi import APIRouter, Request, Form
from fastapi.responses import RedirectResponse

from nexus.config.writer import write_config, write_env
from nexus.web.app import get_templates

logger = logging.getLogger(__name__)

router = APIRouter()

PROVIDERS = [
    {"key": "GEMINI_API_KEY", "name": "Gemini (Google)", "prefix": "gemini"},
    {"key": "OPENAI_API_KEY", "name": "OpenAI", "prefix": "openai"},
    {"key": "ANTHROPIC_API_KEY", "name": "Anthropic (Claude)", "prefix": "claude"},
    {"key": "DEEPSEEK_API_KEY", "name": "DeepSeek", "prefix": "deepseek"},
    {"key": "ELEVENLABS_API_KEY", "name": "ElevenLabs (TTS)", "prefix": "elevenlabs"},
    {"key": "TELEGRAM_BOT_TOKEN", "name": "Telegram Bot", "prefix": "telegram"},
]


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

    return templates.TemplateResponse(request, "settings.html", {
        "providers": _provider_status(),
        "config": config,
        "raw": raw,
        "preset": getattr(config, "preset", raw.get("preset")),
        "saved": saved,
    })


@router.post("/settings/keys")
async def settings_update_keys(request: Request):
    """Update API keys in .env file."""
    form = await request.form()
    data_dir = _data_dir(request)

    keys = {}
    for p in PROVIDERS:
        val = (form.get(p["key"]) or "").strip()
        if val:  # Only update keys that have a new value entered
            keys[p["key"]] = val

    if keys:
        write_env(data_dir.parent, keys)

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


@router.post("/settings/preset")
async def settings_update_preset(request: Request):
    """Switch model preset."""
    form = await request.form()
    data_dir = _data_dir(request)
    raw = _get_config_dict(request)

    raw["preset"] = (form.get("preset") or "balanced").strip()

    write_config(data_dir, raw)
    return RedirectResponse(url="/settings?saved=preset", status_code=303)
