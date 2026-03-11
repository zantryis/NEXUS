"""Settings page — API key status and configuration."""

import os

from fastapi import APIRouter, Request

from nexus.web.app import get_templates

router = APIRouter()

PROVIDERS = [
    {"key": "GEMINI_API_KEY", "name": "Gemini (Google)", "prefix": "gemini"},
    {"key": "DEEPSEEK_API_KEY", "name": "DeepSeek", "prefix": "deepseek"},
    {"key": "ANTHROPIC_API_KEY", "name": "Anthropic (Claude)", "prefix": "claude"},
]


@router.get("/settings")
async def settings_page(request: Request):
    """Settings page showing API key status."""
    templates = get_templates(request)

    provider_status = []
    for p in PROVIDERS:
        val = os.getenv(p["key"], "")
        provider_status.append({
            "key": p["key"],
            "name": p["name"],
            "is_set": bool(val),
            "masked": f"...{val[-4:]}" if len(val) > 4 else ("***" if val else "not set"),
        })

    # Get current preset from app state config if available
    config = getattr(request.app.state, "config", None)
    preset = getattr(config, "preset", None) if config else None

    return templates.TemplateResponse(request, "settings.html", {
        "providers": provider_status,
        "preset": preset,
    })
