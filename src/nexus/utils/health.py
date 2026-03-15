"""Runtime health helpers for dashboard, bot, and hosted deployments."""

from __future__ import annotations

import base64
import json
import os
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from nexus.config.models import NexusConfig

LITELLM_ENV_NAMES = (
    "LITELLM_BASE_URL",
    "LITELLM_PROXY_URL",
    "LITELLM_API_KEY",
    "LITELLM_PROXY_API_KEY",
)

LITELLM_ALIAS_ENV_MAP = {
    "litellm/gpt": "LITELLM_MODEL_GPT",
    "litellm/opus": "LITELLM_MODEL_OPUS",
    "litellm/sonnet": "LITELLM_MODEL_SONNET",
    "litellm/gemini": "LITELLM_MODEL_GEMINI",
}


def _env_first(*names: str) -> str:
    for name in names:
        value = os.getenv(name, "").strip()
        if value:
            return value
    return ""


def _safe_parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    text = value.strip()
    if not text:
        return None
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _decode_jwt_expiry(token: str | None) -> datetime | None:
    if not token or token.count(".") < 2:
        return None
    try:
        payload = token.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        decoded = base64.urlsafe_b64decode(payload.encode())
        data = json.loads(decoded.decode())
        exp = data.get("exp")
        if exp is None:
            return None
        return datetime.fromtimestamp(float(exp), tz=timezone.utc)
    except Exception:
        return None


def _litellm_token_expiry() -> datetime | None:
    explicit = _safe_parse_datetime(os.getenv("LITELLM_PROXY_TOKEN_EXPIRES_AT"))
    if explicit:
        return explicit
    token = _env_first("LITELLM_API_KEY", "LITELLM_PROXY_API_KEY")
    return _decode_jwt_expiry(token)


def _minutes_until(dt: datetime | None, now: datetime) -> int | None:
    if dt is None:
        return None
    return int((dt - now).total_seconds() // 60)


def _configured_model_map(config: NexusConfig) -> dict[str, str]:
    return {
        "discovery": config.models.discovery,
        "filtering": config.models.filtering,
        "synthesis": config.models.synthesis,
        "dialogue_script": config.models.dialogue_script,
        "knowledge_summary": config.models.knowledge_summary,
        "breaking_news": config.models.breaking_news,
        "agent": config.models.agent,
    }


def _today_artifact_paths(data_dir: Path, today: date) -> tuple[Path, Path]:
    return (
        data_dir / "artifacts" / "briefings" / f"{today.isoformat()}.md",
        data_dir / "artifacts" / "audio" / f"{today.isoformat()}.mp3",
    )


async def build_health_snapshot(
    config: NexusConfig,
    data_dir: Path,
    store=None,
) -> dict[str, Any]:
    """Build a product-facing health snapshot.

    Keeps the checks lightweight so it can be called from the dashboard or bot
    without making external network requests.
    """
    now = datetime.now(timezone.utc)
    today = now.date()
    briefing_path, audio_path = _today_artifact_paths(data_dir, today)

    model_map = _configured_model_map(config)
    litellm_steps: dict[str, str] = {
        step: model for step, model in model_map.items() if model.startswith("litellm/")
    }
    alias_targets = {
        step: _env_first(LITELLM_ALIAS_ENV_MAP[model])
        for step, model in litellm_steps.items()
        if model in LITELLM_ALIAS_ENV_MAP
    }
    missing_aliases = {
        step: LITELLM_ALIAS_ENV_MAP[model]
        for step, model in litellm_steps.items()
        if model in LITELLM_ALIAS_ENV_MAP and not alias_targets.get(step)
    }

    base_url = _env_first("LITELLM_BASE_URL", "LITELLM_PROXY_URL")
    api_key = _env_first("LITELLM_API_KEY", "LITELLM_PROXY_API_KEY")
    token_expiry = _litellm_token_expiry()
    token_ttl_min = _minutes_until(token_expiry, now)

    pipeline_running = False
    last_run = None
    if store is not None:
        pipeline_running = await store.is_pipeline_running()
        last_run = await store.get_last_pipeline_run()

    configured_topics = [t.name for t in config.topics]
    last_topics = last_run["topics"] if last_run else []
    missing_topics = [topic for topic in configured_topics if topic not in last_topics]

    issues: list[dict[str, str]] = []

    if litellm_steps and (not base_url or not api_key):
        issues.append({
            "severity": "critical",
            "message": "LiteLLM is configured for one or more steps, but proxy credentials are missing.",
        })

    for step, env_name in missing_aliases.items():
        issues.append({
            "severity": "critical",
            "message": f"{step} uses a hosted LiteLLM alias, but {env_name} is not set.",
        })

    if token_ttl_min is not None and token_ttl_min < 0:
        issues.append({
            "severity": "critical",
            "message": "The hosted LiteLLM proxy token appears expired.",
        })
    elif token_ttl_min is not None and token_ttl_min < 15:
        issues.append({
            "severity": "warning",
            "message": f"The hosted LiteLLM proxy token expires soon ({token_ttl_min}m).",
        })

    if config.telegram.enabled and config.telegram.chat_id is None:
        issues.append({
            "severity": "warning",
            "message": "Telegram is enabled but no chat_id has been registered yet.",
        })

    if last_run and last_run["status"] == "failed":
        issues.append({
            "severity": "critical",
            "message": f"The last pipeline run failed: {last_run.get('error') or 'unknown error'}",
        })

    if last_run and last_run["status"] == "completed" and missing_topics:
        issues.append({
            "severity": "warning",
            "message": (
                f"The last completed run covered {len(last_topics)}/{len(configured_topics)} topics; "
                f"missing: {', '.join(missing_topics[:4])}"
            ),
        })

    if briefing_path.exists() and config.audio.enabled and not audio_path.exists():
        issues.append({
            "severity": "warning",
            "message": "Today's briefing exists, but today's podcast audio is missing.",
        })

    overall = "ok"
    if any(i["severity"] == "critical" for i in issues):
        overall = "critical"
    elif issues:
        overall = "warning"

    return {
        "status": overall,
        "checked_at": now.isoformat(),
        "pipeline": {
            "running": pipeline_running,
            "last_run": last_run,
            "configured_topics": configured_topics,
            "configured_topic_count": len(configured_topics),
            "last_run_topic_count": len(last_topics),
            "missing_topics": missing_topics,
        },
        "deliverables": {
            "briefing_today": briefing_path.exists(),
            "audio_today": audio_path.exists(),
        },
        "telegram": {
            "enabled": config.telegram.enabled,
            "chat_id_configured": config.telegram.chat_id is not None,
        },
        "litellm": {
            "used": bool(litellm_steps),
            "configured": bool(base_url and api_key) if litellm_steps else True,
            "base_url_present": bool(base_url),
            "api_key_present": bool(api_key),
            "steps": litellm_steps,
            "alias_targets": alias_targets,
            "missing_aliases": missing_aliases,
            "token_expires_at": token_expiry.isoformat() if token_expiry else None,
            "token_ttl_minutes": token_ttl_min,
        },
        "issues": issues,
    }


def health_summary_lines(snapshot: dict[str, Any]) -> list[str]:
    """Convert a health snapshot into compact human-readable lines."""
    status_emoji = {
        "ok": "✅",
        "warning": "⚠️",
        "critical": "🚨",
    }
    lines = [f"{status_emoji.get(snapshot['status'], 'ℹ️')} <b>System health:</b> {snapshot['status']}"]

    litellm = snapshot["litellm"]
    if litellm["used"]:
        proxy_line = "configured" if litellm["configured"] else "missing config"
        ttl = litellm.get("token_ttl_minutes")
        if ttl is not None:
            proxy_line += f", token TTL {ttl}m"
        lines.append(f"• LiteLLM proxy: {proxy_line}")

    pipeline = snapshot["pipeline"]
    if pipeline["last_run"]:
        last_run = pipeline["last_run"]
        lines.append(
            f"• Last run: {last_run['status']} — {last_run['event_count']} events across "
            f"{len(last_run['topics'])}/{pipeline['configured_topic_count']} topics"
        )
    else:
        lines.append("• Last run: none yet")

    deliverables = snapshot["deliverables"]
    lines.append(
        f"• Today's outputs: briefing={'yes' if deliverables['briefing_today'] else 'no'}, "
        f"podcast={'yes' if deliverables['audio_today'] else 'no'}"
    )
    telegram = snapshot["telegram"]
    lines.append(
        f"• Telegram: {'enabled' if telegram['enabled'] else 'disabled'}, "
        f"chat={'configured' if telegram['chat_id_configured'] else 'missing'}"
    )

    if snapshot["issues"]:
        lines.append("")
        lines.append("<b>Attention</b>")
        for issue in snapshot["issues"][:4]:
            lines.append(f"• {issue['message']}")

    return lines
