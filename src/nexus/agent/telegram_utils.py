"""Telegram Bot API utilities — token validation and chat_id capture.

Uses raw httpx calls so these work without starting a full bot instance.
"""

import logging

import httpx

logger = logging.getLogger(__name__)

TELEGRAM_API = "https://api.telegram.org"


async def validate_token(token: str) -> dict | None:
    """Call getMe to verify a bot token.

    Returns the bot info dict (with 'username', 'first_name', etc.) on success,
    or None if the token is invalid or the request fails.
    """
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{TELEGRAM_API}/bot{token}/getMe",
                timeout=10.0,
            )
        if resp.status_code == 200:
            data = resp.json()
            if data.get("ok"):
                return data["result"]
        return None
    except (httpx.HTTPError, Exception) as e:
        logger.debug(f"Telegram token validation failed: {e}")
        return None


async def poll_for_chat_id(token: str, timeout: float = 5.0) -> int | None:
    """Short-poll getUpdates looking for a /start message.

    Makes a single getUpdates call with the given timeout. Returns the chat_id
    from the first /start message found, or None if none arrived.
    """
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{TELEGRAM_API}/bot{token}/getUpdates",
                params={
                    "timeout": int(timeout),
                    "allowed_updates": '["message"]',
                },
                timeout=timeout + 5.0,  # HTTP timeout > long-poll timeout
            )
        if resp.status_code != 200:
            return None

        data = resp.json()
        if not data.get("ok"):
            return None

        for update in data.get("result", []):
            message = update.get("message", {})
            text = message.get("text", "")
            if text.strip().startswith("/start"):
                chat_id = message.get("chat", {}).get("id")
                if chat_id:
                    # Acknowledge the update so it's not re-read
                    offset = update["update_id"] + 1
                    try:
                        async with httpx.AsyncClient() as client:
                            await client.get(
                                f"{TELEGRAM_API}/bot{token}/getUpdates",
                                params={"offset": offset, "limit": 0},
                                timeout=5.0,
                            )
                    except Exception:
                        pass  # Best-effort acknowledgment
                    return chat_id

        return None
    except (httpx.HTTPError, Exception) as e:
        logger.debug(f"Telegram poll failed: {e}")
        return None
