"""OpenAI OAuth routes — PKCE flow for subscription-based API access (experimental)."""

import logging
import secrets
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, RedirectResponse

from nexus.llm.oauth import (
    OpenAIOAuthManager,
    build_authorize_url,
    generate_pkce_pair,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# Default client ID — users can override via OPENAI_OAUTH_CLIENT_ID env var
DEFAULT_CLIENT_ID = "nexus-intelligence"


def _data_dir(request: Request) -> Path:
    return getattr(request.app.state, "data_dir", Path("data"))


def _client_id(request: Request) -> str:
    import os
    return os.getenv("OPENAI_OAUTH_CLIENT_ID", DEFAULT_CLIENT_ID)


def _redirect_uri(request: Request) -> str:
    """Build the callback URL from the request's host."""
    scheme = request.headers.get("x-forwarded-proto", request.url.scheme)
    host = request.headers.get("x-forwarded-host", request.url.netloc)
    return f"{scheme}://{host}/auth/openai/callback"


def _get_manager(request: Request) -> OpenAIOAuthManager:
    return OpenAIOAuthManager(token_path=_data_dir(request) / ".oauth-tokens.json")


@router.get("/auth/openai")
async def oauth_initiate(request: Request):
    """Start OpenAI OAuth PKCE flow — redirect to OpenAI authorization."""
    verifier, challenge = generate_pkce_pair()
    state = secrets.token_urlsafe(32)

    # Store PKCE verifier + origin in server-side session
    if not hasattr(request.app.state, "oauth_sessions"):
        request.app.state.oauth_sessions = {}
    origin = request.query_params.get("from", "")
    request.app.state.oauth_sessions[state] = {
        "code_verifier": verifier,
        "from": origin,
    }

    url = build_authorize_url(
        client_id=_client_id(request),
        redirect_uri=_redirect_uri(request),
        code_challenge=challenge,
        state=state,
    )
    return RedirectResponse(url=url, status_code=307)


@router.get("/auth/openai/callback")
async def oauth_callback(request: Request):
    """Handle OpenAI OAuth callback — exchange code for tokens."""
    code = request.query_params.get("code")
    state = request.query_params.get("state")
    error = request.query_params.get("error")

    if error:
        logger.error(f"OAuth error from OpenAI: {error}")
        return JSONResponse(
            {"error": f"OpenAI authorization failed: {error}"},
            status_code=400,
        )

    if not code or not state:
        return JSONResponse({"error": "Missing code or state parameter."}, status_code=400)

    # Validate state
    sessions = getattr(request.app.state, "oauth_sessions", {})
    session = sessions.pop(state, None)
    if not session:
        return JSONResponse({"error": "Invalid or expired state parameter."}, status_code=400)

    mgr = _get_manager(request)
    try:
        await mgr.exchange_code(
            code=code,
            code_verifier=session["code_verifier"],
            client_id=_client_id(request),
            redirect_uri=_redirect_uri(request),
        )
    except RuntimeError as e:
        logger.error(f"OAuth token exchange failed: {e}")
        return JSONResponse({"error": str(e)}, status_code=400)

    # If initiated from setup wizard, mark OAuth in the setup session and redirect back
    if session.get("from") == "setup":
        setup_sessions = getattr(request.app.state, "setup_sessions", {})
        # Find the active setup session from cookies
        setup_sid = request.cookies.get("nexus_setup")
        if setup_sid and setup_sid in setup_sessions:
            setup_sessions[setup_sid]["oauth_openai"] = True
        return RedirectResponse(url="/setup/step/3", status_code=303)

    return RedirectResponse(url="/settings?saved=oauth", status_code=303)


@router.post("/auth/openai/revoke")
async def oauth_revoke(request: Request):
    """Clear stored OAuth tokens."""
    mgr = _get_manager(request)
    mgr.clear_tokens()
    return RedirectResponse(url="/settings?saved=oauth", status_code=303)
