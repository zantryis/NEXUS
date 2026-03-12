"""OpenAI OAuth PKCE flow — experimental alternative to API keys."""

import hashlib
import json
import os
import secrets
import time
from base64 import urlsafe_b64encode
from pathlib import Path
from urllib.parse import urlencode

import httpx

# OpenAI OAuth endpoints (OpenClaw-style PKCE)
AUTHORIZE_URL = "https://auth.openai.com/oauth/authorize"
TOKEN_URL = "https://auth.openai.com/oauth/token"


def generate_pkce_pair() -> tuple[str, str]:
    """Generate PKCE code_verifier and code_challenge (S256).

    Returns (verifier, challenge).
    """
    verifier = secrets.token_urlsafe(64)
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return verifier, challenge


def build_authorize_url(
    client_id: str,
    redirect_uri: str,
    code_challenge: str,
    state: str,
    scope: str = "openai.public",
) -> str:
    """Build the OpenAI OAuth authorization URL."""
    params = {
        "response_type": "code",
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
        "state": state,
        "scope": scope,
    }
    return f"{AUTHORIZE_URL}?{urlencode(params)}"


class OpenAIOAuthManager:
    """Manages OpenAI OAuth tokens — exchange, refresh, persist."""

    def __init__(self, token_path: Path):
        self._token_path = token_path

    def load_tokens(self) -> dict | None:
        """Load stored tokens from disk."""
        if not self._token_path.exists():
            return None
        try:
            return json.loads(self._token_path.read_text())
        except (json.JSONDecodeError, OSError):
            return None

    def save_tokens(self, tokens: dict) -> None:
        """Save tokens to disk with restricted permissions."""
        self._token_path.parent.mkdir(parents=True, exist_ok=True)
        self._token_path.write_text(json.dumps(tokens, indent=2))
        os.chmod(self._token_path, 0o600)

    def clear_tokens(self) -> None:
        """Remove stored token file."""
        if self._token_path.exists():
            self._token_path.unlink()

    def is_expired(self, tokens: dict) -> bool:
        """Check if access token is expired (with 60s buffer)."""
        expires_at = tokens.get("expires_at", 0)
        return time.time() >= (expires_at - 60)

    async def exchange_code(
        self,
        code: str,
        code_verifier: str,
        client_id: str,
        redirect_uri: str,
    ) -> dict:
        """Exchange authorization code for access + refresh tokens."""
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                TOKEN_URL,
                data={
                    "grant_type": "authorization_code",
                    "code": code,
                    "code_verifier": code_verifier,
                    "client_id": client_id,
                    "redirect_uri": redirect_uri,
                },
            )

        data = resp.json()
        if resp.status_code != 200:
            raise RuntimeError(f"Token exchange failed: {data}")

        tokens = {
            "access_token": data["access_token"],
            "refresh_token": data.get("refresh_token"),
            "expires_at": time.time() + data.get("expires_in", 3600),
            "token_type": data.get("token_type", "Bearer"),
        }
        self.save_tokens(tokens)
        return tokens

    async def _refresh(self, refresh_token: str, client_id: str) -> dict:
        """Refresh an expired access token."""
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                TOKEN_URL,
                data={
                    "grant_type": "refresh_token",
                    "refresh_token": refresh_token,
                    "client_id": client_id,
                },
            )

        data = resp.json()
        if resp.status_code != 200:
            raise RuntimeError(f"Token refresh failed: {data}")

        tokens = {
            "access_token": data["access_token"],
            "refresh_token": data.get("refresh_token", refresh_token),
            "expires_at": time.time() + data.get("expires_in", 3600),
            "token_type": data.get("token_type", "Bearer"),
        }
        self.save_tokens(tokens)
        return tokens

    async def get_valid_token(self, client_id: str) -> str | None:
        """Get a valid access token, refreshing if needed.

        Returns the access token string, or None if no tokens stored.
        """
        tokens = self.load_tokens()
        if not tokens:
            return None

        if self.is_expired(tokens):
            refresh_token = tokens.get("refresh_token")
            if not refresh_token:
                return None
            tokens = await self._refresh(refresh_token, client_id)

        return tokens.get("access_token")
