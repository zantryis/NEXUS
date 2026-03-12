"""Tests for OpenAI OAuth PKCE flow."""

import json
import time
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock

from nexus.llm.oauth import (
    generate_pkce_pair,
    build_authorize_url,
    OpenAIOAuthManager,
)


def test_pkce_pair_generates_valid_values():
    """PKCE verifier and challenge should be valid base64url strings."""
    verifier, challenge = generate_pkce_pair()
    # Verifier: 43-128 chars, URL-safe base64
    assert 43 <= len(verifier) <= 128
    assert all(c in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_" for c in verifier)
    # Challenge: base64url encoded SHA256
    assert len(challenge) > 0
    assert "=" not in challenge  # no padding


def test_pkce_pair_is_unique():
    """Each call should produce a different pair."""
    pair1 = generate_pkce_pair()
    pair2 = generate_pkce_pair()
    assert pair1[0] != pair2[0]


def test_build_authorize_url():
    """Authorization URL should contain required OAuth params."""
    verifier, challenge = generate_pkce_pair()
    url = build_authorize_url(
        client_id="test-client-id",
        redirect_uri="http://localhost:8080/auth/openai/callback",
        code_challenge=challenge,
        state="random-state",
    )
    assert "auth.openai.com" in url
    assert "test-client-id" in url
    assert "code_challenge=" in url
    assert "random-state" in url
    assert "S256" in url


def test_oauth_manager_load_tokens_missing(tmp_path):
    """Loading tokens from non-existent file returns None."""
    mgr = OpenAIOAuthManager(token_path=tmp_path / "tokens.json")
    assert mgr.load_tokens() is None


def test_oauth_manager_save_and_load(tmp_path):
    """Saving tokens then loading should return them."""
    token_path = tmp_path / "tokens.json"
    mgr = OpenAIOAuthManager(token_path=token_path)
    tokens = {
        "access_token": "at-123",
        "refresh_token": "rt-456",
        "expires_at": time.time() + 3600,
    }
    mgr.save_tokens(tokens)
    loaded = mgr.load_tokens()
    assert loaded["access_token"] == "at-123"
    assert loaded["refresh_token"] == "rt-456"


def test_oauth_manager_token_file_permissions(tmp_path):
    """Token file should be created with restricted permissions."""
    token_path = tmp_path / "tokens.json"
    mgr = OpenAIOAuthManager(token_path=token_path)
    mgr.save_tokens({"access_token": "secret", "expires_at": time.time() + 3600})
    # On Unix, file should be readable only by owner
    mode = token_path.stat().st_mode & 0o777
    assert mode == 0o600


def test_oauth_manager_is_expired():
    """Should detect expired tokens."""
    mgr = OpenAIOAuthManager(token_path=Path("/tmp/fake"))
    assert mgr.is_expired({"expires_at": time.time() - 100}) is True
    assert mgr.is_expired({"expires_at": time.time() + 3600}) is False


def test_oauth_manager_is_expired_missing_field():
    """Missing expires_at should be treated as expired."""
    mgr = OpenAIOAuthManager(token_path=Path("/tmp/fake"))
    assert mgr.is_expired({}) is True


def test_oauth_manager_clear_tokens(tmp_path):
    """Clearing should remove the token file."""
    token_path = tmp_path / "tokens.json"
    mgr = OpenAIOAuthManager(token_path=token_path)
    mgr.save_tokens({"access_token": "x", "expires_at": time.time() + 3600})
    assert token_path.exists()
    mgr.clear_tokens()
    assert not token_path.exists()
    assert mgr.load_tokens() is None


@pytest.mark.asyncio
async def test_oauth_manager_exchange_code(tmp_path):
    """Exchange auth code for tokens via HTTP POST."""
    token_path = tmp_path / "tokens.json"
    mgr = OpenAIOAuthManager(token_path=token_path)

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "access_token": "new-at",
        "refresh_token": "new-rt",
        "expires_in": 3600,
        "token_type": "Bearer",
    }

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_response):
        tokens = await mgr.exchange_code(
            code="auth-code-123",
            code_verifier="test-verifier",
            client_id="client-id",
            redirect_uri="http://localhost:8080/auth/openai/callback",
        )

    assert tokens["access_token"] == "new-at"
    assert tokens["refresh_token"] == "new-rt"
    assert "expires_at" in tokens
    # Should auto-save
    loaded = mgr.load_tokens()
    assert loaded["access_token"] == "new-at"


@pytest.mark.asyncio
async def test_oauth_manager_refresh_token(tmp_path):
    """Refresh expired token using refresh_token."""
    token_path = tmp_path / "tokens.json"
    mgr = OpenAIOAuthManager(token_path=token_path)
    mgr.save_tokens({
        "access_token": "old-at",
        "refresh_token": "rt-valid",
        "expires_at": time.time() - 100,  # expired
    })

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "access_token": "fresh-at",
        "refresh_token": "fresh-rt",
        "expires_in": 3600,
        "token_type": "Bearer",
    }

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_response):
        token = await mgr.get_valid_token(client_id="cid")

    assert token == "fresh-at"
