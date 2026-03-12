"""Tests for ASGI middleware: admin protection and locality checks."""

import pytest
from unittest.mock import MagicMock, patch

import yaml
from httpx import AsyncClient, ASGITransport

from nexus.engine.knowledge.store import KnowledgeStore
from nexus.config.loader import load_config
from nexus.web.app import create_app
from nexus.web.middleware import _is_local_request, _is_private_ip


# --- Unit tests for helper functions ---


class TestIsPrivateIp:
    def test_loopback(self):
        assert _is_private_ip("127.0.0.1") is True

    def test_ipv6_loopback(self):
        assert _is_private_ip("::1") is True

    def test_docker_bridge(self):
        assert _is_private_ip("172.17.0.2") is True

    def test_lan_192(self):
        assert _is_private_ip("192.168.1.100") is True

    def test_lan_10(self):
        assert _is_private_ip("10.0.0.1") is True

    def test_public_ip(self):
        assert _is_private_ip("8.8.8.8") is False

    def test_public_ip_2(self):
        assert _is_private_ip("93.184.216.34") is False

    def test_invalid_string(self):
        assert _is_private_ip("not-an-ip") is False


class TestIsLocalRequest:
    def _make_request(self, client_host=None, host_header="localhost"):
        """Create a mock Request with specified client IP and Host header."""
        request = MagicMock()
        if client_host:
            request.client = MagicMock()
            request.client.host = client_host
        else:
            request.client = None
        request.headers = {"host": host_header}
        request.url = MagicMock()
        request.url.hostname = host_header.split(":")[0] if host_header else None
        return request

    def test_loopback_is_local(self):
        request = self._make_request(client_host="127.0.0.1")
        assert _is_local_request(request) is True

    def test_ipv6_loopback_is_local(self):
        request = self._make_request(client_host="::1")
        assert _is_local_request(request) is True

    def test_docker_bridge_is_local(self):
        request = self._make_request(client_host="172.17.0.2")
        assert _is_local_request(request) is True

    def test_public_ip_not_local(self):
        request = self._make_request(client_host="8.8.8.8", host_header="example.com")
        assert _is_local_request(request) is False

    def test_spoofed_host_header_blocked(self):
        """Remote client with Host: localhost should NOT be treated as local."""
        request = self._make_request(client_host="8.8.8.8", host_header="localhost")
        assert _is_local_request(request) is False

    def test_spoofed_host_header_loopback_blocked(self):
        """Remote client with Host: 127.0.0.1 should NOT be treated as local."""
        request = self._make_request(client_host="93.184.216.34", host_header="127.0.0.1:8080")
        assert _is_local_request(request) is False

    def test_no_client_not_local(self):
        request = self._make_request(client_host=None)
        assert _is_local_request(request) is False


# --- Integration tests for AdminProtectionMiddleware ---


@pytest.fixture
async def admin_app(tmp_path):
    """App with config for admin middleware testing."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    config_dict = {
        "preset": "balanced",
        "user": {"name": "Tester", "timezone": "UTC", "output_language": "en"},
        "topics": [{"name": "AI", "priority": "high"}],
    }
    (data_dir / "config.yaml").write_text(yaml.dump(config_dict, sort_keys=False))

    db_path = data_dir / "knowledge.db"
    app = create_app(db_path, data_dir=data_dir)
    store = KnowledgeStore(db_path)
    await store.initialize()
    app.state.store = store
    app.state.audio_dir = data_dir / "artifacts" / "audio"
    app.state.config = load_config(data_dir / "config.yaml")
    return app


@pytest.mark.asyncio
async def test_local_request_passes_through(admin_app):
    """GET /settings from localhost (127.0.0.1) is allowed."""
    transport = ASGITransport(app=admin_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/settings")
        assert resp.status_code == 200


@pytest.mark.asyncio
@patch.dict("os.environ", {"NEXUS_ADMIN_TOKEN": "secret-token-123"})
@patch("nexus.web.middleware._is_local_request", return_value=False)
async def test_valid_bearer_token_accepted(mock_local, admin_app):
    """Non-local request with correct Bearer token passes through."""
    transport = ASGITransport(app=admin_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get(
            "/settings",
            headers={"Authorization": "Bearer secret-token-123"},
        )
        assert resp.status_code == 200


@pytest.mark.asyncio
@patch.dict("os.environ", {"NEXUS_ADMIN_TOKEN": "secret-token-123"})
@patch("nexus.web.middleware._is_local_request", return_value=False)
async def test_invalid_bearer_token_rejected(mock_local, admin_app):
    """Non-local request with wrong Bearer token is denied."""
    transport = ASGITransport(app=admin_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get(
            "/settings",
            headers={"Authorization": "Bearer wrong-token"},
        )
        assert resp.status_code == 403


@pytest.mark.asyncio
@patch.dict("os.environ", {}, clear=False)
@patch("nexus.web.middleware._is_local_request", return_value=False)
async def test_no_token_non_local_rejected(mock_local, admin_app, monkeypatch):
    """Non-local request with no NEXUS_ADMIN_TOKEN configured is denied."""
    monkeypatch.delenv("NEXUS_ADMIN_TOKEN", raising=False)
    transport = ASGITransport(app=admin_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/settings")
        assert resp.status_code == 403
