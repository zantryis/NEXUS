"""ASGI middleware for setup redirect, admin protection, security headers, and demo mode."""

import functools
import hmac
import ipaddress
import os
from pathlib import Path
from urllib.parse import urlencode

from starlette.datastructures import MutableHeaders
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, RedirectResponse
from starlette.types import ASGIApp, Receive, Scope, Send


class SetupRedirectMiddleware:
    """Redirect all routes to /setup when config.yaml is missing."""

    PASSTHROUGH_PREFIXES = ("/setup", "/static", "/auth", "/api")

    def __init__(self, app: ASGIApp, data_dir: Path) -> None:
        self.app = app
        self.state = getattr(app, "state", None)
        self.data_dir = data_dir

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            path = scope.get("path", "")
            config_exists = (self.data_dir / "config.yaml").exists()
            if not config_exists and not any(
                path.startswith(p) for p in self.PASSTHROUGH_PREFIXES
            ):
                response = RedirectResponse(url="/setup", status_code=307)
                await response(scope, receive, send)
                return
        await self.app(scope, receive, send)


def _is_loopback_ip(addr: str) -> bool:
    """Return True when the address is a loopback interface."""
    try:
        ip = ipaddress.ip_address(addr)
        return ip.is_loopback
    except ValueError:
        return False


def _trust_docker_gateway() -> bool:
    """Allow Docker bridge gateway requests only when explicitly enabled."""
    return os.getenv("NEXUS_TRUST_DOCKER_GATEWAY", "").lower() in ("1", "true", "yes")


@functools.lru_cache(maxsize=1)
def _docker_gateway_ip() -> ipaddress.IPv4Address | None:
    """Read the default IPv4 gateway from /proc/net/route for Docker localhost access."""
    route_path = Path("/proc/net/route")
    if not route_path.exists():
        return None

    try:
        for line in route_path.read_text().splitlines()[1:]:
            fields = line.split()
            if len(fields) < 3 or fields[1] != "00000000" or fields[0] == "lo":
                continue
            gateway_hex = fields[2]
            gateway_int = int(gateway_hex, 16)
            return ipaddress.ip_address(gateway_int.to_bytes(4, "little"))
    except (OSError, ValueError):
        return None

    return None


def _is_local_request(request: Request) -> bool:
    """Allow same-machine access only, with an opt-in Docker gateway exception."""
    if request.client and request.client.host:
        client_host = request.client.host
        if _is_loopback_ip(client_host):
            return True
        if _trust_docker_gateway():
            gateway_ip = _docker_gateway_ip()
            if gateway_ip is not None:
                try:
                    return ipaddress.ip_address(client_host) == gateway_ip
                except ValueError:
                    return False
    return False


def _is_demo_enabled(app) -> bool:
    """Demo mode can be enabled via env or loaded config."""
    if os.getenv("NEXUS_DEMO_MODE", "").lower() in ("1", "true", "yes"):
        return True
    state = getattr(app, "state", None)
    config = getattr(state, "config", None)
    return bool(getattr(config, "demo_mode", False))


class AdminProtectionMiddleware:
    """Protect writable admin routes from non-local access."""

    PROTECTED_PREFIXES = ("/settings", "/setup")
    COOKIE_NAME = "nexus_admin"

    def __init__(self, app: ASGIApp, data_dir: Path) -> None:
        self.app = app
        self.state = getattr(app, "state", None)
        self.data_dir = data_dir

    def _token(self) -> str:
        return os.getenv("NEXUS_ADMIN_TOKEN", "").strip()

    def _setup_reset_allowed(self) -> bool:
        return os.getenv("NEXUS_ALLOW_SETUP_RESET", "").lower() in ("1", "true", "yes")

    def _provided_token(self, request: Request) -> str:
        auth = request.headers.get("authorization", "")
        if auth.lower().startswith("bearer "):
            return auth[7:].strip()
        return (
            request.query_params.get("admin_token")
            or request.headers.get("x-nexus-admin-token")
            or request.cookies.get(self.COOKIE_NAME)
            or ""
        ).strip()

    def _deny_response(self, request: Request, token_configured: bool):
        remote_hint = (
            "Set NEXUS_ADMIN_TOKEN and reopen this URL with ?admin_token=YOUR_TOKEN."
            if token_configured else
            "Admin routes are limited to same-machine access unless NEXUS_ADMIN_TOKEN is configured."
        )
        if request.method == "GET":
            return HTMLResponse(remote_hint, status_code=403)
        return JSONResponse({"error": remote_hint}, status_code=403)

    def _clean_redirect_target(self, request: Request) -> str:
        params = [
            (key, value)
            for key, value in request.query_params.multi_items()
            if key != "admin_token"
        ]
        query = urlencode(params, doseq=True)
        return f"{request.url.path}?{query}" if query else request.url.path

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        if not any(path.startswith(prefix) for prefix in self.PROTECTED_PREFIXES):
            await self.app(scope, receive, send)
            return

        request = Request(scope, receive=receive)
        config_exists = (self.data_dir / "config.yaml").exists()

        # After initial setup, block wizard pages but allow launch/status (used from dashboard)
        if config_exists and path.startswith("/setup") and not self._setup_reset_allowed():
            if path in ("/setup/launch", "/setup/status"):
                pass  # These post-setup endpoints are needed from the dashboard
            elif request.method == "GET":
                response = RedirectResponse(url="/settings", status_code=303)
                await response(scope, receive, send)
                return
            else:
                response = JSONResponse(
                    {
                        "error": (
                            "Setup wizard is disabled after initial configuration. "
                            "Delete data/config.yaml or set NEXUS_ALLOW_SETUP_RESET=1 to re-run it."
                        )
                    },
                    status_code=403,
                )
                await response(scope, receive, send)
                return

        if _is_local_request(request):
            await self.app(scope, receive, send)
            return

        token = self._token()
        provided = self._provided_token(request)
        if token and hmac.compare_digest(provided, token):
            if request.query_params.get("admin_token") and request.method in {"GET", "HEAD"}:
                response = RedirectResponse(
                    url=self._clean_redirect_target(request),
                    status_code=303,
                )
                response.set_cookie(
                    self.COOKIE_NAME,
                    token,
                    httponly=True,
                    max_age=86400,
                    samesite="lax",
                )
                await response(scope, receive, send)
                return
            await self.app(scope, receive, send)
            return

        response = self._deny_response(request, token_configured=bool(token))
        await response(scope, receive, send)


class SecurityHeadersMiddleware:
    """Attach baseline browser hardening headers to every HTTP response."""

    CONTENT_SECURITY_POLICY = (
        "default-src 'self'; "
        "base-uri 'self'; "
        "frame-ancestors 'none'; "
        "form-action 'self'; "
        "script-src 'self' 'unsafe-inline' https://unpkg.com https://d3js.org; "
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
        "font-src 'self' https://fonts.gstatic.com; "
        "img-src 'self' data: https:; "
        "connect-src 'self'; "
        "media-src 'self'"
    )

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_with_headers(message):
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                headers.setdefault("X-Frame-Options", "DENY")
                headers.setdefault("X-Content-Type-Options", "nosniff")
                headers.setdefault("Referrer-Policy", "same-origin")
                headers.setdefault(
                    "Content-Security-Policy",
                    self.CONTENT_SECURITY_POLICY,
                )
            await send(message)

        await self.app(scope, receive, send_with_headers)


class DemoModeMiddleware:
    """Block mutation requests in demo mode."""

    BLOCKED_PREFIXES = ("/settings", "/setup")
    BLOCKED_METHODS = {"POST", "PUT", "DELETE", "PATCH"}
    # Allow chat API even in demo mode (rate-limited separately)
    ALLOWED_PATHS = ("/api/chat",)

    def __init__(self, app: ASGIApp) -> None:
        self.app = app
        self.state = getattr(app, "state", None)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and _is_demo_enabled(self.app):
            method = scope.get("method", "GET")
            path = scope.get("path", "")
            if (
                method in self.BLOCKED_METHODS
                and any(path.startswith(p) for p in self.BLOCKED_PREFIXES)
                and not any(path.startswith(p) for p in self.ALLOWED_PATHS)
            ):
                response = JSONResponse(
                    {"error": "This instance is in demo mode. Settings cannot be changed."},
                    status_code=403,
                )
                await response(scope, receive, send)
                return
        await self.app(scope, receive, send)
