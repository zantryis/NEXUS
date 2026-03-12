"""ASGI middleware for setup redirect and demo mode."""

import os
from pathlib import Path
from starlette.responses import RedirectResponse, JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send


class SetupRedirectMiddleware:
    """Redirect all routes to /setup when config.yaml is missing."""

    PASSTHROUGH_PREFIXES = ("/setup", "/static", "/auth")

    def __init__(self, app: ASGIApp, data_dir: Path) -> None:
        self.app = app
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


class DemoModeMiddleware:
    """Block mutation requests in demo mode."""

    BLOCKED_PREFIXES = ("/settings", "/setup")
    BLOCKED_METHODS = {"POST", "PUT", "DELETE", "PATCH"}
    # Allow chat API even in demo mode (rate-limited separately)
    ALLOWED_PATHS = ("/api/chat",)

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    def _is_demo(self) -> bool:
        return os.getenv("NEXUS_DEMO_MODE", "").lower() in ("1", "true", "yes")

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and self._is_demo():
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
