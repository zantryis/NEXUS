"""FastAPI application factory with KnowledgeStore lifespan."""

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from nexus.engine.knowledge.store import KnowledgeStore


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Open KnowledgeStore on startup, close on shutdown."""
    store = KnowledgeStore(app.state.db_path)
    await store.initialize()
    app.state.store = store

    # Check if knowledge store has any data (for nav visibility)
    try:
        count = await store.db.execute("SELECT COUNT(*) FROM events")
        row = await count.fetchone()
        app.state.has_data = (row[0] > 0) if row else False
    except Exception:
        app.state.has_data = False

    yield
    await store.close()


def create_app(db_path: Path | None = None, data_dir: Path | None = None) -> FastAPI:
    """Build the FastAPI dashboard app."""
    app = FastAPI(title="Nexus Dashboard", lifespan=_lifespan)
    app.state.db_path = db_path or Path("data/knowledge.db")

    # Setup redirect middleware — redirects to /setup when config.yaml is missing
    resolved_data_dir = data_dir or Path("data")
    app.state.data_dir = resolved_data_dir
    from nexus.web.middleware import (
        AdminProtectionMiddleware,
        DemoModeMiddleware,
        SecurityHeadersMiddleware,
        SetupRedirectMiddleware,
    )
    app.add_middleware(SetupRedirectMiddleware, data_dir=resolved_data_dir)
    app.add_middleware(AdminProtectionMiddleware, data_dir=resolved_data_dir)
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(DemoModeMiddleware)

    # Static files and templates
    web_dir = Path(__file__).parent
    app.mount("/static", StaticFiles(directory=web_dir / "static"), name="static")

    templates = Jinja2Templates(directory=web_dir / "templates")
    app.state.templates = templates

    # Register custom Jinja2 filters and globals
    from nexus.web.filters import timeago
    templates.env.filters["timeago"] = timeago
    templates.env.globals["is_demo_mode"] = lambda: os.getenv(
        "NEXUS_DEMO_MODE", ""
    ).lower() in ("1", "true", "yes")
    templates.env.globals["has_data"] = lambda: getattr(app.state, "has_data", False)

    # Register routes
    from nexus.web.routes.dashboard import router as dashboard_router
    from nexus.web.routes.topics import router as topics_router
    from nexus.web.routes.threads import router as threads_router
    from nexus.web.routes.events import router as events_router
    from nexus.web.routes.entities import router as entities_router
    from nexus.web.routes.pages import router as pages_router
    from nexus.web.routes.filters import router as filters_router
    from nexus.web.routes.sources import router as sources_router
    from nexus.web.routes.podcast import router as podcast_router
    from nexus.web.routes.cost import router as cost_router
    from nexus.web.routes.settings import router as settings_router
    from nexus.web.routes.explore import router as explore_router
    from nexus.web.routes.graph import router as graph_router
    from nexus.web.routes.setup import router as setup_router
    from nexus.web.routes.changes import router as changes_router
    from nexus.web.routes.chat import router as chat_router
    from nexus.web.routes.oauth import router as oauth_router

    app.include_router(setup_router)
    app.include_router(chat_router)
    app.include_router(oauth_router)
    app.include_router(dashboard_router)
    app.include_router(topics_router)
    app.include_router(threads_router)
    app.include_router(events_router)
    app.include_router(entities_router)
    app.include_router(pages_router)
    app.include_router(filters_router)
    app.include_router(sources_router)
    app.include_router(podcast_router)
    app.include_router(cost_router)
    app.include_router(settings_router)
    app.include_router(explore_router)
    app.include_router(graph_router)
    app.include_router(changes_router)

    return app


def get_store(request: Request) -> KnowledgeStore:
    """Dependency: extract store from app state."""
    return request.app.state.store


def get_templates(request: Request) -> Jinja2Templates:
    """Dependency: extract Jinja2 templates from app state."""
    return request.app.state.templates
