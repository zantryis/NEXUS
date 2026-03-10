"""FastAPI application factory with KnowledgeStore lifespan."""

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
    yield
    await store.close()


def create_app(db_path: Path | None = None) -> FastAPI:
    """Build the FastAPI dashboard app."""
    app = FastAPI(title="Nexus Dashboard", lifespan=_lifespan)
    app.state.db_path = db_path or Path("data/knowledge.db")

    # Static files and templates
    web_dir = Path(__file__).parent
    app.mount("/static", StaticFiles(directory=web_dir / "static"), name="static")

    templates = Jinja2Templates(directory=web_dir / "templates")
    app.state.templates = templates

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

    app.include_router(dashboard_router)
    app.include_router(topics_router)
    app.include_router(threads_router)
    app.include_router(events_router)
    app.include_router(entities_router)
    app.include_router(pages_router)
    app.include_router(filters_router)
    app.include_router(sources_router)
    app.include_router(podcast_router)

    return app


def get_store(request: Request) -> KnowledgeStore:
    """Dependency: extract store from app state."""
    return request.app.state.store


def get_templates(request: Request) -> Jinja2Templates:
    """Dependency: extract Jinja2 templates from app state."""
    return request.app.state.templates
