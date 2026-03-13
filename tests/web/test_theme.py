"""Tests for theme toggle (light/dark mode)."""

import pytest
from httpx import AsyncClient, ASGITransport

from nexus.engine.knowledge.store import KnowledgeStore
from nexus.web.app import create_app


@pytest.fixture
async def app(tmp_path):
    """Create app with minimal config and injected store."""
    (tmp_path / "config.yaml").write_text("preset: balanced\ntopics:\n  - name: test\n")
    app = create_app(tmp_path / "test.db", data_dir=tmp_path)

    store = KnowledgeStore(tmp_path / "test.db")
    await store.initialize()
    app.state.store = store

    yield app
    await store.close()


async def test_dashboard_has_theme_toggle(app):
    """Dashboard HTML includes theme toggle button and JS."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/")
        assert resp.status_code == 200
        assert "theme-toggle" in resp.text
        assert "toggleTheme" in resp.text


async def test_dashboard_has_flash_prevention(app):
    """Dashboard includes inline script to prevent theme flash on load."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/")
        assert "nexus-theme" in resp.text
        assert "prefers-color-scheme" in resp.text


async def test_css_has_light_theme(app):
    """Static CSS includes light theme variables."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/static/style.css")
        assert resp.status_code == 200
        assert '[data-theme="light"]' in resp.text
        assert "--nx-heading" in resp.text
        assert "--nx-glass" in resp.text
