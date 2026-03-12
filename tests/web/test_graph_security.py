"""Tests for SVG graph rendering security."""

import pytest
from httpx import AsyncClient, ASGITransport

from nexus.web.graph import render_entity_network_svg
from nexus.engine.knowledge.store import KnowledgeStore
from nexus.web.app import create_app


def test_entity_names_are_escaped():
    """Entity names with HTML/SVG special chars must be escaped."""
    related = [
        {"id": 1, "canonical_name": '<script>alert("xss")</script>', "co_occurrence_count": 5},
    ]
    svg = render_entity_network_svg("Center", related)
    assert "<script>" not in svg
    assert "&lt;script&gt;" in svg


def test_center_name_is_escaped():
    """Center entity name must also be escaped — no raw HTML tags."""
    related = [
        {"id": 1, "canonical_name": "Normal", "co_occurrence_count": 3},
    ]
    svg = render_entity_network_svg('<img src=x>', related)
    # The raw <img> tag must be escaped, not rendered as HTML
    assert "<img" not in svg
    assert "&lt;img" in svg


def test_normal_names_unchanged():
    """Normal names render without escaping artifacts."""
    related = [
        {"id": 1, "canonical_name": "OpenAI", "co_occurrence_count": 5},
        {"id": 2, "canonical_name": "Google DeepMind", "co_occurrence_count": 3},
    ]
    svg = render_entity_network_svg("Anthropic", related)
    assert "OpenAI" in svg
    assert "Anthropic" in svg
    assert "Google DeepMind" in svg


# ── Route-level XSS tests ──


@pytest.fixture
async def graph_client(tmp_path):
    """Async HTTP client for graph route tests."""
    db_path = tmp_path / "test.db"
    (tmp_path / "config.yaml").write_text("preset: balanced\ntopics:\n  - name: test\n")
    app = create_app(db_path, data_dir=tmp_path)
    store = KnowledgeStore(db_path)
    await store.initialize()
    app.state.store = store
    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
    await store.close()


async def test_focus_param_js_injection_blocked(graph_client):
    """JS injection via focus query param must be neutralized."""
    resp = await graph_client.get("/explore/graph?focus=1);alert(1);//")
    assert resp.status_code == 200
    # The injected JS must NOT appear in the response
    assert "alert(1)" not in resp.text


async def test_focus_param_valid_integer(graph_client):
    """Valid integer focus param should pass through."""
    resp = await graph_client.get("/explore/graph?focus=42")
    assert resp.status_code == 200
    assert "initGraph(42)" in resp.text


async def test_focus_param_empty(graph_client):
    """Empty focus param should render null."""
    resp = await graph_client.get("/explore/graph")
    assert resp.status_code == 200
    assert "initGraph(null)" in resp.text
