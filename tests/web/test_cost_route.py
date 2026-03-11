"""Tests for cost dashboard routes."""

import pytest
from datetime import date
from httpx import AsyncClient, ASGITransport

from nexus.engine.knowledge.store import KnowledgeStore
from nexus.web.app import create_app


@pytest.fixture
async def app_with_usage(tmp_path):
    db_path = tmp_path / "test.db"
    app = create_app(db_path)
    store = KnowledgeStore(db_path)
    await store.initialize()

    # Seed usage data
    await store.add_usage_record(
        date.today().isoformat(), "gemini", "gemini-3-flash-preview",
        "filtering", 1000, 500, 0.0005,
    )
    await store.add_usage_record(
        date.today().isoformat(), "deepseek", "deepseek-chat",
        "synthesis", 2000, 1000, 0.0008,
    )

    app.state.store = store
    yield app
    await store.close()


@pytest.fixture
async def client(app_with_usage):
    transport = ASGITransport(app=app_with_usage, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def test_cost_page_returns_200(client):
    resp = await client.get("/cost")
    assert resp.status_code == 200
    assert "Cost Dashboard" in resp.text


async def test_cost_page_shows_data(client):
    resp = await client.get("/cost")
    # Usage summary aggregates by date; check date and cost appear
    assert "Last 30 Days" in resp.text
    assert "3000" in resp.text  # total input tokens


async def test_cost_api_returns_json(client):
    resp = await client.get("/api/cost")
    assert resp.status_code == 200
    data = resp.json()
    assert "today_usd" in data
    assert data["today_usd"] >= 0


async def test_settings_page_returns_200(client):
    resp = await client.get("/settings")
    assert resp.status_code == 200
    assert "Settings" in resp.text
    assert "Gemini" in resp.text
