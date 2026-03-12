"""Tests for the web chat widget API."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

import yaml
from httpx import AsyncClient, ASGITransport

from nexus.engine.knowledge.store import KnowledgeStore
from nexus.config.loader import load_config
from nexus.web.app import create_app


@pytest.fixture
async def chat_app(tmp_path):
    """App with config, store, llm, and config for chat testing."""
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
    app.state.llm = MagicMock()  # mock LLM client
    return app


@pytest.mark.asyncio
@patch("nexus.agent.qa.answer_question", new_callable=AsyncMock)
async def test_chat_returns_answer(mock_qa, chat_app):
    """POST /api/chat with a question returns an answer."""
    mock_qa.return_value = "The answer is 42."
    transport = ASGITransport(app=chat_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/api/chat", json={"question": "What is the meaning?"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["answer"] == "The answer is 42."
        assert "remaining" in data


@pytest.mark.asyncio
async def test_chat_empty_question(chat_app):
    """POST /api/chat with empty question returns 400."""
    transport = ASGITransport(app=chat_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/api/chat", json={"question": ""})
        assert resp.status_code == 400


@pytest.mark.asyncio
async def test_chat_no_llm_returns_503(chat_app):
    """POST /api/chat without LLM initialized returns 503."""
    del chat_app.state.llm
    transport = ASGITransport(app=chat_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/api/chat", json={"question": "Hello?"})
        assert resp.status_code == 503


@pytest.mark.asyncio
@patch("nexus.agent.qa.answer_question", new_callable=AsyncMock)
async def test_chat_rate_limit(mock_qa, chat_app):
    """After 5 requests, the 6th returns 429."""
    mock_qa.return_value = "Answer."
    transport = ASGITransport(app=chat_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        for i in range(5):
            resp = await client.post("/api/chat", json={"question": f"Q{i}"})
            assert resp.status_code == 200
            remaining = resp.json()["remaining"]
            assert remaining == 4 - i

        # 6th request should be rate-limited
        resp = await client.post("/api/chat", json={"question": "One more?"})
        assert resp.status_code == 429
        assert resp.json()["remaining"] == 0


@pytest.mark.asyncio
@patch("nexus.agent.qa.answer_question", new_callable=AsyncMock)
async def test_chat_qa_error_returns_500(mock_qa, chat_app):
    """If Q&A pipeline raises, return 500."""
    mock_qa.side_effect = RuntimeError("LLM exploded")
    transport = ASGITransport(app=chat_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/api/chat", json={"question": "Boom?"})
        assert resp.status_code == 500
        assert "Failed" in resp.json()["error"]
