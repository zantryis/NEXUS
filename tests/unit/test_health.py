"""Tests for runtime health reporting."""

from datetime import datetime, timedelta, timezone

import pytest

from nexus.config.models import (
    NexusConfig,
    TopicConfig,
    UserConfig,
    TelegramConfig,
    ModelsConfig,
    AudioConfig,
)
from nexus.engine.knowledge.store import KnowledgeStore
from nexus.utils.health import build_health_snapshot, health_summary_lines


@pytest.fixture
async def store(tmp_path):
    db = tmp_path / "test.db"
    store = KnowledgeStore(db)
    await store.initialize()
    yield store
    await store.close()


@pytest.mark.asyncio
async def test_build_health_snapshot_flags_missing_audio_and_topic_coverage(store, tmp_path, monkeypatch):
    briefing_dir = tmp_path / "artifacts" / "briefings"
    briefing_dir.mkdir(parents=True, exist_ok=True)
    (briefing_dir / f"{datetime.now(timezone.utc).date().isoformat()}.md").write_text("# briefing")

    config = NexusConfig(
        user=UserConfig(name="Tester"),
        telegram=TelegramConfig(enabled=True, chat_id=None),
        audio=AudioConfig(enabled=True),
        topics=[
            TopicConfig(name="AI"),
            TopicConfig(name="Energy"),
        ],
        models=ModelsConfig(filtering="litellm/gpt", synthesis="litellm/opus"),
    )

    monkeypatch.setenv("LITELLM_PROXY_URL", "https://proxy.example/v1")
    monkeypatch.setenv("LITELLM_PROXY_API_KEY", "secret")
    monkeypatch.setenv("LITELLM_MODEL_GPT", "gpt-5.4")
    monkeypatch.delenv("LITELLM_MODEL_OPUS", raising=False)
    monkeypatch.setenv(
        "LITELLM_PROXY_TOKEN_EXPIRES_AT",
        (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat(),
    )

    run_id = await store.start_pipeline_run(["AI"], trigger="manual")
    await store.complete_pipeline_run(run_id, article_count=1, event_count=2, cost_usd=0.01)

    snapshot = await build_health_snapshot(config, tmp_path, store)

    assert snapshot["status"] == "critical"
    assert snapshot["deliverables"]["briefing_today"] is True
    assert snapshot["deliverables"]["audio_today"] is False
    assert snapshot["pipeline"]["missing_topics"] == ["Energy"]
    messages = [issue["message"] for issue in snapshot["issues"]]
    assert any("podcast audio is missing" in message for message in messages)
    assert any("covered 1/2 topics" in message for message in messages)
    assert any("expires soon" in message for message in messages)
    assert any("LITELLM_MODEL_OPUS" in message for message in messages)


@pytest.mark.asyncio
async def test_build_health_snapshot_ok_when_outputs_and_aliases_present(store, tmp_path, monkeypatch):
    today = datetime.now(timezone.utc).date().isoformat()
    briefing_dir = tmp_path / "artifacts" / "briefings"
    audio_dir = tmp_path / "artifacts" / "audio"
    briefing_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)
    (briefing_dir / f"{today}.md").write_text("# briefing")
    (audio_dir / f"{today}.mp3").write_bytes(b"audio")

    config = NexusConfig(
        user=UserConfig(name="Tester"),
        telegram=TelegramConfig(enabled=True, chat_id=123),
        topics=[TopicConfig(name="AI")],
        models=ModelsConfig(filtering="litellm/gpt", synthesis="litellm/opus"),
    )

    monkeypatch.setenv("LITELLM_PROXY_URL", "https://proxy.example/v1")
    monkeypatch.setenv("LITELLM_PROXY_API_KEY", "secret")
    monkeypatch.setenv("LITELLM_MODEL_GPT", "gpt-5.4")
    monkeypatch.setenv("LITELLM_MODEL_OPUS", "opus-4.6")
    monkeypatch.setenv(
        "LITELLM_PROXY_TOKEN_EXPIRES_AT",
        (datetime.now(timezone.utc) + timedelta(minutes=45)).isoformat(),
    )

    run_id = await store.start_pipeline_run(["AI"], trigger="manual")
    await store.complete_pipeline_run(run_id, article_count=4, event_count=5, cost_usd=0.02)

    snapshot = await build_health_snapshot(config, tmp_path, store)

    assert snapshot["status"] == "ok"
    assert snapshot["pipeline"]["missing_topics"] == []
    assert snapshot["litellm"]["alias_targets"]["filtering"] == "gpt-5.4"
    assert snapshot["litellm"]["alias_targets"]["synthesis"] == "opus-4.6"

    lines = health_summary_lines(snapshot)
    assert any("System health" in line for line in lines)
    assert any("Last run: completed" in line for line in lines)
