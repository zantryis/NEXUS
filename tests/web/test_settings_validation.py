"""Tests for settings validation, audio style presets, and restart indicators."""

import pytest
from httpx import AsyncClient, ASGITransport

from nexus.engine.knowledge.store import KnowledgeStore
from nexus.web.app import create_app
from nexus.web.routes.settings import validate_settings


@pytest.fixture(autouse=True)
def _clear_demo_mode(monkeypatch):
    """Ensure NEXUS_DEMO_MODE is not set (may leak from .env via load_dotenv)."""
    monkeypatch.delenv("NEXUS_DEMO_MODE", raising=False)


# ── Unit tests for validation helper ──


class TestValidateSettings:
    def test_valid_settings(self):
        errors = validate_settings(timezone="America/Denver", schedule="06:00")
        assert errors == {}

    def test_valid_utc(self):
        errors = validate_settings(timezone="UTC", schedule="14:30")
        assert errors == {}

    def test_invalid_timezone(self):
        errors = validate_settings(timezone="Mars/Olympus", schedule="06:00")
        assert "timezone" in errors

    def test_empty_timezone_ok(self):
        """Empty timezone defaults to UTC — should not error."""
        errors = validate_settings(timezone="", schedule="06:00")
        assert "timezone" not in errors

    def test_invalid_schedule_25_hours(self):
        errors = validate_settings(timezone="UTC", schedule="25:00")
        assert "schedule" in errors

    def test_invalid_schedule_bad_format(self):
        errors = validate_settings(timezone="UTC", schedule="not-a-time")
        assert "schedule" in errors

    def test_valid_schedule_midnight(self):
        errors = validate_settings(timezone="UTC", schedule="00:00")
        assert "schedule" not in errors

    def test_valid_schedule_2359(self):
        errors = validate_settings(timezone="UTC", schedule="23:59")
        assert "schedule" not in errors


# ── Integration tests for settings save with validation ──


@pytest.fixture
async def app(tmp_path):
    (tmp_path / "config.yaml").write_text(
        "preset: balanced\nuser:\n  name: Test\n  timezone: UTC\ntopics:\n  - name: test\n"
    )
    app = create_app(tmp_path / "test.db", data_dir=tmp_path)
    store = KnowledgeStore(tmp_path / "test.db")
    await store.initialize()
    app.state.store = store
    yield app
    await store.close()


async def test_save_with_invalid_timezone_returns_errors(app):
    """POST /settings/save with bad timezone returns error, does NOT save."""
    transport = ASGITransport(app=app, client=("127.0.0.1", 0))
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/settings/save", data={
            "name": "Test",
            "timezone": "Mars/Olympus",
            "schedule": "06:00",
            "output_language": "en",
            "style": "analytical",
            "depth": "detailed",
            "daily_limit_usd": "1.00",
            "warning_threshold_usd": "0.50",
            "degradation_strategy": "skip_expensive",
        }, follow_redirects=False)
    # Should redirect with error flag, not saved
    assert resp.status_code == 303
    assert "error=validation" in resp.headers.get("location", "")


async def test_save_with_invalid_schedule_returns_errors(app):
    """POST /settings/save with bad schedule returns error."""
    transport = ASGITransport(app=app, client=("127.0.0.1", 0))
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/settings/save", data={
            "name": "Test",
            "timezone": "UTC",
            "schedule": "25:00",
            "output_language": "en",
            "style": "analytical",
            "depth": "detailed",
            "daily_limit_usd": "1.00",
            "warning_threshold_usd": "0.50",
            "degradation_strategy": "skip_expensive",
        }, follow_redirects=False)
    assert resp.status_code == 303
    assert "error=validation" in resp.headers.get("location", "")


async def test_save_valid_settings_succeeds(app):
    """POST /settings/save with valid data saves and redirects."""
    transport = ASGITransport(app=app, client=("127.0.0.1", 0))
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/settings/save", data={
            "name": "Tristan",
            "timezone": "America/Denver",
            "schedule": "07:30",
            "output_language": "en",
            "style": "editorial",
            "depth": "detailed",
            "daily_limit_usd": "2.00",
            "warning_threshold_usd": "1.00",
            "degradation_strategy": "skip_expensive",
        }, follow_redirects=False)
    assert resp.status_code == 303
    assert "saved=all" in resp.headers.get("location", "")


# ── Audio style preset tests ──


async def test_settings_page_shows_podcast_style(app):
    """Settings page renders podcast style selector."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/settings")
    assert resp.status_code == 200
    assert "podcast_style" in resp.text


async def test_save_podcast_style(app):
    """Podcast style is persisted in config."""
    import yaml
    data_dir = app.state.data_dir
    transport = ASGITransport(app=app, client=("127.0.0.1", 0))
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await client.post("/settings/save", data={
            "name": "Test",
            "timezone": "UTC",
            "schedule": "06:00",
            "output_language": "en",
            "style": "analytical",
            "depth": "detailed",
            "podcast_style": "energetic",
            "daily_limit_usd": "1.00",
            "warning_threshold_usd": "0.50",
            "degradation_strategy": "skip_expensive",
        }, follow_redirects=False)
    config = yaml.safe_load((data_dir / "config.yaml").read_text())
    assert config["audio"]["podcast_style"] == "energetic"


# ── Restart indicator tests ──


async def test_settings_page_shows_restart_indicators(app):
    """Settings page shows restart-required indicators on relevant fields."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/settings")
    assert resp.status_code == 200
    assert "restart-required" in resp.text


async def test_validation_errors_shown_on_settings_page(app):
    """Settings page with ?error=validation shows error banner."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/settings?error=validation&fields=timezone")
    assert resp.status_code == 200
    assert "invalid" in resp.text.lower() or "error" in resp.text.lower()
