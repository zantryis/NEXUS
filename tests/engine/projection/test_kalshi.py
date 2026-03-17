"""Tests for Kalshi benchmark ledger and comparison helpers."""

from datetime import date, datetime, time, timezone
import json

import pytest

from nexus.config.models import KalshiBenchmarkConfig
from nexus.engine.knowledge.store import KnowledgeStore
from nexus.engine.projection.kalshi import (
    KalshiClient,
    KalshiLedger,
    bootstrap_kalshi_credentials,
    compare_forecasts_to_kalshi,
    load_kalshi_credentials,
    parse_kalshi_cred_file,
    sync_kalshi_tickers,
)
from nexus.engine.projection.models import ForecastQuestion, ForecastResolution, ForecastRun


@pytest.fixture
async def store(tmp_path):
    s = KnowledgeStore(tmp_path / "kalshi-forecast.db")
    await s.initialize()
    yield s
    await s.close()


async def test_sync_kalshi_tickers_populates_separate_ledger(tmp_path):
    class FakeClient:
        async def fetch_market(self, ticker: str) -> dict:
            return {
                "ticker": ticker,
                "event_ticker": "EVT",
                "series_ticker": "SERIES",
                "title": "Test market",
                "status": "open",
                "last_price": 61,
            }

        async def fetch_candlesticks(self, *, series_ticker: str, ticker: str, start: date, end: date) -> list[dict]:
            return [
                {
                    "end_ts": int(datetime.combine(end, time.max, tzinfo=timezone.utc).timestamp()),
                    "close": 0.61,
                    "volume": 10,
                }
            ]

    ledger = KalshiLedger(tmp_path / "kalshi.sqlite")
    await ledger.initialize()
    try:
        report = await sync_kalshi_tickers(
            ledger,
            FakeClient(),
            tickers=["TEST-001"],
            start=date(2026, 3, 8),
            end=date(2026, 3, 8),
        )
        assert report["tickers_synced"] == 1
        assert report["snapshots_inserted"] == 1
        snapshot = await ledger.get_nearest_snapshot(
            "TEST-001",
            captured_at=datetime.combine(date(2026, 3, 8), time.max, tzinfo=timezone.utc),
        )
        assert snapshot is not None
        assert snapshot["implied_probability"] == 0.61
    finally:
        await ledger.close()


async def test_compare_forecasts_to_kalshi_uses_forecast_key_mapping(store, tmp_path):
    forecast_run = ForecastRun(
        topic_slug="iran-us",
        topic_name="Iran-US",
        engine="native",
        generated_for=date(2026, 3, 8),
        summary="Test run",
        questions=[ForecastQuestion(
            question="Will sanctions pressure continue?",
            forecast_type="binary",
            target_variable="entity_recurrence",
            target_metadata={"topic_slug": "iran-us", "entity": "Iran"},
            probability=0.62,
            base_rate=0.5,
            resolution_criteria="Resolves true if Iran is mentioned again.",
            resolution_date=date(2026, 3, 10),
            horizon_days=3,
            signpost="Watch for new sanctions reporting",
            expected_direction=None,
        )],
    )
    await store.save_forecast_run(forecast_run)
    question = (await store.get_latest_forecast_run("iran-us")).questions[0]
    await store.set_forecast_resolution(ForecastResolution(
        forecast_question_id=question.question_id,
        outcome_status="resolved",
        resolved_bool=True,
        realized_direction="up",
        actual_value=1.0,
        brier_score=0.0,
        log_loss=0.0,
        resolved_at=date(2026, 3, 10),
    ))

    ledger = KalshiLedger(tmp_path / "kalshi.sqlite")
    await ledger.initialize()
    try:
        await ledger.upsert_market({
            "ticker": "TEST-001",
            "event_ticker": "EVT",
            "series_ticker": "SERIES",
            "title": "Test market",
            "status": "open",
        })
        await ledger.insert_snapshot(
            "TEST-001",
            {
                "captured_at": datetime.combine(date(2026, 3, 8), time.max, tzinfo=timezone.utc).isoformat(),
                "implied_probability": 0.57,
                "last_price": 57,
                "status": "open",
                "raw": {},
            },
        )
        mapping_path = tmp_path / "kalshi_mappings.json"
        mapping_path.write_text(json.dumps({
            "mappings": [
                {
                    "forecast_key": question.forecast_key,
                    "market_ticker": "TEST-001",
                    "side": "yes",
                }
            ]
        }))
        report = await compare_forecasts_to_kalshi(
            store,
            ledger,
            start=date(2026, 3, 8),
            end=date(2026, 3, 8),
            mapping_path=mapping_path,
        )
        assert report["meta"]["mapped_forecasts"] == 1
        assert report["rows"][0]["market_ticker"] == "TEST-001"
        assert report["rows"][0]["market_probability"] == 0.57
    finally:
        await ledger.close()


def test_load_kalshi_credentials_from_env(monkeypatch, tmp_path):
    key_path = tmp_path / "kalshi.pem"
    key_path.write_text("-----BEGIN PRIVATE KEY-----\nTEST\n-----END PRIVATE KEY-----\n")
    config = KalshiBenchmarkConfig()
    monkeypatch.setenv(config.api_key_id_env, "kid")
    monkeypatch.setenv(config.private_key_path_env, str(key_path))

    credentials = load_kalshi_credentials(config)
    assert credentials is not None
    assert credentials.key_id == "kid"
    assert "BEGIN PRIVATE KEY" in credentials.private_key_pem


def test_parse_kalshi_cred_file_reads_key_id_and_pem(tmp_path):
    cred_path = tmp_path / "kalshi-cred"
    cred_path.write_text(
        "kalshi\n"
        "kid-123\n"
        "-----BEGIN PRIVATE KEY-----\n"
        "TEST\n"
        "-----END PRIVATE KEY-----\n"
    )

    credentials = parse_kalshi_cred_file(cred_path)

    assert credentials.key_id == "kid-123"
    assert "BEGIN PRIVATE KEY" in credentials.private_key_pem


def test_bootstrap_kalshi_credentials_writes_local_key_file(tmp_path):
    cred_path = tmp_path / "kalshi-cred"
    cred_path.write_text(
        "kalshi\n"
        "kid-123\n"
        "-----BEGIN PRIVATE KEY-----\n"
        "TEST\n"
        "-----END PRIVATE KEY-----\n"
    )
    target_key_path = tmp_path / ".secrets" / "kalshi_private_key.pem"

    result = bootstrap_kalshi_credentials(
        cred_path=cred_path,
        config=KalshiBenchmarkConfig(),
        target_key_path=target_key_path,
    )

    assert target_key_path.exists()
    assert "BEGIN PRIVATE KEY" in target_key_path.read_text()
    assert result["env"]["KALSHI_API_KEY_ID"] == "kid-123"
    assert result["env"]["KALSHI_PRIVATE_KEY_PATH"] == str(target_key_path)


async def test_kalshi_auth_check_reports_missing_creds(monkeypatch):
    monkeypatch.delenv("KALSHI_API_KEY_ID", raising=False)
    monkeypatch.delenv("KALSHI_PRIVATE_KEY_PATH", raising=False)
    client = KalshiClient(KalshiBenchmarkConfig())

    status = await client.auth_check()

    assert status["configured"] is False
    assert status["auth_capable"] is False
    assert status["auth_success"] is False


async def test_kalshi_auth_check_reports_success(monkeypatch, tmp_path):
    key_path = tmp_path / "kalshi.pem"
    key_path.write_text("-----BEGIN PRIVATE KEY-----\nTEST\n-----END PRIVATE KEY-----\n")
    config = KalshiBenchmarkConfig()
    monkeypatch.setenv(config.api_key_id_env, "kid")
    monkeypatch.setenv(config.private_key_path_env, str(key_path))

    async def fake_request_json(self, method: str, path: str, *, params=None, auth_required: bool = False):
        assert method == "GET"
        assert path == "/portfolio/balance"
        assert auth_required is True
        return {"balance": {"cash": "100"}}

    monkeypatch.setattr(KalshiClient, "_request_json", fake_request_json)

    client = KalshiClient(config)
    status = await client.auth_check()

    assert status["configured"] is True
    assert status["auth_capable"] is True
    assert status["auth_success"] is True
    assert status["error"] is None


async def test_list_events_returns_paginated_results(monkeypatch):
    """list_events should return events from the public API."""
    config = KalshiBenchmarkConfig()
    call_log = []

    async def fake_request_json(self, method: str, path: str, *, params=None, auth_required: bool = False):
        call_log.append({"method": method, "path": path, "params": params})
        return {
            "events": [
                {"title": "Will X happen?", "category": "Politics", "markets": [{"ticker": "X-YES"}]},
                {"title": "Will Y happen?", "category": "Economics", "markets": []},
            ],
            "cursor": None,
        }

    monkeypatch.setattr(KalshiClient, "_request_json", fake_request_json)

    client = KalshiClient(config)
    result = await client.list_events(status="open", limit=100)

    assert len(result["events"]) == 2
    assert result["events"][0]["title"] == "Will X happen?"
    assert call_log[0]["params"]["status"] == "open"
    assert call_log[0]["params"]["limit"] == 100
    assert call_log[0]["params"]["with_nested_markets"] == "true"
