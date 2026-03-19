"""Tests for Kalshi live odds refresh."""

import pytest
from unittest.mock import AsyncMock

from nexus.engine.projection.kalshi_odds import refresh_kalshi_odds, _snapshot_from_market


def test_snapshot_from_market_extracts_fields():
    """_snapshot_from_market should extract relevant fields from API response."""
    market = {
        "last_price": 0.62,
        "yes_bid": 0.61,
        "yes_ask": 0.63,
        "no_bid": 0.37,
        "no_ask": 0.39,
        "volume": 1000,
        "open_interest": 500,
        "status": "open",
    }
    snap = _snapshot_from_market(market)
    assert snap["implied_probability"] == 0.62
    assert snap["yes_bid"] == 0.61
    assert snap["volume"] == 1000
    assert "captured_at" in snap


@pytest.mark.asyncio
async def test_refresh_no_kalshi_forecasts():
    """refresh_kalshi_odds should return early if no Kalshi-aligned forecasts."""
    store = AsyncMock()
    store.get_open_forecasts = AsyncMock(return_value=[
        {"forecast_question_id": 1, "external_ref": None, "probability": 0.5},
    ])
    result = await refresh_kalshi_odds(store, AsyncMock(), AsyncMock())
    assert result["markets_refreshed"] == 0
    assert result["skipped"] == 0


@pytest.mark.asyncio
async def test_refresh_fetches_and_inserts_snapshot():
    """refresh_kalshi_odds should fetch market data and insert a snapshot."""
    store = AsyncMock()
    store.get_open_forecasts = AsyncMock(return_value=[
        {
            "forecast_question_id": 42,
            "external_ref": "TICKER-A",
            "probability": 0.65,
            "topic_slug": "ai",
        },
    ])
    store.update_forecast_probability = AsyncMock()

    kalshi_client = AsyncMock()
    kalshi_client.fetch_market = AsyncMock(return_value={
        "last_price": 0.72,
        "status": "open",
        "volume": 500,
    })

    ledger = AsyncMock()
    ledger.insert_snapshot = AsyncMock()

    result = await refresh_kalshi_odds(store, kalshi_client, ledger)

    assert result["markets_refreshed"] == 1
    assert result["snapshots_inserted"] == 1
    kalshi_client.fetch_market.assert_called_once_with("TICKER-A")
    ledger.insert_snapshot.assert_called_once()

    # Verify base_rate update on the forecast question
    store.update_forecast_probability.assert_called_once()
    call_kw = store.update_forecast_probability.call_args[1]
    assert call_kw["market_probability"] == pytest.approx(0.72)
    assert call_kw["source"] == "kalshi_odds_refresh"


@pytest.mark.asyncio
async def test_refresh_deduplicates_tickers():
    """Multiple questions with same ticker should only fetch once."""
    store = AsyncMock()
    store.get_open_forecasts = AsyncMock(return_value=[
        {"forecast_question_id": 1, "external_ref": "TICKER-A", "probability": 0.65, "topic_slug": "ai"},
        {"forecast_question_id": 2, "external_ref": "TICKER-A", "probability": 0.60, "topic_slug": "ai"},
    ])
    store.update_forecast_probability = AsyncMock()

    kalshi_client = AsyncMock()
    kalshi_client.fetch_market = AsyncMock(return_value={"last_price": 0.72, "status": "open"})
    ledger = AsyncMock()

    result = await refresh_kalshi_odds(store, kalshi_client, ledger)

    assert result["markets_refreshed"] == 1
    kalshi_client.fetch_market.assert_called_once_with("TICKER-A")
    # Both questions should get updated
    assert store.update_forecast_probability.call_count == 2


@pytest.mark.asyncio
async def test_refresh_handles_api_error():
    """API errors for individual tickers should not crash the whole refresh."""
    store = AsyncMock()
    store.get_open_forecasts = AsyncMock(return_value=[
        {"forecast_question_id": 1, "external_ref": "GOOD-TICKER", "probability": 0.65, "topic_slug": "ai"},
        {"forecast_question_id": 2, "external_ref": "BAD-TICKER", "probability": 0.50, "topic_slug": "ai"},
    ])
    store.update_forecast_probability = AsyncMock()

    kalshi_client = AsyncMock()

    async def _fetch(ticker):
        if ticker == "BAD-TICKER":
            raise RuntimeError("API error")
        return {"last_price": 0.72, "status": "open"}

    kalshi_client.fetch_market = AsyncMock(side_effect=_fetch)
    ledger = AsyncMock()

    result = await refresh_kalshi_odds(store, kalshi_client, ledger)
    assert result["markets_refreshed"] == 1
    assert result["errors"] == 1
