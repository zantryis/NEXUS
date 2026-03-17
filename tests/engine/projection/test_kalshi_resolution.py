"""Tests for Kalshi forecast resolution — market-based ground truth scoring."""

from __future__ import annotations

from datetime import date
from unittest.mock import AsyncMock

import pytest

from nexus.engine.knowledge.store import KnowledgeStore
from nexus.engine.projection.kalshi_resolution import (
    resolve_kalshi_forecasts,
    kalshi_scoring_report,
)
from nexus.engine.projection.models import ForecastQuestion, ForecastRun


@pytest.fixture
async def store(tmp_path):
    s = KnowledgeStore(tmp_path / "kalshi-res.db")
    await s.initialize()
    yield s
    await s.close()


def _make_kalshi_client(market_results: dict[str, dict]):
    """Mock KalshiClient that returns preset results by ticker."""
    client = AsyncMock()

    async def mock_fetch(ticker):
        if ticker in market_results:
            return market_results[ticker]
        raise ValueError(f"Unknown ticker: {ticker}")

    client.fetch_market = mock_fetch
    return client


def _make_kalshi_ledger():
    """Mock KalshiLedger."""
    ledger = AsyncMock()
    ledger.get_nearest_snapshot = AsyncMock(return_value=None)
    return ledger


async def _seed_kalshi_forecast(
    store: KnowledgeStore,
    *,
    ticker: str,
    our_prob: float,
    market_prob: float,
    run_date: date = date(2026, 3, 10),
) -> int:
    """Seed a Kalshi-aligned forecast run and return the run_id."""
    run = ForecastRun(
        topic_slug="kalshi-aligned",
        topic_name="Kalshi Market Alignment",
        engine="actor",
        generated_for=run_date,
        summary=f"Kalshi forecast for {ticker}",
        questions=[
            ForecastQuestion(
                question=f"Will {ticker} resolve YES?",
                forecast_type="binary",
                target_variable="kalshi_aligned",
                probability=our_prob,
                base_rate=market_prob,
                resolution_criteria=f"Kalshi market {ticker} resolution",
                resolution_date=run_date + __import__("datetime").timedelta(days=14),
                horizon_days=14,
                signpost=f"Kalshi market: {ticker}",
                external_ref=ticker,
                target_metadata={
                    "kalshi_ticker": ticker,
                    "kalshi_implied": market_prob,
                },
                signals_cited=[
                    f"kalshi:implied={market_prob:.3f}",
                    f"kalshi:our={our_prob:.3f}",
                ],
            ),
        ],
        metadata={"kalshi_aligned": True},
    )
    return await store.save_forecast_run(run)


class TestResolveKalshiForecasts:
    async def test_resolves_settled_yes(self, store):
        """Should resolve a YES-settled market correctly."""
        await _seed_kalshi_forecast(store, ticker="TRUMP-WIN", our_prob=0.70, market_prob=0.60)
        client = _make_kalshi_client({
            "TRUMP-WIN": {"ticker": "TRUMP-WIN", "status": "settled", "result": "yes"},
        })
        result = await resolve_kalshi_forecasts(store, client, as_of=date(2026, 3, 30))
        assert result["resolved"] >= 1
        assert result["mean_brier"] is not None
        # Our prob 0.70, outcome YES=1.0, brier = (0.70 - 1.0)^2 = 0.09
        assert result["brier_scores"][0] == pytest.approx(0.09, abs=0.01)

    async def test_resolves_settled_no(self, store):
        """Should resolve a NO-settled market correctly."""
        await _seed_kalshi_forecast(store, ticker="AI-BAN", our_prob=0.80, market_prob=0.75)
        client = _make_kalshi_client({
            "AI-BAN": {"ticker": "AI-BAN", "status": "settled", "result": "no"},
        })
        result = await resolve_kalshi_forecasts(store, client, as_of=date(2026, 3, 30))
        assert result["resolved"] >= 1
        # Our prob 0.80, outcome NO=0.0, brier = (0.80 - 0.0)^2 = 0.64
        assert result["brier_scores"][0] == pytest.approx(0.64, abs=0.01)

    async def test_skips_open_markets(self, store):
        """Should not resolve markets that are still open."""
        await _seed_kalshi_forecast(store, ticker="OPEN-MKT", our_prob=0.55, market_prob=0.50)
        client = _make_kalshi_client({
            "OPEN-MKT": {"ticker": "OPEN-MKT", "status": "open"},
        })
        result = await resolve_kalshi_forecasts(store, client, as_of=date(2026, 3, 30))
        assert result["resolved"] == 0
        assert result["still_open"] >= 1

    async def test_computes_brier(self, store):
        """Brier score should be (our_prob - outcome)^2."""
        await _seed_kalshi_forecast(store, ticker="TEST-BRIER", our_prob=0.60, market_prob=0.50)
        client = _make_kalshi_client({
            "TEST-BRIER": {"ticker": "TEST-BRIER", "status": "settled", "result": "yes"},
        })
        result = await resolve_kalshi_forecasts(store, client, as_of=date(2026, 3, 30))
        # (0.60 - 1.0)^2 = 0.16
        assert result["brier_scores"][0] == pytest.approx(0.16, abs=0.01)

    async def test_saves_resolution(self, store):
        """Should persist the resolution to the store."""
        await _seed_kalshi_forecast(store, ticker="SAVE-TEST", our_prob=0.65, market_prob=0.55)
        client = _make_kalshi_client({
            "SAVE-TEST": {"ticker": "SAVE-TEST", "status": "settled", "result": "yes"},
        })
        await resolve_kalshi_forecasts(store, client, as_of=date(2026, 3, 30))
        # Check that resolution was saved by querying pending questions
        pending = await store.get_pending_forecast_questions(until=date(2026, 3, 30))
        # The resolved question should no longer be pending
        ticker_questions = [
            q for q in pending
            if (q.get("target_metadata") or {}).get("kalshi_ticker") == "SAVE-TEST"
        ]
        assert len(ticker_questions) == 0


class TestKalshiScoringReport:
    async def test_aggregates_brier(self, store):
        """Should aggregate Brier scores across resolved questions."""
        # Seed two resolved forecasts
        await _seed_kalshi_forecast(store, ticker="RPT-A", our_prob=0.70, market_prob=0.60)
        await _seed_kalshi_forecast(store, ticker="RPT-B", our_prob=0.30, market_prob=0.40)
        client = _make_kalshi_client({
            "RPT-A": {"ticker": "RPT-A", "status": "settled", "result": "yes"},
            "RPT-B": {"ticker": "RPT-B", "status": "settled", "result": "no"},
        })
        await resolve_kalshi_forecasts(store, client, as_of=date(2026, 3, 30))

        report = await kalshi_scoring_report(
            store,
            start=date(2026, 3, 1),
            end=date(2026, 3, 31),
        )
        assert report["total_resolved"] >= 2
        assert "our_mean_brier" in report
        assert "market_mean_brier" in report
        assert isinstance(report["our_mean_brier"], float)

    async def test_compares_us_to_market(self, store):
        """Should include both our Brier and market Brier for comparison."""
        await _seed_kalshi_forecast(store, ticker="CMP-A", our_prob=0.80, market_prob=0.50)
        client = _make_kalshi_client({
            "CMP-A": {"ticker": "CMP-A", "status": "settled", "result": "yes"},
        })
        await resolve_kalshi_forecasts(store, client, as_of=date(2026, 3, 30))

        report = await kalshi_scoring_report(
            store, start=date(2026, 3, 1), end=date(2026, 3, 31),
        )
        assert "our_mean_brier" in report
        assert "market_mean_brier" in report
        # Our prob 0.80 for YES → brier 0.04
        # Market prob 0.50 for YES → brier 0.25
        # We should be better here
        assert report["our_mean_brier"] < report["market_mean_brier"]
