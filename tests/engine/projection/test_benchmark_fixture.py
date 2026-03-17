"""Tests for benchmark fixture additions: derive_series_ticker, ActorBenchmarkEngine,
StructuralAssessment.numeric_probability, MultiSampleWrapper."""

import asyncio
from datetime import date
from unittest.mock import AsyncMock

import pytest

from nexus.engine.projection.kalshi import derive_series_ticker
from nexus.engine.projection.models import StructuralAssessment


# ── derive_series_ticker ─────────────────────────────────────────────


def test_derive_series_ticker_splits_on_last_hyphen():
    assert derive_series_ticker("KXNEXTPOPE-35") == "KXNEXTPOPE"


def test_derive_series_ticker_multi_hyphen():
    assert derive_series_ticker("KX-NEXT-POPE-35") == "KX-NEXT-POPE"


def test_derive_series_ticker_no_hyphen():
    assert derive_series_ticker("KXNEXTPOPE") == "KXNEXTPOPE"


def test_derive_series_ticker_empty():
    assert derive_series_ticker("") == ""


def test_derive_series_ticker_single_hyphen():
    assert derive_series_ticker("A-B") == "A"


# ── StructuralAssessment.numeric_probability ─────────────────────────


def test_implied_probability_uses_table_when_no_numeric():
    sa = StructuralAssessment(
        question="test", verdict="yes", confidence="high",
    )
    assert sa.numeric_probability is None
    assert sa.implied_probability == 0.92


def test_implied_probability_uses_numeric_when_consistent():
    sa = StructuralAssessment(
        question="test", verdict="yes", confidence="high",
        numeric_probability=0.85,
    )
    assert sa.implied_probability == 0.85


def test_implied_probability_averages_when_inconsistent():
    # verdict=no but numeric=0.7 → inconsistent
    sa = StructuralAssessment(
        question="test", verdict="no", confidence="medium",
        numeric_probability=0.70,
    )
    # table_prob for (no, medium) = 0.25
    # (0.25 + 0.70) / 2 = 0.475
    assert sa.implied_probability == 0.475


def test_implied_probability_yes_low_with_consistent_numeric():
    sa = StructuralAssessment(
        question="test", verdict="yes", confidence="low",
        numeric_probability=0.55,
    )
    assert sa.implied_probability == 0.55


def test_implied_probability_no_high_with_consistent_numeric():
    sa = StructuralAssessment(
        question="test", verdict="no", confidence="high",
        numeric_probability=0.12,
    )
    assert sa.implied_probability == 0.12


def test_implied_probability_yes_with_low_numeric_triggers_average():
    # verdict=yes but numeric=0.30 → inconsistent (< 0.4 threshold)
    sa = StructuralAssessment(
        question="test", verdict="yes", confidence="medium",
        numeric_probability=0.30,
    )
    # table_prob for (yes, medium) = 0.75
    # (0.75 + 0.30) / 2 = 0.525
    assert sa.implied_probability == 0.525


def test_implied_probability_uncertain_with_numeric():
    sa = StructuralAssessment(
        question="test", verdict="uncertain", confidence="low",
        numeric_probability=0.45,
    )
    # uncertain doesn't trigger inconsistency check, just uses numeric
    assert sa.implied_probability == 0.45


# ── ActorBenchmarkEngine ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_actor_benchmark_engine_returns_050_without_llm():
    from nexus.engine.projection.actor_engine import ActorBenchmarkEngine

    engine = ActorBenchmarkEngine()
    assert engine.engine_name == "actor"
    result = await engine.predict_probability("Will X happen?")
    assert result == 0.50


@pytest.mark.asyncio
async def test_actor_benchmark_engine_returns_050_without_store():
    from nexus.engine.projection.actor_engine import ActorBenchmarkEngine

    engine = ActorBenchmarkEngine()
    result = await engine.predict_probability("Will X happen?", llm=object())
    assert result == 0.50


# ── MultiSampleWrapper ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_multi_sample_wrapper_returns_median():
    # noinspection PyUnresolvedReferences
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "scripts"))

    from build_kalshi_fixture import MultiSampleWrapper

    class FakeEngine:
        engine_name = "fake"
        call_count = 0

        async def predict_probability(self, question, **kwargs):
            self.call_count += 1
            # Return different values on each call
            return [0.3, 0.7, 0.5][self.call_count - 1]

    fake = FakeEngine()
    wrapper = MultiSampleWrapper(fake, n=3)
    assert wrapper.engine_name == "fake_x3"

    result = await wrapper.predict_probability("test")
    assert fake.call_count == 3
    # median of [0.3, 0.7, 0.5] sorted = [0.3, 0.5, 0.7] → 0.5
    assert result == 0.5


# ── sync_kalshi_tickers with derived series ──────────────────────────


@pytest.mark.asyncio
async def test_sync_derives_series_from_event_ticker(tmp_path):
    """When market payload has no series_ticker, derive it from event_ticker."""
    from datetime import datetime, time, timezone
    from nexus.engine.projection.kalshi import KalshiLedger, sync_kalshi_tickers

    class FakeClient:
        async def fetch_market(self, ticker: str) -> dict:
            return {
                "ticker": ticker,
                "event_ticker": "KXTEST-42",
                "series_ticker": "",  # empty — should trigger derivation
                "title": "Test",
                "status": "open",
                "last_price": 55,
            }

        async def fetch_candlesticks(self, *, series_ticker: str, ticker: str, start, end) -> list[dict]:
            assert series_ticker == "KXTEST", f"Expected derived series KXTEST, got {series_ticker}"
            return [{
                "end_ts": int(datetime.combine(end, time.max, tzinfo=timezone.utc).timestamp()),
                "close": 0.55,
                "volume": 5,
            }]

    ledger = KalshiLedger(tmp_path / "kalshi.sqlite")
    await ledger.initialize()
    try:
        result = await sync_kalshi_tickers(
            ledger, FakeClient(),
            tickers=["T-001"],
            start=date(2026, 3, 1),
            end=date(2026, 3, 15),
        )
        assert result["tickers_synced"] == 1
        assert result["snapshots_inserted"] == 1
    finally:
        await ledger.close()
