"""Tests for Kalshi benchmark — settled market discovery, dataset building, harness."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pytest

from nexus.engine.projection.kalshi_benchmark import (
    BenchmarkQuestion,
    SettledMarket,
    build_benchmark_dataset,
    build_benchmark_from_metadata,
    discover_settled_markets,
    run_benchmark,
)


# ── Fixtures ──────────────────────────────────────────────────────────


def _mock_settled_events():
    """Simulated Kalshi response with settled events and nested markets."""
    return {
        "events": [
            {
                "event_ticker": "OSCAR-BEST-PIC-2026",
                "title": "2026 Oscars Best Picture",
                "category": "entertainment",
                "markets": [
                    {
                        "ticker": "OSCAR-BP-2026-ANORA",
                        "event_ticker": "OSCAR-BEST-PIC-2026",
                        "series_ticker": "OSCAR-BP",
                        "title": "Anora wins Best Picture at 2026 Oscars",
                        "subtitle": "Resolves Yes if Anora wins",
                        "status": "settled",
                        "result": "yes",
                        "close_time": "2026-03-02T23:00:00Z",
                        "last_price": 0.72,
                        "yes_bid": 0.70,
                        "yes_ask": 0.74,
                        "volume": 52000,
                    },
                    {
                        "ticker": "OSCAR-BP-2026-BRUTALIST",
                        "event_ticker": "OSCAR-BEST-PIC-2026",
                        "series_ticker": "OSCAR-BP",
                        "title": "The Brutalist wins Best Picture at 2026 Oscars",
                        "subtitle": "Resolves Yes if The Brutalist wins",
                        "status": "settled",
                        "result": "no",
                        "close_time": "2026-03-02T23:00:00Z",
                        "last_price": 0.15,
                        "volume": 31000,
                    },
                ],
            },
            {
                "event_ticker": "IRAN-STRIKE-2026",
                "title": "Iran military strike before April 2026",
                "category": "world",
                "markets": [
                    {
                        "ticker": "IRAN-STRIKE-APR",
                        "event_ticker": "IRAN-STRIKE-2026",
                        "series_ticker": "IRAN-STRIKE",
                        "title": "Iran conducts military strike before April 1, 2026",
                        "subtitle": "",
                        "status": "settled",
                        "result": "no",
                        "close_time": "2026-04-01T23:59:00Z",
                        "last_price": 0.08,
                        "volume": 18000,
                    },
                ],
            },
        ],
        "cursor": None,
    }


def _mock_settled_events_page2():
    """Second page of settled events (for pagination test)."""
    return {
        "events": [
            {
                "event_ticker": "FED-RATE-MAR-2026",
                "title": "Fed rate decision March 2026",
                "category": "economics",
                "markets": [
                    {
                        "ticker": "FED-RATE-MAR-CUT",
                        "event_ticker": "FED-RATE-MAR-2026",
                        "series_ticker": "FED-RATE",
                        "title": "Fed cuts rates at March 2026 meeting",
                        "subtitle": "",
                        "status": "settled",
                        "result": "no",
                        "close_time": "2026-03-19T18:00:00Z",
                        "last_price": 0.12,
                        "volume": 45000,
                    },
                ],
            },
        ],
        "cursor": None,
    }


@pytest.fixture
async def ledger(tmp_path):
    from nexus.engine.projection.kalshi import KalshiLedger

    led = KalshiLedger(tmp_path / "bench.sqlite")
    await led.initialize()
    yield led
    await led.close()


# ── discover_settled_markets ──────────────────────────────────────────


class TestDiscoverSettledMarkets:
    async def test_returns_settled_markets(self):
        """Should extract settled binary markets from event response."""
        client = AsyncMock()
        client.list_events = AsyncMock(return_value=_mock_settled_events())

        markets = await discover_settled_markets(client, days_back=90)

        assert len(markets) == 3
        tickers = {m.ticker for m in markets}
        assert "OSCAR-BP-2026-ANORA" in tickers
        assert "OSCAR-BP-2026-BRUTALIST" in tickers
        assert "IRAN-STRIKE-APR" in tickers

    async def test_calls_with_settled_status(self):
        """Should call list_events with status='settled'."""
        client = AsyncMock()
        client.list_events = AsyncMock(return_value={"events": [], "cursor": None})

        await discover_settled_markets(client, days_back=90)

        client.list_events.assert_called()
        call_kwargs = client.list_events.call_args[1]
        assert call_kwargs["status"] == "settled"

    async def test_parses_result_correctly(self):
        """YES result → outcome=True, NO result → outcome=False."""
        client = AsyncMock()
        client.list_events = AsyncMock(return_value=_mock_settled_events())

        markets = await discover_settled_markets(client, days_back=90)

        by_ticker = {m.ticker: m for m in markets}
        assert by_ticker["OSCAR-BP-2026-ANORA"].outcome is True
        assert by_ticker["OSCAR-BP-2026-BRUTALIST"].outcome is False
        assert by_ticker["IRAN-STRIKE-APR"].outcome is False

    async def test_paginates(self):
        """Should follow cursor to get all pages."""
        client = AsyncMock()
        page1 = _mock_settled_events()
        page1["cursor"] = "page2_cursor"
        client.list_events = AsyncMock(
            side_effect=[page1, _mock_settled_events_page2()]
        )

        markets = await discover_settled_markets(client, days_back=90)

        assert len(markets) == 4  # 3 from page1 + 1 from page2
        assert client.list_events.call_count == 2

    async def test_includes_title_with_full_context(self):
        """Market title should be the full title (with deadline info)."""
        client = AsyncMock()
        client.list_events = AsyncMock(return_value=_mock_settled_events())

        markets = await discover_settled_markets(client, days_back=90)

        anora = next(m for m in markets if m.ticker == "OSCAR-BP-2026-ANORA")
        assert "Anora" in anora.title
        assert "2026 Oscars" in anora.title

    async def test_skips_non_binary_results(self):
        """Should skip markets without yes/no result."""
        events = _mock_settled_events()
        events["events"][0]["markets"][0]["result"] = "unknown"
        client = AsyncMock()
        client.list_events = AsyncMock(return_value=events)

        markets = await discover_settled_markets(client, days_back=90)

        tickers = {m.ticker for m in markets}
        assert "OSCAR-BP-2026-ANORA" not in tickers


# ── build_benchmark_dataset ───────────────────────────────────────────


class TestBuildBenchmarkDataset:
    async def test_produces_questions_at_cutoffs(self, ledger):
        """Should produce questions at each cutoff day."""
        # Seed a settled market
        settled = [
            SettledMarket(
                ticker="TEST-MKT",
                event_ticker="TEST-EVT",
                title="Test market resolves before March 15",
                outcome=True,
                settlement_date=date(2026, 3, 15),
                category="test",
            ),
        ]
        # Seed snapshots at various dates
        for days_before in [30, 14, 7, 3]:
            snap_date = date(2026, 3, 15) - timedelta(days=days_before)
            snap_dt = datetime.combine(snap_date, datetime.min.time(), tzinfo=timezone.utc)
            await ledger.upsert_market({
                "ticker": "TEST-MKT",
                "event_ticker": "TEST-EVT",
                "series_ticker": "TEST",
                "title": "Test market",
                "status": "settled",
            })
            await ledger.insert_snapshot("TEST-MKT", {
                "captured_at": snap_dt.isoformat(),
                "implied_probability": 0.5 + (days_before * 0.01),
                "last_price": 0.5 + (days_before * 0.01),
                "status": "open",
                "raw": {},
            })

        questions = await build_benchmark_dataset(
            ledger, settled, cutoff_days=[7, 14, 30]
        )

        assert len(questions) == 3  # one per cutoff
        cutoff_dates = {q.cutoff_date for q in questions}
        assert date(2026, 3, 8) in cutoff_dates   # 7 days before
        assert date(2026, 3, 1) in cutoff_dates   # 14 days before
        assert date(2026, 2, 13) in cutoff_dates  # 30 days before

    async def test_question_has_correct_outcome(self, ledger):
        """Outcome should match the settled market result."""
        settled = [
            SettledMarket(
                ticker="YES-MKT",
                event_ticker="EVT",
                title="Market that resolved YES",
                outcome=True,
                settlement_date=date(2026, 3, 10),
                category="test",
            ),
        ]
        await ledger.upsert_market({"ticker": "YES-MKT", "event_ticker": "EVT", "series_ticker": "T", "title": "x", "status": "settled"})
        snap_dt = datetime(2026, 3, 3, tzinfo=timezone.utc)
        await ledger.insert_snapshot("YES-MKT", {
            "captured_at": snap_dt.isoformat(),
            "implied_probability": 0.65,
            "last_price": 0.65,
            "status": "open",
            "raw": {},
        })

        questions = await build_benchmark_dataset(ledger, settled, cutoff_days=[7])

        assert len(questions) == 1
        assert questions[0].outcome is True
        assert questions[0].market_prob_at_cutoff == pytest.approx(0.65, abs=0.01)

    async def test_skips_cutoffs_without_snapshots(self, ledger):
        """If no snapshot exists before cutoff date, skip that cutoff."""
        settled = [
            SettledMarket(
                ticker="LATE-MKT",
                event_ticker="EVT",
                title="Market with only recent data",
                outcome=False,
                settlement_date=date(2026, 3, 15),
                category="test",
            ),
        ]
        await ledger.upsert_market({"ticker": "LATE-MKT", "event_ticker": "EVT", "series_ticker": "T", "title": "x", "status": "settled"})
        # Only a snapshot 3 days before settlement
        snap_dt = datetime(2026, 3, 12, tzinfo=timezone.utc)
        await ledger.insert_snapshot("LATE-MKT", {
            "captured_at": snap_dt.isoformat(),
            "implied_probability": 0.10,
            "last_price": 0.10,
            "status": "open",
            "raw": {},
        })

        questions = await build_benchmark_dataset(ledger, settled, cutoff_days=[7, 14, 30])

        # Only the 7-day cutoff should have a snapshot (March 8 → nearest is March 12, which is AFTER)
        # Actually March 12 > March 8 so no snapshot ≤ March 8 exists. Should be 0.
        assert len(questions) == 0


# ── build_benchmark_from_metadata ─────────────────────────────────────


class TestBuildBenchmarkFromMetadata:
    async def test_produces_questions_from_metadata(self, ledger):
        """Should create one question per settled market using last_price."""
        await ledger.upsert_market({
            "ticker": "META-T1", "event_ticker": "E1", "series_ticker": "",
            "title": "Will X happen?", "status": "finalized",
            "last_price_dollars": 0.65, "result": "yes",
        })
        settled = [
            SettledMarket(
                ticker="META-T1", event_ticker="E1",
                title="Will X happen?", outcome=True,
                settlement_date=date(2026, 3, 10), category="politics",
            ),
        ]

        questions = await build_benchmark_from_metadata(ledger, settled)

        assert len(questions) == 1
        assert questions[0].market_prob_at_cutoff == pytest.approx(0.65, abs=0.01)
        assert questions[0].outcome is True
        assert questions[0].ticker == "META-T1"

    async def test_filters_by_probability_range(self, ledger):
        """Should exclude markets outside min_prob/max_prob."""
        for i, price in enumerate([0.02, 0.45, 0.98]):
            ticker = f"RANGE-{i}"
            await ledger.upsert_market({
                "ticker": ticker, "event_ticker": "E", "series_ticker": "",
                "title": f"Q{i}", "status": "finalized",
                "last_price_dollars": price,
            })
        settled = [
            SettledMarket(ticker=f"RANGE-{i}", event_ticker="E", title=f"Q{i}",
                          outcome=True, settlement_date=date(2026, 3, 10), category="t")
            for i in range(3)
        ]

        questions = await build_benchmark_from_metadata(
            ledger, settled, min_prob=0.05, max_prob=0.95
        )

        assert len(questions) == 1
        assert questions[0].ticker == "RANGE-1"

    async def test_skips_markets_without_metadata(self, ledger):
        """Markets not in ledger should be skipped."""
        settled = [
            SettledMarket(ticker="MISSING", event_ticker="E", title="Q",
                          outcome=False, settlement_date=date(2026, 3, 10), category="t"),
        ]

        questions = await build_benchmark_from_metadata(ledger, settled)

        assert len(questions) == 0

    async def test_handles_cents_prices(self, ledger):
        """Prices > 1.0 should be divided by 100 (cents → dollars)."""
        await ledger.upsert_market({
            "ticker": "CENTS-T", "event_ticker": "E", "series_ticker": "",
            "title": "Q", "status": "finalized",
            "last_price_dollars": 72,  # 72 cents = 0.72
        })
        settled = [
            SettledMarket(ticker="CENTS-T", event_ticker="E", title="Q",
                          outcome=True, settlement_date=date(2026, 3, 10), category="t"),
        ]

        questions = await build_benchmark_from_metadata(ledger, settled)

        assert len(questions) == 1
        assert questions[0].market_prob_at_cutoff == pytest.approx(0.72, abs=0.01)


# ── run_benchmark ─────────────────────────────────────────────────────


class TestRunBenchmark:
    async def test_computes_brier_scores(self):
        """Should compute Brier scores for each engine."""
        questions = [
            BenchmarkQuestion(
                ticker="T1",
                question="Will X happen?",
                outcome=True,
                settlement_date=date(2026, 3, 15),
                cutoff_date=date(2026, 3, 8),
                market_prob_at_cutoff=0.70,
                category="test",
            ),
        ]

        class FixedEngine:
            engine_name = "fixed"
            async def predict_probability(self, question, **kwargs):
                return 0.80

        report = await run_benchmark(
            questions,
            engines={"fixed": FixedEngine()},
        )

        assert "fixed" in report.engine_results
        # Brier = (0.80 - 1.0)^2 = 0.04
        assert report.engine_results["fixed"]["mean_brier"] == pytest.approx(0.04, abs=0.01)

    async def test_includes_market_baseline(self):
        """Should include market baseline Brier automatically."""
        questions = [
            BenchmarkQuestion(
                ticker="T1",
                question="Will Y happen?",
                outcome=False,
                settlement_date=date(2026, 3, 15),
                cutoff_date=date(2026, 3, 8),
                market_prob_at_cutoff=0.30,
                category="test",
            ),
        ]

        report = await run_benchmark(questions, engines={})

        assert "market" in report.engine_results
        # Market prob 0.30, outcome NO=0.0, brier = (0.30 - 0.0)^2 = 0.09
        assert report.engine_results["market"]["mean_brier"] == pytest.approx(0.09, abs=0.01)

    async def test_multiple_questions(self):
        """Should average Brier scores across questions."""
        questions = [
            BenchmarkQuestion(
                ticker="A", question="Q1", outcome=True,
                settlement_date=date(2026, 3, 15),
                cutoff_date=date(2026, 3, 8),
                market_prob_at_cutoff=0.60, category="test",
            ),
            BenchmarkQuestion(
                ticker="B", question="Q2", outcome=False,
                settlement_date=date(2026, 3, 15),
                cutoff_date=date(2026, 3, 8),
                market_prob_at_cutoff=0.20, category="test",
            ),
        ]

        class PerfectEngine:
            engine_name = "perfect"
            async def predict_probability(self, question, **kwargs):
                return 1.0 if "Q1" in question else 0.0

        report = await run_benchmark(
            questions,
            engines={"perfect": PerfectEngine()},
        )

        # Perfect engine: brier(1.0, True) + brier(0.0, False) = 0 + 0 = 0
        assert report.engine_results["perfect"]["mean_brier"] == pytest.approx(0.0, abs=0.01)

    async def test_report_has_question_count(self):
        """Report should include total question count."""
        questions = [
            BenchmarkQuestion(
                ticker="X", question="Q", outcome=True,
                settlement_date=date(2026, 3, 15),
                cutoff_date=date(2026, 3, 8),
                market_prob_at_cutoff=0.50, category="test",
            ),
        ]

        report = await run_benchmark(questions, engines={})

        assert report.total_questions == 1

    async def test_engine_receives_question_text(self):
        """Engine should receive the question string."""
        questions = [
            BenchmarkQuestion(
                ticker="X", question="Will the Fed cut rates?", outcome=True,
                settlement_date=date(2026, 3, 15),
                cutoff_date=date(2026, 3, 8),
                market_prob_at_cutoff=0.50, category="test",
            ),
        ]
        received_questions = []

        class SpyEngine:
            engine_name = "spy"
            async def predict_probability(self, question, **kwargs):
                received_questions.append(question)
                return 0.50

        await run_benchmark(questions, engines={"spy": SpyEngine()})

        assert "Will the Fed cut rates?" in received_questions

    async def test_engine_receives_cutoff_date(self):
        """Engine should receive the as_of date."""
        questions = [
            BenchmarkQuestion(
                ticker="X", question="Q", outcome=True,
                settlement_date=date(2026, 3, 15),
                cutoff_date=date(2026, 3, 1),
                market_prob_at_cutoff=0.50, category="test",
            ),
        ]
        received_dates = []

        class SpyEngine:
            engine_name = "spy"
            async def predict_probability(self, question, *, as_of=None, **kwargs):
                received_dates.append(as_of)
                return 0.50

        await run_benchmark(questions, engines={"spy": SpyEngine()})

        assert date(2026, 3, 1) in received_dates
