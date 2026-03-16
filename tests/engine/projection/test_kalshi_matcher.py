"""Tests for Kalshi market matching and aligned forecast generation."""

from __future__ import annotations

import json
from datetime import date
from unittest.mock import AsyncMock, MagicMock

import pytest

from nexus.engine.projection.models import ForecastQuestion, ForecastRun


# ── Fixtures ─────────────────────────────────────────────────────────


def _mock_kalshi_events():
    """Simulated Kalshi event listing with nested markets."""
    return {
        "events": [
            {
                "event_ticker": "IRAN-STRIKE-2026",
                "title": "Will Iran conduct a military strike before April 2026?",
                "category": "world",
                "markets": [
                    {
                        "ticker": "IRAN-STRIKE-2026-Y",
                        "title": "Iran military strike before April 2026",
                        "subtitle": "Resolves Yes if Iran conducts a military strike",
                        "status": "open",
                        "yes_bid": 0.31,
                        "yes_ask": 0.34,
                        "last_price": 0.32,
                        "volume": 15420,
                        "open_interest": 8700,
                    },
                ],
            },
            {
                "event_ticker": "AI-REGULATION-2026",
                "title": "Will the EU pass AI regulation before July 2026?",
                "category": "tech",
                "markets": [
                    {
                        "ticker": "AI-REG-EU-2026-Y",
                        "title": "EU AI regulation before July 2026",
                        "subtitle": "Resolves Yes if EU passes comprehensive AI regulation",
                        "status": "open",
                        "yes_bid": 0.62,
                        "yes_ask": 0.65,
                        "last_price": 0.63,
                        "volume": 9300,
                        "open_interest": 4200,
                    },
                ],
            },
            {
                "event_ticker": "BITCOIN-100K",
                "title": "Will Bitcoin reach $100,000 before June 2026?",
                "category": "crypto",
                "markets": [
                    {
                        "ticker": "BTC-100K-2026-Y",
                        "title": "Bitcoin reaches $100K before June 2026",
                        "subtitle": "",
                        "status": "open",
                        "yes_bid": 0.45,
                        "yes_ask": 0.48,
                        "last_price": 0.46,
                        "volume": 32100,
                        "open_interest": 18500,
                    },
                ],
            },
        ],
        "cursor": None,
    }


# ── Market Scanning ──────────────────────────────────────────────────


class TestScanKalshiMarkets:

    async def test_matches_topic_entities(self):
        from nexus.engine.projection.kalshi_matcher import scan_kalshi_markets

        mock_client = AsyncMock()
        mock_client.list_events.return_value = _mock_kalshi_events()

        matches = await scan_kalshi_markets(
            mock_client,
            entity_names=["Iran", "Israel", "United States"],
            topic_name="Iran-US Relations",
        )
        assert len(matches) >= 1
        # Should match the Iran strike market
        tickers = [m["ticker"] for m in matches]
        assert "IRAN-STRIKE-2026-Y" in tickers

    async def test_scores_by_entity_overlap(self):
        from nexus.engine.projection.kalshi_matcher import scan_kalshi_markets

        mock_client = AsyncMock()
        mock_client.list_events.return_value = _mock_kalshi_events()

        matches = await scan_kalshi_markets(
            mock_client,
            entity_names=["Iran", "Israel"],
            topic_name="Iran-US Relations",
        )
        # Iran market should have higher match score than unrelated ones
        if len(matches) >= 2:
            assert matches[0]["match_score"] >= matches[1]["match_score"]

    async def test_returns_implied_probability(self):
        from nexus.engine.projection.kalshi_matcher import scan_kalshi_markets

        mock_client = AsyncMock()
        mock_client.list_events.return_value = _mock_kalshi_events()

        matches = await scan_kalshi_markets(
            mock_client,
            entity_names=["Iran"],
            topic_name="Iran-US Relations",
        )
        assert len(matches) >= 1
        iran_match = next(m for m in matches if m["ticker"] == "IRAN-STRIKE-2026-Y")
        assert iran_match["implied_probability"] is not None
        assert 0.0 <= iran_match["implied_probability"] <= 1.0

    async def test_excludes_zero_overlap(self):
        from nexus.engine.projection.kalshi_matcher import scan_kalshi_markets

        mock_client = AsyncMock()
        mock_client.list_events.return_value = _mock_kalshi_events()

        matches = await scan_kalshi_markets(
            mock_client,
            entity_names=["Japan", "South Korea"],
            topic_name="East Asian Diplomacy",
        )
        # None of the mock markets mention Japan or South Korea
        assert len(matches) == 0

    async def test_respects_max_markets(self):
        from nexus.engine.projection.kalshi_matcher import scan_kalshi_markets

        mock_client = AsyncMock()
        mock_client.list_events.return_value = _mock_kalshi_events()

        matches = await scan_kalshi_markets(
            mock_client,
            entity_names=["Iran", "AI", "Bitcoin", "EU"],
            topic_name="Everything",
            max_markets=2,
        )
        assert len(matches) <= 2


# ── Aligned Forecast Generation ──────────────────────────────────────


class TestGenerateAlignedForecasts:

    async def test_generates_question_per_market(self):
        from nexus.engine.projection.kalshi_matcher import generate_aligned_forecasts

        matched_markets = [
            {
                "ticker": "IRAN-STRIKE-2026-Y",
                "title": "Iran military strike before April 2026",
                "event_title": "Will Iran conduct a military strike before April 2026?",
                "implied_probability": 0.32,
                "match_score": 3,
                "matched_entities": ["Iran"],
            },
        ]

        mock_llm = AsyncMock()
        mock_llm.complete.return_value = json.dumps({
            "reasoning": "Graph shows active 'threatens' relationship with strength 0.9",
            "probability": 0.58,
        })

        mock_store = AsyncMock()
        mock_store.find_entity.return_value = {
            "id": 1, "canonical_name": "Iran", "entity_type": "country",
        }
        mock_store.get_entity_neighborhood.return_value = {
            "entities": [], "relationships": [],
        }
        mock_store.get_relationship_timeline.return_value = []

        questions = await generate_aligned_forecasts(
            mock_llm,
            mock_store,
            matched_markets,
            topic_slug="iran-us-relations",
            run_date=date(2026, 3, 15),
        )
        assert len(questions) == 1
        q = questions[0]
        assert isinstance(q, ForecastQuestion)
        # Question text should reference the Kalshi market
        assert "Iran" in q.question or "military" in q.question.lower()
        assert q.external_ref == "IRAN-STRIKE-2026-Y"
        assert 0.05 <= q.probability <= 0.95

    async def test_falls_back_without_llm(self):
        from nexus.engine.projection.kalshi_matcher import generate_aligned_forecasts

        matched_markets = [
            {
                "ticker": "IRAN-STRIKE-2026-Y",
                "title": "Iran military strike before April 2026",
                "event_title": "Will Iran conduct a military strike before April 2026?",
                "implied_probability": 0.32,
                "match_score": 3,
                "matched_entities": ["Iran"],
            },
        ]

        mock_store = AsyncMock()
        mock_store.find_entity.return_value = None
        mock_store.get_entity_neighborhood.return_value = {
            "entities": [], "relationships": [],
        }
        mock_store.get_relationship_timeline.return_value = []

        questions = await generate_aligned_forecasts(
            None,  # no LLM
            mock_store,
            matched_markets,
            topic_slug="iran-us-relations",
            run_date=date(2026, 3, 15),
        )
        assert len(questions) == 1
        q = questions[0]
        # Without LLM, should use implied probability as base
        assert 0.05 <= q.probability <= 0.95


# ── Divergence Tracking ──────────────────────────────────────────────


class TestDivergenceTracking:

    async def test_returns_sorted_gaps(self):
        from nexus.engine.projection.kalshi_matcher import compute_divergences

        forecasts = [
            ForecastQuestion(
                question="Will Iran strike?",
                forecast_type="binary",
                target_variable="kalshi_aligned",
                probability=0.65,
                base_rate=0.32,
                resolution_criteria="Market resolution",
                resolution_date=date(2026, 4, 1),
                horizon_days=14,
                signpost="Active threatens relationship",
                external_ref="IRAN-STRIKE-2026-Y",
                target_metadata={"kalshi_implied": 0.32},
            ),
            ForecastQuestion(
                question="Will EU pass AI regulation?",
                forecast_type="binary",
                target_variable="kalshi_aligned",
                probability=0.60,
                base_rate=0.63,
                resolution_criteria="Market resolution",
                resolution_date=date(2026, 7, 1),
                horizon_days=14,
                signpost="Regulatory signals",
                external_ref="AI-REG-EU-2026-Y",
                target_metadata={"kalshi_implied": 0.63},
            ),
        ]

        divergences = compute_divergences(forecasts)
        assert len(divergences) == 2
        # Should be sorted by absolute gap descending
        assert divergences[0]["gap_pp"] >= divergences[1]["gap_pp"]
        # Iran gap should be larger (65% vs 32% = 33pp)
        assert divergences[0]["ticker"] == "IRAN-STRIKE-2026-Y"
        assert divergences[0]["gap_pp"] == pytest.approx(33.0, abs=1.0)

    def test_empty_forecasts(self):
        from nexus.engine.projection.kalshi_matcher import compute_divergences

        assert compute_divergences([]) == []


# ── Briefing Section Rendering ───────────────────────────────────────


class TestDivergenceBriefingSection:

    def test_renders_divergence_section(self):
        from nexus.engine.projection.kalshi_matcher import render_divergence_section

        divergences = [
            {
                "ticker": "IRAN-STRIKE-2026-Y",
                "question": "Will Iran strike?",
                "our_probability": 0.65,
                "kalshi_probability": 0.32,
                "gap_pp": 33.0,
                "direction": "above",
            },
            {
                "ticker": "AI-REG-EU-2026-Y",
                "question": "Will EU pass AI regulation?",
                "our_probability": 0.60,
                "kalshi_probability": 0.63,
                "gap_pp": 3.0,
                "direction": "below",
            },
        ]

        section = render_divergence_section(divergences)
        assert "Market Divergence" in section
        assert "Iran" in section
        assert "33" in section  # gap
        # Small gaps should be noted as aligned
        assert "aligned" in section.lower() or "3" in section

    def test_empty_divergences(self):
        from nexus.engine.projection.kalshi_matcher import render_divergence_section

        section = render_divergence_section([])
        assert section == ""
