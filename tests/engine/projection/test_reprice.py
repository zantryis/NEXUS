"""Tests for forecast repricing gate logic and service."""

import pytest
from datetime import date, datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

from nexus.engine.projection.reprice import (
    run_reprice_pass,
    should_reprice,
)


def _make_question(
    *,
    probability=0.65,
    resolution_days=14,
    updated_at=None,
    generated_for=None,
    topic_slug="ai",
    external_ref=None,
    base_rate=None,
):
    today = date.today()
    return {
        "forecast_question_id": 1,
        "topic_slug": topic_slug,
        "engine": "structural",
        "generated_for": (generated_for or today).isoformat(),
        "question": "Will X happen?",
        "probability": probability,
        "base_rate": base_rate,
        "resolution_date": (today + timedelta(days=resolution_days)).isoformat(),
        "target_variable": "kg_native",
        "external_ref": external_ref,
        "updated_at": updated_at,
        "horizon_days": 7,
    }


@pytest.mark.asyncio
async def test_should_reprice_stale_question():
    """Questions older than max_age_hours should be repriced."""
    store = AsyncMock()
    store.get_recent_events = AsyncMock(return_value=[])
    q = _make_question(updated_at=None, generated_for=date.today() - timedelta(days=2))
    assert await should_reprice(q, store, max_age_hours=24) is True


@pytest.mark.asyncio
async def test_should_reprice_fresh_question_no_events():
    """Fresh questions without new events should not be repriced."""
    store = AsyncMock()
    store.get_recent_events = AsyncMock(return_value=[])
    now = datetime.now(timezone.utc).isoformat()
    q = _make_question(updated_at=now)
    assert await should_reprice(q, store, max_age_hours=24) is False


@pytest.mark.asyncio
async def test_should_reprice_fresh_with_new_events():
    """Fresh questions WITH new events should be repriced."""
    store = AsyncMock()
    store.get_recent_events = AsyncMock(return_value=[{"id": 1}])
    now = datetime.now(timezone.utc).isoformat()
    q = _make_question(updated_at=now)
    assert await should_reprice(q, store, max_age_hours=24) is True


@pytest.mark.asyncio
async def test_should_reprice_skip_near_resolution():
    """Questions resolving within 1 day should be skipped."""
    store = AsyncMock()
    store.get_recent_events = AsyncMock(return_value=[{"id": 1}])
    q = _make_question(resolution_days=0)
    assert await should_reprice(q, store) is False


@pytest.mark.asyncio
async def test_should_reprice_skip_extreme_no_events():
    """Extreme probabilities without new events should be skipped."""
    store = AsyncMock()
    store.get_recent_events = AsyncMock(return_value=[])
    q = _make_question(probability=0.03, updated_at=None, generated_for=date.today() - timedelta(days=2))
    assert await should_reprice(q, store) is False


@pytest.mark.asyncio
async def test_should_reprice_extreme_with_events():
    """Extreme probabilities WITH new events should be repriced."""
    store = AsyncMock()
    store.get_recent_events = AsyncMock(return_value=[{"id": 1}])
    q = _make_question(probability=0.03, updated_at=None, generated_for=date.today() - timedelta(days=2))
    assert await should_reprice(q, store) is True


@pytest.mark.asyncio
async def test_run_reprice_pass_reprices_stale():
    """run_reprice_pass should reprice stale questions."""
    store = AsyncMock()
    q = _make_question(probability=0.65, updated_at=None, generated_for=date.today() - timedelta(days=2))
    store.get_open_forecasts = AsyncMock(return_value=[q])
    store.get_recent_events = AsyncMock(return_value=[])
    store.update_forecast_probability = AsyncMock()

    llm = AsyncMock()
    config = AsyncMock()

    mock_assessment = AsyncMock()
    mock_assessment.implied_probability = 0.80

    with patch("nexus.engine.projection.reprice.reprice_forecast", return_value=0.80):
        stats = await run_reprice_pass(store, llm, config)

    assert stats["total_open"] == 1
    assert stats["repriced"] == 1
    assert stats["skipped"] == 0
    store.update_forecast_probability.assert_called_once()
    call_args = store.update_forecast_probability.call_args
    assert call_args[0][1] == pytest.approx(0.80)


@pytest.mark.asyncio
async def test_run_reprice_pass_skips_unchanged():
    """run_reprice_pass should skip if probability change is <2pp."""
    store = AsyncMock()
    q = _make_question(probability=0.65, updated_at=None, generated_for=date.today() - timedelta(days=2))
    store.get_open_forecasts = AsyncMock(return_value=[q])
    store.get_recent_events = AsyncMock(return_value=[])
    store.update_forecast_probability = AsyncMock()

    with patch("nexus.engine.projection.reprice.reprice_forecast", return_value=0.66):
        stats = await run_reprice_pass(store, AsyncMock(), AsyncMock())

    assert stats["repriced"] == 0
    assert stats["skipped"] == 1
    # Still updates to mark as checked
    store.update_forecast_probability.assert_called_once()
    assert store.update_forecast_probability.call_args[1]["source"] == "daily_reprice_unchanged"


@pytest.mark.asyncio
async def test_run_reprice_pass_handles_errors():
    """run_reprice_pass should handle individual question errors gracefully."""
    store = AsyncMock()
    q = _make_question(updated_at=None, generated_for=date.today() - timedelta(days=2))
    store.get_open_forecasts = AsyncMock(return_value=[q])
    store.get_recent_events = AsyncMock(return_value=[])

    with patch("nexus.engine.projection.reprice.reprice_forecast", side_effect=RuntimeError("LLM down")):
        stats = await run_reprice_pass(store, AsyncMock(), AsyncMock())

    assert stats["errors"] == 1
    assert stats["repriced"] == 0
