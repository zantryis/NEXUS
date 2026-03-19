"""Kalshi live odds refresh — periodic snapshot sync for open predictions."""

import logging
from datetime import datetime, timezone

from nexus.engine.knowledge.store import KnowledgeStore

logger = logging.getLogger(__name__)


def _snapshot_from_market(market: dict) -> dict:
    """Extract snapshot data from a Kalshi market API response."""
    now = datetime.now(timezone.utc).isoformat()
    return {
        "captured_at": now,
        "implied_probability": market.get("last_price", market.get("implied_probability")),
        "last_price": market.get("last_price"),
        "yes_bid": market.get("yes_bid"),
        "yes_ask": market.get("yes_ask"),
        "no_bid": market.get("no_bid"),
        "no_ask": market.get("no_ask"),
        "volume": market.get("volume"),
        "open_interest": market.get("open_interest"),
        "status": market.get("status", ""),
        "raw_json": "{}",
    }


async def refresh_kalshi_odds(
    store: KnowledgeStore,
    kalshi_client,
    ledger,
) -> dict:
    """Refresh odds snapshots for all open Kalshi-aligned forecasts.

    For each open forecast with an external_ref (Kalshi ticker):
    1. Fetch current market data from Kalshi API
    2. Insert a new snapshot into the ledger
    3. Update forecast_questions.base_rate with latest implied probability

    Returns {markets_refreshed, snapshots_inserted, errors, skipped}.
    """
    stats = {"markets_refreshed": 0, "snapshots_inserted": 0, "errors": 0, "skipped": 0}

    open_forecasts = await store.get_open_forecasts()
    kalshi_forecasts = [f for f in open_forecasts if f.get("external_ref")]

    if not kalshi_forecasts:
        logger.info("Kalshi odds refresh: no open Kalshi-aligned forecasts")
        return stats

    # Deduplicate tickers (multiple questions can reference same market)
    seen_tickers: dict[str, dict] = {}
    for f in kalshi_forecasts:
        ticker = f["external_ref"]
        if ticker not in seen_tickers:
            seen_tickers[ticker] = f

    for ticker, forecast in seen_tickers.items():
        try:
            market = await kalshi_client.fetch_market(ticker)
            if not market:
                stats["skipped"] += 1
                continue

            snapshot = _snapshot_from_market(market)
            await ledger.insert_snapshot(ticker, snapshot)
            stats["snapshots_inserted"] += 1

            # Update base_rate on all questions with this ticker
            implied_prob = snapshot.get("implied_probability")
            if implied_prob is not None:
                for f in kalshi_forecasts:
                    if f["external_ref"] == ticker:
                        await store.update_forecast_probability(
                            f["forecast_question_id"],
                            f["probability"],  # Keep our probability unchanged
                            source="kalshi_odds_refresh",
                            market_probability=implied_prob,
                        )
            stats["markets_refreshed"] += 1

        except Exception as e:
            logger.warning(f"Kalshi odds refresh failed for {ticker}: {e}")
            stats["errors"] += 1

    logger.info(
        f"Kalshi odds refresh: {stats['markets_refreshed']} markets, "
        f"{stats['snapshots_inserted']} snapshots, {stats['errors']} errors"
    )

    # Emit signal for downstream consumers
    if stats["markets_refreshed"]:
        try:
            from nexus.signals import Signal, SignalType, bus

            await bus.emit(
                Signal(
                    SignalType.KALSHI_ODDS_UPDATED,
                    {"markets_refreshed": stats["markets_refreshed"]},
                )
            )
        except Exception:
            pass  # Signal bus is non-critical

    return stats
