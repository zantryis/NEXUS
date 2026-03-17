"""Kalshi benchmark-side adapter, ledger, and comparison helpers."""

from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from math import log
from pathlib import Path

import aiosqlite
import httpx

from nexus.config.models import KalshiBenchmarkConfig

try:
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding
except Exception:  # pragma: no cover - optional import path
    hashes = serialization = padding = None


_LEDGER_DDL = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS kalshi_markets (
    ticker TEXT PRIMARY KEY,
    event_ticker TEXT NOT NULL DEFAULT '',
    series_ticker TEXT NOT NULL DEFAULT '',
    title TEXT NOT NULL DEFAULT '',
    subtitle TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT '',
    raw_json TEXT NOT NULL DEFAULT '{}',
    first_seen_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS kalshi_market_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL REFERENCES kalshi_markets(ticker),
    captured_at TEXT NOT NULL,
    implied_probability REAL,
    last_price REAL,
    yes_bid REAL,
    yes_ask REAL,
    no_bid REAL,
    no_ask REAL,
    volume REAL,
    open_interest REAL,
    status TEXT NOT NULL DEFAULT '',
    raw_json TEXT NOT NULL DEFAULT '{}',
    UNIQUE(ticker, captured_at)
);

CREATE INDEX IF NOT EXISTS idx_kalshi_snapshots_ticker_time
    ON kalshi_market_snapshots(ticker, captured_at);
"""


@dataclass
class KalshiCredentials:
    key_id: str
    private_key_pem: str


class KalshiLedger:
    """Separate benchmark ledger for Kalshi market data."""

    def __init__(self, db_path: Path):
        self._db_path = db_path
        self._db: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(str(self._db_path))
        self._db.row_factory = aiosqlite.Row
        await self._db.executescript(_LEDGER_DDL)
        await self._db.execute("INSERT OR IGNORE INTO schema_version (version) VALUES (1)")
        await self._db.commit()

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None

    @property
    def db(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError("KalshiLedger not initialized.")
        return self._db

    async def upsert_market(self, payload: dict) -> None:
        await self.db.execute(
            "INSERT INTO kalshi_markets (ticker, event_ticker, series_ticker, title, subtitle, status, raw_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(ticker) DO UPDATE SET "
            "event_ticker = excluded.event_ticker, "
            "series_ticker = excluded.series_ticker, "
            "title = excluded.title, "
            "subtitle = excluded.subtitle, "
            "status = excluded.status, "
            "raw_json = excluded.raw_json, "
            "updated_at = datetime('now')",
            (
                payload.get("ticker", ""),
                payload.get("event_ticker", ""),
                payload.get("series_ticker", ""),
                payload.get("title", ""),
                payload.get("subtitle", ""),
                payload.get("status", ""),
                json.dumps(payload),
            ),
        )
        await self.db.commit()

    async def insert_snapshot(self, ticker: str, snapshot: dict) -> None:
        await self.db.execute(
            "INSERT OR IGNORE INTO kalshi_market_snapshots "
            "(ticker, captured_at, implied_probability, last_price, yes_bid, yes_ask, no_bid, no_ask, volume, open_interest, status, raw_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                ticker,
                snapshot["captured_at"],
                snapshot.get("implied_probability"),
                snapshot.get("last_price"),
                snapshot.get("yes_bid"),
                snapshot.get("yes_ask"),
                snapshot.get("no_bid"),
                snapshot.get("no_ask"),
                snapshot.get("volume"),
                snapshot.get("open_interest"),
                snapshot.get("status", ""),
                json.dumps(snapshot.get("raw", {})),
            ),
        )
        await self.db.commit()

    async def get_nearest_snapshot(
        self,
        ticker: str,
        *,
        captured_at: datetime,
    ) -> dict | None:
        cursor = await self.db.execute(
            "SELECT ticker, captured_at, implied_probability, last_price, yes_bid, yes_ask, no_bid, no_ask, "
            "volume, open_interest, status "
            "FROM kalshi_market_snapshots WHERE ticker = ? AND captured_at <= ? "
            "ORDER BY captured_at DESC LIMIT 1",
            (ticker, captured_at.isoformat()),
        )
        row = await cursor.fetchone()
        if not row:
            return None
        return {
            "ticker": row[0],
            "captured_at": row[1],
            "implied_probability": row[2],
            "last_price": row[3],
            "yes_bid": row[4],
            "yes_ask": row[5],
            "no_bid": row[6],
            "no_ask": row[7],
            "volume": row[8],
            "open_interest": row[9],
            "status": row[10],
        }

    async def get_market_metadata(self, ticker: str) -> dict | None:
        """Return raw market metadata for a ticker, or None."""
        cursor = await self.db.execute(
            "SELECT raw_json FROM kalshi_markets WHERE ticker = ?", (ticker,)
        )
        row = await cursor.fetchone()
        if not row:
            return None
        return json.loads(row[0])

    async def get_finalized_markets(self) -> list[dict]:
        """Return all finalized markets with their metadata."""
        cursor = await self.db.execute(
            "SELECT ticker, raw_json FROM kalshi_markets WHERE status IN ('settled', 'finalized')"
        )
        rows = await cursor.fetchall()
        return [{"ticker": row[0], **json.loads(row[1])} for row in rows]

    async def counts(self) -> dict:
        cursor = await self.db.execute("SELECT COUNT(*) FROM kalshi_markets")
        market_count = (await cursor.fetchone())[0]
        cursor = await self.db.execute("SELECT COUNT(*) FROM kalshi_market_snapshots")
        snapshot_count = (await cursor.fetchone())[0]
        return {"markets": market_count, "snapshots": snapshot_count}


def load_kalshi_credentials(config: KalshiBenchmarkConfig) -> KalshiCredentials | None:
    key_id = os.getenv(config.api_key_id_env, "").strip()
    pem = os.getenv(config.private_key_pem_env, "").strip()
    key_path = os.getenv(config.private_key_path_env, "").strip()
    if not pem and key_path:
        path = Path(key_path)
        if path.exists():
            pem = path.read_text()
    if not key_id or not pem:
        return None
    return KalshiCredentials(key_id=key_id, private_key_pem=pem)


def parse_kalshi_cred_file(cred_path: Path) -> KalshiCredentials:
    """Parse a local bootstrap credential file without changing runtime auth flow."""
    lines = cred_path.read_text().splitlines()
    if len(lines) < 3:
        raise ValueError("Kalshi credential file must contain a label, key ID, and PEM.")
    key_id = lines[1].strip()
    pem = "\n".join(lines[2:]).strip()
    if not key_id:
        raise ValueError("Kalshi credential file is missing the API key ID line.")
    if "BEGIN" not in pem or "PRIVATE KEY" not in pem:
        raise ValueError("Kalshi credential file is missing a valid PEM private key block.")
    return KalshiCredentials(key_id=key_id, private_key_pem=f"{pem}\n")


def bootstrap_kalshi_credentials(
    *,
    cred_path: Path,
    config: KalshiBenchmarkConfig,
    target_key_path: Path | None = None,
) -> dict:
    """Normalize a local credential bootstrap file into the env/path flow used by the client."""
    credentials = parse_kalshi_cred_file(cred_path)
    key_path = target_key_path or Path(".secrets/kalshi_private_key.pem")
    key_path.parent.mkdir(parents=True, exist_ok=True)
    key_path.write_text(credentials.private_key_pem)
    try:
        key_path.chmod(0o600)
    except OSError:
        pass
    return {
        "cred_file": str(cred_path),
        "key_file": str(key_path),
        "env": {
            config.api_key_id_env: credentials.key_id,
            config.private_key_path_env: str(key_path),
        },
        "exports": [
            f'export {config.api_key_id_env}="{credentials.key_id}"',
            f'export {config.private_key_path_env}="{key_path}"',
        ],
    }


class KalshiClient:
    """Small Kalshi client for benchmark-side market sync."""

    def __init__(self, config: KalshiBenchmarkConfig):
        self.config = config

    def credentials_available(self) -> bool:
        return load_kalshi_credentials(self.config) is not None

    def auth_capable(self) -> bool:
        return self.credentials_available() and all((hashes, serialization, padding))

    def _auth_headers(self, method: str, path: str, timestamp_ms: int) -> dict[str, str]:
        credentials = load_kalshi_credentials(self.config)
        if credentials is None:
            raise RuntimeError("Kalshi credentials are not configured.")
        if not all((hashes, serialization, padding)):
            raise RuntimeError("cryptography is required for Kalshi RSA signing.")
        path_without_query = path.split("?", 1)[0]
        base_path = httpx.URL(self.config.base_url).path.rstrip("/")
        signed_path = (
            path_without_query
            if base_path and path_without_query.startswith(base_path)
            else f"{base_path}{path_without_query}"
        )
        private_key = serialization.load_pem_private_key(
            credentials.private_key_pem.encode("utf-8"),
            password=None,
        )
        message = f"{timestamp_ms}{method.upper()}{signed_path}".encode("utf-8")
        signature = private_key.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH,
            ),
            hashes.SHA256(),
        )
        return {
            "KALSHI-ACCESS-KEY": credentials.key_id,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(signature).decode("utf-8"),
            "KALSHI-ACCESS-TIMESTAMP": str(timestamp_ms),
        }

    async def _request_json(
        self,
        method: str,
        path: str,
        *,
        params: dict | None = None,
        auth_required: bool = False,
    ) -> dict:
        timestamp_ms = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
        headers: dict[str, str] = {}
        if auth_required:
            headers.update(self._auth_headers(method, path, timestamp_ms))
        async with httpx.AsyncClient(base_url=self.config.base_url, timeout=30.0) as client:
            response = await client.request(method, path, params=params, headers=headers)
            response.raise_for_status()
            return response.json()

    async def fetch_market(self, ticker: str) -> dict:
        payload = await self._request_json("GET", f"/markets/{ticker}")
        return payload.get("market", payload)

    async def fetch_candlesticks(
        self,
        *,
        series_ticker: str,
        ticker: str,
        start: date,
        end: date,
    ) -> list[dict]:
        start_ts = int(datetime.combine(start, time.min, tzinfo=timezone.utc).timestamp())
        end_ts = int(datetime.combine(end, time.max, tzinfo=timezone.utc).timestamp())
        payload = await self._request_json(
            "GET",
            f"/series/{series_ticker}/markets/{ticker}/candlesticks",
            params={"start_ts": start_ts, "end_ts": end_ts, "period_interval": 1440},
        )
        return payload.get("candlesticks", [])

    async def probe_authenticated(self) -> bool:
        try:
            status = await self.auth_check()
            return bool(status["auth_success"])
        except Exception:
            return False

    async def auth_check(self) -> dict:
        """Run a small authenticated smoke test and return a human-readable status payload."""
        status = {
            "configured": self.credentials_available(),
            "auth_capable": self.auth_capable(),
            "auth_success": False,
            "http_status": None,
            "error": None,
        }
        if not status["configured"]:
            status["error"] = "Kalshi credentials are not configured."
            return status
        if not status["auth_capable"]:
            status["error"] = "Kalshi auth is not available. Check cryptography and credential env vars."
            return status
        try:
            await self._request_json("GET", "/portfolio/balance", auth_required=True)
            status["auth_success"] = True
            return status
        except httpx.HTTPStatusError as exc:
            status["http_status"] = exc.response.status_code
            try:
                payload = exc.response.json()
                status["error"] = json.dumps(payload)
            except Exception:
                status["error"] = exc.response.text[:500]
            return status
        except Exception as exc:
            status["error"] = str(exc)
            return status

    async def list_events(
        self,
        *,
        status: str = "open",
        limit: int = 200,
        cursor: str | None = None,
    ) -> dict:
        """List Kalshi events (public, no auth needed).

        Returns {events: [...], cursor: str | None}.
        """
        params: dict = {"status": status, "limit": limit, "with_nested_markets": "true"}
        if cursor:
            params["cursor"] = cursor
        return await self._request_json("GET", "/events", params=params)


def derive_series_ticker(event_ticker: str) -> str:
    """Derive series_ticker from event_ticker by stripping the trailing segment.

    Kalshi event tickers follow the pattern SERIES-SUFFIX (e.g. "KXNEXTPOPE-35").
    The series ticker is the prefix before the last hyphen.
    Validated: 40/40 random settled markets returned candles with this derivation.
    """
    if "-" in event_ticker:
        return event_ticker.rsplit("-", 1)[0]
    return event_ticker


def _coerce_float(value) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _snapshot_from_market_payload(payload: dict, *, captured_at: datetime) -> dict:
    last_price = (
        _coerce_float(payload.get("last_price"))
        or _coerce_float(payload.get("last_price_dollars"))
        or _coerce_float(payload.get("close"))
    )
    yes_bid = _coerce_float(payload.get("yes_bid")) or _coerce_float(payload.get("yes_bid_dollars"))
    yes_ask = _coerce_float(payload.get("yes_ask")) or _coerce_float(payload.get("yes_ask_dollars"))
    no_bid = _coerce_float(payload.get("no_bid")) or _coerce_float(payload.get("no_bid_dollars"))
    no_ask = _coerce_float(payload.get("no_ask")) or _coerce_float(payload.get("no_ask_dollars"))
    implied = _coerce_float(payload.get("implied_probability"))
    if implied is None:
        raw_last = last_price
        if raw_last is not None and raw_last > 1.0:
            implied = round(raw_last / 100.0, 4)
        elif raw_last is not None:
            implied = round(raw_last, 4)
    return {
        "captured_at": captured_at.isoformat(),
        "implied_probability": implied,
        "last_price": last_price,
        "yes_bid": yes_bid,
        "yes_ask": yes_ask,
        "no_bid": no_bid,
        "no_ask": no_ask,
        "volume": _coerce_float(payload.get("volume")) or _coerce_float(payload.get("volume_fp")),
        "open_interest": _coerce_float(payload.get("open_interest")) or _coerce_float(payload.get("open_interest_fp")),
        "status": payload.get("status", ""),
        "raw": payload,
    }


async def sync_kalshi_tickers(
    ledger: KalshiLedger,
    client: KalshiClient,
    *,
    tickers: list[str],
    start: date,
    end: date,
    series_override: str | None = None,
) -> dict:
    """Sync market metadata and daily snapshots for a set of Kalshi tickers.

    If series_override is provided, use it for all tickers' candlestick fetches.
    Otherwise, try market payload's series_ticker, then derive from event_ticker.
    """
    synced = 0
    snapshots = 0
    for ticker in tickers:
        market = await client.fetch_market(ticker)
        await ledger.upsert_market(market)
        synced += 1
        series_ticker = series_override or market.get("series_ticker", "")
        if not series_ticker:
            event_ticker = market.get("event_ticker", "")
            if event_ticker:
                series_ticker = derive_series_ticker(event_ticker)
        if series_ticker:
            try:
                candles = await client.fetch_candlesticks(
                    series_ticker=series_ticker,
                    ticker=ticker,
                    start=start,
                    end=end,
                )
            except Exception:
                candles = []
        else:
            candles = []
        if candles:
            for candle in candles:
                end_ts = candle.get("end_period_ts") or candle.get("end_ts") or candle.get("ts")
                if end_ts is None:
                    continue
                captured_at = datetime.fromtimestamp(int(end_ts), tz=timezone.utc)
                snapshot = _snapshot_from_market_payload(candle, captured_at=captured_at)
                await ledger.insert_snapshot(ticker, snapshot)
                snapshots += 1
        else:
            snapshot = _snapshot_from_market_payload(
                market,
                captured_at=datetime.combine(end, time.max, tzinfo=timezone.utc),
            )
            await ledger.insert_snapshot(ticker, snapshot)
            snapshots += 1
    counts = await ledger.counts()
    return {
        "tickers_synced": synced,
        "snapshots_inserted": snapshots,
        "ledger_counts": counts,
    }


def load_kalshi_mappings(mapping_path: Path) -> dict[str, dict]:
    """Load manual forecast-to-market mappings keyed by forecast_key."""
    if not mapping_path.exists():
        return {}
    payload = json.loads(mapping_path.read_text())
    mappings = payload.get("mappings", payload if isinstance(payload, list) else [])
    result: dict[str, dict] = {}
    for mapping in mappings:
        forecast_key = mapping.get("forecast_key", "").strip()
        market_ticker = mapping.get("market_ticker", "").strip()
        if forecast_key and market_ticker:
            result[forecast_key] = mapping
    return result


def _bounded_probability(value: float) -> float:
    return max(0.05, min(0.95, float(value)))


def _brier(probability: float, outcome: bool) -> float:
    target = 1.0 if outcome else 0.0
    return round((_bounded_probability(probability) - target) ** 2, 4)


def _log_loss(probability: float, outcome: bool) -> float:
    p = _bounded_probability(probability)
    return round(-log(p if outcome else (1.0 - p)), 4)


async def compare_forecasts_to_kalshi(
    store,
    ledger: KalshiLedger,
    *,
    start: date,
    end: date,
    mapping_path: Path,
    engine: str | None = None,
) -> dict:
    """Compare stored forecasts against mapped Kalshi market snapshots."""
    mappings = load_kalshi_mappings(mapping_path)
    questions = await store.get_forecast_questions_between(start=start, end=end, engine=engine)
    rows: list[dict] = []
    for question in questions:
        mapping = mappings.get(question["forecast_key"])
        if not mapping:
            continue
        market_ticker = mapping["market_ticker"]
        side = mapping.get("side", "yes").lower()
        generated_for = date.fromisoformat(question["generated_for"])
        as_of = datetime.combine(generated_for, time.max, tzinfo=timezone.utc)
        snapshot = await ledger.get_nearest_snapshot(market_ticker, captured_at=as_of)
        if not snapshot or snapshot.get("implied_probability") is None:
            continue
        market_probability = float(snapshot["implied_probability"])
        if side == "no":
            market_probability = round(1.0 - market_probability, 4)
        outcome = question["resolved_bool"]
        rows.append({
            "forecast_key": question["forecast_key"],
            "topic_slug": question["topic_slug"],
            "engine": question["engine"],
            "question": question["question"],
            "market_ticker": market_ticker,
            "side": side,
            "generated_for": question["generated_for"],
            "market_snapshot_at": snapshot["captured_at"],
            "our_probability": question["probability"],
            "market_probability": market_probability,
            "probability_gap": round(abs(float(question["probability"]) - market_probability), 4),
            "outcome_status": question["outcome_status"],
            "resolved_bool": outcome,
            "our_brier": _brier(float(question["probability"]), bool(outcome)) if outcome is not None else None,
            "market_brier": _brier(market_probability, bool(outcome)) if outcome is not None else None,
            "our_log_loss": _log_loss(float(question["probability"]), bool(outcome)) if outcome is not None else None,
            "market_log_loss": _log_loss(market_probability, bool(outcome)) if outcome is not None else None,
        })

    resolved_rows = [row for row in rows if row["resolved_bool"] is not None]
    return {
        "meta": {
            "start": start.isoformat(),
            "end": end.isoformat(),
            "mapped_forecasts": len(rows),
            "resolved_mapped_forecasts": len(resolved_rows),
            "mapping_file": str(mapping_path),
            "forecast_time_convention": "end_of_day",
        },
        "rows": rows,
        "summary": {
            "mean_probability_gap": round(
                sum(row["probability_gap"] for row in rows) / len(rows),
                4,
            ) if rows else 0.0,
            "our_mean_brier": round(
                sum(row["our_brier"] for row in resolved_rows if row["our_brier"] is not None) / len(resolved_rows),
                4,
            ) if resolved_rows else 0.0,
            "market_mean_brier": round(
                sum(row["market_brier"] for row in resolved_rows if row["market_brier"] is not None) / len(resolved_rows),
                4,
            ) if resolved_rows else 0.0,
        },
    }
