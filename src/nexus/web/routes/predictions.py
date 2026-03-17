"""Predictions dashboard — grouped by market, engine comparison inline."""

from collections import OrderedDict
from datetime import date, timedelta

from fastapi import APIRouter, Request

from nexus.web.app import get_store, get_templates

router = APIRouter()

PREDICTIONS_LOOKBACK_DAYS = 90
RESOLVING_SOON_DAYS = 7

ENGINE_INFO = {
    "structural": {
        "label": "Structural", "class": "production", "uses_kg": True,
        "has_verdict": True, "calls": "3",
        "desc": "3 LLM passes: base-rate analyst, contrarian, supervisor.",
    },
    "actor": {
        "label": "Actor", "class": "production", "uses_kg": True,
        "has_verdict": True, "calls": "3-6",
        "desc": "Per-actor reasoning with market-anchored synthesis.",
    },
    "graphrag": {
        "label": "GraphRAG", "class": "production", "uses_kg": True,
        "has_verdict": True, "calls": "2",
        "desc": "Graph traversal + holistic system reasoning.",
    },
    "naked": {
        "label": "Naked", "class": "strawman", "uses_kg": False,
        "has_verdict": False, "calls": "1",
        "desc": "LLM world knowledge only — KG contribution baseline.",
    },
    "perspective": {
        "label": "Perspective", "class": "research", "uses_kg": True,
        "has_verdict": False, "calls": "4-6",
        "desc": "Multiple analyst personas, geometric mean aggregation.",
    },
    "debate": {
        "label": "Debate", "class": "research", "uses_kg": True,
        "has_verdict": False, "calls": "11",
        "desc": "Multi-agent debate round. Most expensive.",
    },
}

# Canonical engine display order
ENGINE_ORDER = ["structural", "actor", "graphrag", "naked", "perspective", "debate"]


def _clean_question(raw: str) -> tuple[str, str]:
    """Split 'Event Title?: Market Question?' into (event_title, market_question).

    Kalshi questions are stored as 'Event?: Specific Question?'.
    Returns (group_title, display_question). If no separator, both are the raw string.
    """
    if "?: " in raw:
        parts = raw.split("?: ", 1)
        return parts[0].rstrip("?"), parts[1].rstrip("?")
    if ": " in raw and raw.index(": ") < len(raw) // 2:
        parts = raw.split(": ", 1)
        return parts[0].rstrip("?"), parts[1].rstrip("?")
    return "", raw.rstrip("?")


def _enrich_forecast(q: dict, today: date) -> dict:
    """Add computed display fields to a forecast dict."""
    meta = q.get("target_metadata") or {}
    engine = q.get("engine", "")
    info = ENGINE_INFO.get(engine, {})

    q["ticker"] = meta.get("kalshi_ticker", "")
    q["market_prob"] = meta.get("kalshi_implied")
    q["verdict"] = meta.get("verdict")
    q["confidence"] = meta.get("confidence")
    q["full_question"] = q.get("question", "")
    q["engine_info"] = info
    q["engine_class"] = info.get("class", "unknown")
    q["engine_label"] = info.get("label", engine or "?")

    # Clean question text
    group_title, display_q = _clean_question(q.get("question", ""))
    q["group_title"] = group_title
    q["display_question"] = display_q

    # Source type
    tv = q.get("target_variable", "")
    if tv == "kalshi_aligned":
        q["source_type"] = "kalshi"
    elif tv == "kg_native":
        q["source_type"] = "kg"
    elif tv == "hindcast":
        q["source_type"] = "hindcast"
    else:
        q["source_type"] = "other"

    # Gap vs market
    if q.get("market_prob") is not None and q.get("probability") is not None:
        q["gap_pp"] = round(abs(q["probability"] - q["market_prob"]) * 100, 1)
    else:
        q["gap_pp"] = None

    # Resolution timing
    res_date_raw = q.get("resolution_date")
    if res_date_raw:
        try:
            res_date = date.fromisoformat(str(res_date_raw)) if isinstance(res_date_raw, str) else res_date_raw
            q["days_until_resolution"] = (res_date - today).days
            q["resolution_date_str"] = res_date.isoformat()
        except (ValueError, TypeError):
            q["days_until_resolution"] = None
            q["resolution_date_str"] = None
    else:
        q["days_until_resolution"] = None
        q["resolution_date_str"] = None

    # Status
    if q.get("outcome_status") == "resolved":
        q["status_label"] = "hit" if q.get("brier_score", 1) < 0.25 else "miss"
    else:
        q["status_label"] = "pending"

    return q


def _group_by_market(forecasts: list[dict]) -> list[dict]:
    """Group forecasts by unique market question. Each group contains engine results.

    Returns list of market dicts, each with:
      - display_question, group_title, ticker, market_prob, source_type, etc.
      - engines: dict[engine_name -> forecast dict]
      - engine_list: ordered list of (engine_name, forecast) for template iteration
    """
    # Group by cleaned question text (dedup across engines)
    markets: OrderedDict[str, dict] = OrderedDict()

    for q in forecasts:
        # Use ticker as key if available (most precise), else cleaned question
        key = q.get("ticker") or q["display_question"]
        engine = q.get("engine", "unknown")

        if key not in markets:
            markets[key] = {
                "key": key,
                "display_question": q["display_question"],
                "group_title": q["group_title"],
                "ticker": q.get("ticker", ""),
                "market_prob": q.get("market_prob"),
                "source_type": q["source_type"],
                "days_until_resolution": q.get("days_until_resolution"),
                "resolution_date_str": q.get("resolution_date_str"),
                "generated_for": q.get("generated_for"),
                "status_label": q.get("status_label", "pending"),
                "outcome_status": q.get("outcome_status"),
                "resolved_bool": q.get("resolved_bool"),
                "brier_score": q.get("brier_score"),
                "engines": {},
            }

        # Keep the best (most recent) result per engine
        if engine not in markets[key]["engines"]:
            markets[key]["engines"][engine] = q
        # Update market_prob if we have it from any engine
        if q.get("market_prob") is not None:
            markets[key]["market_prob"] = q["market_prob"]

    # Build ordered engine list and compute consensus
    result = []
    for market in markets.values():
        engine_list = []
        probs = []
        for eng_name in ENGINE_ORDER:
            if eng_name in market["engines"]:
                eq = market["engines"][eng_name]
                engine_list.append((eng_name, eq))
                if eq.get("probability") is not None:
                    probs.append(eq["probability"])

        # Add any engines not in ENGINE_ORDER
        for eng_name, eq in market["engines"].items():
            if eng_name not in ENGINE_ORDER:
                engine_list.append((eng_name, eq))
                if eq.get("probability") is not None:
                    probs.append(eq["probability"])

        market["engine_list"] = engine_list
        market["engine_count"] = len(engine_list)
        market["consensus_prob"] = round(sum(probs) / len(probs), 3) if probs else None

        # Spread: how much engines disagree
        if len(probs) >= 2:
            market["spread_pp"] = round((max(probs) - min(probs)) * 100, 1)
        else:
            market["spread_pp"] = None

        result.append(market)

    return result


def _group_by_event(markets: list[dict]) -> list[dict]:
    """Group markets by event title (e.g., all UK PM candidates together).

    Returns list of event groups, each with title and list of markets.
    """
    events: OrderedDict[str, dict] = OrderedDict()

    for market in markets:
        title = market["group_title"] or "Standalone"
        if title not in events:
            events[title] = {
                "title": title,
                "markets": [],
                "source_type": market["source_type"],
            }
        events[title]["markets"].append(market)

    return list(events.values())


@router.get("/predictions")
async def predictions_page(request: Request):
    """Predictions results page."""
    store = get_store(request)
    templates = get_templates(request)
    today = date.today()

    end = today
    start = end - timedelta(days=PREDICTIONS_LOOKBACK_DAYS)
    questions = await store.get_forecast_questions_between(start=start, end=end)

    all_forecasts = [_enrich_forecast(q, today) for q in questions]

    # Split by resolution status
    resolved_raw = []
    pending_raw = []
    for q in all_forecasts:
        if q.get("outcome_status") == "resolved":
            resolved_raw.append(q)
        else:
            pending_raw.append(q)

    # Group pending by market, then by event
    pending_markets = _group_by_market(pending_raw)
    pending_events = _group_by_event(pending_markets)

    # Group resolved by market
    resolved_markets = _group_by_market(resolved_raw)

    # Count unique markets and engines involved
    all_engines_seen = set()
    for q in all_forecasts:
        all_engines_seen.add(q.get("engine", ""))
    engines_active = [e for e in ENGINE_ORDER if e in all_engines_seen]

    # Summary stats
    unique_markets = len(set(q.get("ticker") or q["display_question"] for q in all_forecasts))

    brier_scores = [q["brier_score"] for q in resolved_raw if q.get("brier_score") is not None]
    mean_brier = round(sum(brier_scores) / len(brier_scores), 4) if brier_scores else None

    market_briers = []
    for q in resolved_raw:
        meta = q.get("target_metadata") or {}
        mp = meta.get("kalshi_implied")
        if mp is not None and q.get("resolved_bool") is not None:
            outcome = 1.0 if q["resolved_bool"] else 0.0
            market_briers.append(round((float(mp) - outcome) ** 2, 4))
    market_mean_brier = round(sum(market_briers) / len(market_briers), 4) if market_briers else None

    beat_market = None
    if mean_brier is not None and market_mean_brier is not None:
        beat_market = mean_brier < market_mean_brier

    summary = {
        "unique_markets": unique_markets,
        "engines_active": len(engines_active),
        "total_forecasts": len(all_forecasts),
        "pending": len(pending_raw),
        "resolved": len(resolved_raw),
        "mean_brier": mean_brier,
        "market_mean_brier": market_mean_brier,
        "beat_market": beat_market,
    }

    return templates.TemplateResponse(request, "predictions.html", {
        "pending_events": pending_events,
        "resolved_markets": resolved_markets,
        "summary": summary,
        "engines_active": engines_active,
        "engine_info": ENGINE_INFO,
    })
