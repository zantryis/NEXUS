"""Predictions dashboard — grouped by market, engine comparison inline."""

from collections import OrderedDict
from datetime import date, timedelta

from fastapi import APIRouter, Request

from nexus.web.app import get_store, get_templates

router = APIRouter()


def _derive_verdict(prob: float | None) -> str | None:
    """Deterministic verdict from probability. Same scale for all engines."""
    if prob is None:
        return None
    if prob >= 0.65:
        return "yes"
    if prob <= 0.35:
        return "no"
    return "uncertain"

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
    q["market_prob"] = meta.get("kalshi_implied") or meta.get("market_prob")
    q["verdict"] = _derive_verdict(q.get("probability"))
    q["confidence"] = meta.get("confidence")
    q["full_question"] = q.get("question", "")
    q["engine_info"] = info
    q["engine_class"] = info.get("class", "unknown")
    q["engine_label"] = info.get("label", engine or "?")

    # Run mode: anchored (engine saw market price) vs independent (blind)
    q["run_label"] = meta.get("run_label")

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
    elif tv == "kalshi_benchmark":
        q["source_type"] = "benchmark"
    else:
        q["source_type"] = "other"

    # Infer run mode for non-benchmark predictions
    if q["run_label"] is None:
        if q["source_type"] == "kalshi":
            q["run_label"] = "anchored"
        elif q["source_type"] in ("kg", "hindcast", "other"):
            q["run_label"] = "independent"

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
        # Use ticker as key if available (most precise), else cleaned question.
        # Include run_label so independent and anchored runs for the same
        # question stay in separate groups.
        base_key = q.get("ticker") or q["display_question"]
        rl = q.get("run_label") or ""
        key = f"{base_key}|{rl}" if rl else base_key
        engine = q.get("engine", "unknown")

        if key not in markets:
            markets[key] = {
                "key": key,
                "display_question": q["display_question"],
                "group_title": q["group_title"],
                "ticker": q.get("ticker", ""),
                "market_prob": q.get("market_prob"),
                "source_type": q["source_type"],
                "run_label": q.get("run_label"),
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

        # Market-level status: hit if best engine's Brier < 0.25
        briers = [eq.get("brier_score") for _, eq in engine_list if eq.get("brier_score") is not None]
        if briers:
            best_brier = min(briers)
            market["status_label"] = "hit" if best_brier < 0.25 else "miss"

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

    # Group resolved by source type
    resolved_by_source: OrderedDict[str, list] = OrderedDict()
    for source in ["benchmark", "kalshi", "hindcast", "kg", "other"]:
        source_markets = [m for m in resolved_markets if m["source_type"] == source]
        if source_markets:
            resolved_by_source[source] = source_markets

    # Count unique markets and engines involved
    all_engines_seen = set()
    for q in all_forecasts:
        all_engines_seen.add(q.get("engine", ""))
    engines_active = [e for e in ENGINE_ORDER if e in all_engines_seen]

    # ── Per-engine analytics ──
    engine_stats: dict[str, dict] = {}
    for q in resolved_raw:
        eng = q.get("engine", "")
        if not eng:
            continue
        if eng not in engine_stats:
            engine_stats[eng] = {
                "brier_sum": 0, "count": 0, "hits": 0,
                "by_source": {}, "by_run_label": {},
            }
        bs = q.get("brier_score")
        if bs is not None:
            engine_stats[eng]["brier_sum"] += bs
            engine_stats[eng]["count"] += 1
            if bs < 0.25:
                engine_stats[eng]["hits"] += 1
            # By source
            src = q.get("source_type", "other")
            if src not in engine_stats[eng]["by_source"]:
                engine_stats[eng]["by_source"][src] = {"brier_sum": 0, "count": 0}
            engine_stats[eng]["by_source"][src]["brier_sum"] += bs
            engine_stats[eng]["by_source"][src]["count"] += 1
            # By run label
            rl = q.get("run_label") or "unknown"
            if rl not in engine_stats[eng]["by_run_label"]:
                engine_stats[eng]["by_run_label"][rl] = {"brier_sum": 0, "count": 0, "hits": 0}
            engine_stats[eng]["by_run_label"][rl]["brier_sum"] += bs
            engine_stats[eng]["by_run_label"][rl]["count"] += 1
            if bs < 0.25:
                engine_stats[eng]["by_run_label"][rl]["hits"] += 1

    # Finalize stats
    for stats in engine_stats.values():
        stats["mean_brier"] = round(stats["brier_sum"] / stats["count"], 4) if stats["count"] else None
        stats["hit_rate"] = round(stats["hits"] / stats["count"] * 100, 1) if stats["count"] else None
        for sub in list(stats["by_source"].values()) + list(stats["by_run_label"].values()):
            sub["mean_brier"] = round(sub["brier_sum"] / sub["count"], 4) if sub["count"] else None
            sub["hit_rate"] = round(sub.get("hits", 0) / sub["count"] * 100, 1) if sub["count"] else None

    # ── Market baseline Brier ──
    market_briers = []
    for q in resolved_raw:
        mp = q.get("market_prob")
        if mp is not None and q.get("resolved_bool") is not None:
            outcome = 1.0 if q["resolved_bool"] else 0.0
            market_briers.append((float(mp) - outcome) ** 2)
    market_mean_brier = round(sum(market_briers) / len(market_briers), 4) if market_briers else None

    # ── Build analytics tables (sorted by Brier, split by run mode) ──
    def _build_analytics_table(run_label: str) -> list[dict]:
        rows = []
        for eng, stats in engine_stats.items():
            rl_stats = stats.get("by_run_label", {}).get(run_label)
            if not rl_stats or rl_stats["mean_brier"] is None:
                continue
            info = ENGINE_INFO.get(eng, {})
            vs_market = (rl_stats["mean_brier"] - market_mean_brier) if market_mean_brier is not None else 0
            rows.append({
                "engine": eng,
                "label": info.get("label", eng),
                "engine_class": info.get("class", "unknown"),
                "brier": rl_stats["mean_brier"],
                "vs_market": round(vs_market, 4),
                "hit_rate": rl_stats.get("hit_rate", 0),
                "count": rl_stats["count"],
            })
        return sorted(rows, key=lambda r: r["brier"])

    analytics_anchored = _build_analytics_table("anchored")
    analytics_independent = _build_analytics_table("independent")

    # ── Best engine per run mode ──
    best_by_run: dict[str, dict] = {}
    for table, label in [(analytics_anchored, "anchored"), (analytics_independent, "independent")]:
        if table:
            best = table[0]  # already sorted
            best_by_run[label] = {
                "engine": best["engine"],
                "label": best["label"],
                "brier": best["brier"],
                "vs_market": best["vs_market"],
            }

    # Pick overall best for summary display (prefer anchored if available)
    best_engine = best_by_run.get("anchored") or best_by_run.get("independent")
    best_run_mode = "anchored" if "anchored" in best_by_run else ("independent" if "independent" in best_by_run else None)
    beat_market = best_engine["vs_market"] < 0 if best_engine and best_engine.get("vs_market") is not None else None

    # Summary stats
    unique_markets = len(set(q.get("ticker") or q["display_question"] for q in all_forecasts))
    summary = {
        "unique_markets": unique_markets,
        "engines_active": len(engines_active),
        "total_forecasts": len(all_forecasts),
        "pending": len(pending_raw),
        "resolved": len(resolved_raw),
        "best_engine": best_engine,
        "best_run_mode": best_run_mode,
        "best_by_run": best_by_run,
        "market_mean_brier": market_mean_brier,
        "beat_market": beat_market,
    }

    return templates.TemplateResponse(request, "predictions.html", {
        "pending_events": pending_events,
        "resolved_by_source": resolved_by_source,
        "analytics_anchored": analytics_anchored,
        "analytics_independent": analytics_independent,
        "summary": summary,
        "engines_active": engines_active,
        "engine_info": ENGINE_INFO,
        "engine_order": ENGINE_ORDER,
    })
