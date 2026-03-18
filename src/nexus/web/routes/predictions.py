"""Predictions dashboard — grouped by market, engine comparison inline."""

import re
from collections import OrderedDict
from datetime import date, timedelta

from fastapi import APIRouter, Request

_MD_BOLD_RE = re.compile(r"\*{1,2}(.+?)\*{1,2}")

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
        "has_verdict": True, "calls": "3", "tier": "production",
        "desc": "3 LLM passes: base-rate analyst, contrarian, supervisor.",
    },
    "actor": {
        "label": "Actor", "class": "production", "uses_kg": True,
        "has_verdict": True, "calls": "3-6", "tier": "production",
        "desc": "Per-actor reasoning with market-anchored synthesis.",
    },
    "graphrag": {
        "label": "GraphRAG", "class": "production", "uses_kg": True,
        "has_verdict": True, "calls": "2", "tier": "experimental",
        "desc": "Graph traversal + holistic system reasoning.",
    },
    "naked": {
        "label": "Naked", "class": "strawman", "uses_kg": False,
        "has_verdict": False, "calls": "1", "tier": "experimental",
        "desc": "LLM world knowledge only — KG contribution baseline.",
    },
    "perspective": {
        "label": "Perspective", "class": "research", "uses_kg": True,
        "has_verdict": False, "calls": "4-6", "tier": "experimental",
        "desc": "Multiple analyst personas, geometric mean aggregation.",
    },
    "debate": {
        "label": "Debate", "class": "research", "uses_kg": True,
        "has_verdict": False, "calls": "11", "tier": "experimental",
        "desc": "Multi-agent debate round. Most expensive.",
    },
}

PRODUCTION_ENGINES = {k for k, v in ENGINE_INFO.items() if v.get("tier") == "production"}
EXPERIMENTAL_ENGINES = {k for k, v in ENGINE_INFO.items() if v.get("tier") == "experimental"}

# Source types excluded from the main predictions page (shown on /benchmark)
BENCHMARK_SOURCES = {"benchmark", "hindcast"}

FEATURED_MAX = 8

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
    display_q = _MD_BOLD_RE.sub(r"\1", display_q)
    group_title = _MD_BOLD_RE.sub(r"\1", group_title) if group_title else group_title
    q["group_title"] = group_title
    q["display_question"] = display_q

    # Source type
    tv = q.get("target_variable", "")
    if tv == "kalshi_aligned":
        q["source_type"] = "kalshi"
    elif tv == "kg_native":
        q["source_type"] = "kg"
    elif tv == "topic_claim":
        q["source_type"] = "freeform"
    elif tv == "thread_development":
        q["source_type"] = "thread"
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
        elif q["source_type"] in ("kg", "freeform", "thread", "hindcast", "other"):
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


def _compute_interest_score(f: dict) -> float:
    """Score a prediction's interest level for prioritization.

    Additive scoring (~10 max): temporal urgency, contrarian signal,
    engine disagreement, reasoning depth, source type.
    Mutates f to add 'interest_score' and optional '_tag_contrarian'.
    """
    score = 0.0

    # Temporal urgency — resolving soon is inherently interesting
    days = f.get("days_until_resolution")
    if days is not None:
        if days <= 3:
            score += 4.0
        elif days <= 7:
            score += 3.0
        elif days <= 14:
            score += 1.5

    # Contrarian signal — we disagree with the market
    gap = f.get("gap_pp")
    if gap is not None:
        if gap >= 25:
            score += 3.0
        elif gap >= 15:
            score += 2.0
        elif gap >= 8:
            score += 1.0

    # Engine disagreement
    spread = f.get("spread_pp")
    if spread is not None and spread >= 20:
        score += 1.0

    # Rich reasoning available
    meta = f.get("target_metadata") or {}
    if meta.get("actor_analyses"):
        score += 0.5

    # Source type bonus — freeform claims are inherently interesting
    if f.get("source_type") == "freeform":
        score += 1.0

    # Directional contrarian tag — we say YES but market says NO, or vice versa
    prob = f.get("probability")
    market = f.get("market_prob")
    if prob is not None and market is not None:
        if (prob >= 0.65 and market <= 0.4) or (prob <= 0.35 and market >= 0.6):
            f["_tag_contrarian"] = True
            score += 1.5

    # Bold claim tag — high-confidence forward-looking predictions without market price
    if prob is not None and market is None and f.get("source_type") in ("freeform", "thread", "kg"):
        if prob >= 0.7:
            f["_tag_bold"] = True
            score += 2.0
        elif prob <= 0.15:
            f["_tag_bold"] = True
            score += 1.5

    f["interest_score"] = round(score, 1)
    return score


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

    # Exclude benchmark/hindcast from main predictions page
    live_forecasts = [q for q in all_forecasts if q["source_type"] not in BENCHMARK_SOURCES]

    # Split by resolution status
    resolved_raw = []
    pending_raw = []
    for q in live_forecasts:
        if q.get("outcome_status") == "resolved":
            resolved_raw.append(q)
        else:
            pending_raw.append(q)

    # Compute interest scores for pending forecasts
    for q in pending_raw:
        _compute_interest_score(q)

    # Group pending by market, then split into featured vs all
    pending_markets = _group_by_market(pending_raw)

    # Compute market-level interest score (max of engine interest scores)
    for market in pending_markets:
        engine_scores = []
        for _, eq in market.get("engine_list", []):
            s = eq.get("interest_score", 0)
            if s:
                engine_scores.append(s)
        market["interest_score"] = max(engine_scores) if engine_scores else 0
        # Propagate contrarian tag if any engine has it
        market["_tag_contrarian"] = any(
            eq.get("_tag_contrarian") for _, eq in market.get("engine_list", [])
        )
        # Propagate spread_pp to interest score (computed at market level)
        if market.get("spread_pp") and market["spread_pp"] >= 20:
            market["interest_score"] = round(market["interest_score"] + 1.0, 1)

    # Featured: top N by interest score (must have score > 0)
    featured_markets = sorted(
        [m for m in pending_markets if m["interest_score"] > 0],
        key=lambda m: m["interest_score"],
        reverse=True,
    )[:FEATURED_MAX]
    featured_keys = {m["key"] for m in featured_markets}

    # All active: remaining markets grouped by event
    remaining_markets = [m for m in pending_markets if m["key"] not in featured_keys]
    all_active_events = _group_by_event(remaining_markets)

    # Also provide old-style pending_events for backwards compat
    pending_events = _group_by_event(pending_markets)

    # Fetch trend data for featured markets
    for market in featured_markets:
        ticker = market.get("ticker")
        if ticker:
            try:
                # Use primary engine (first in engine_list) to avoid mixing
                # different engine runs, which produces fake "trends"
                primary_engine = market["engine_list"][0][0] if market.get("engine_list") else None
                history = await store.get_prediction_history(
                    ticker, lookback_days=14,
                    engine=primary_engine,
                    run_label=market.get("run_label"),
                )
                if len(history) >= 2:
                    delta = history[-1]["probability"] - history[0]["probability"]
                    market["trend_delta"] = round(delta * 100, 1)
                    market["trend_direction"] = "up" if delta > 0.02 else "down" if delta < -0.02 else "stable"
                    market["trend_points"] = history
            except Exception:
                pass

    # Group resolved by market
    resolved_markets = _group_by_market(resolved_raw)

    # Group resolved by source type (exclude benchmark/hindcast)
    resolved_by_source: OrderedDict[str, list] = OrderedDict()
    for source in ["kalshi", "freeform", "thread", "kg", "other"]:
        source_markets = [m for m in resolved_markets if m["source_type"] == source]
        if source_markets:
            resolved_by_source[source] = source_markets

    # Count unique markets and engines involved
    all_engines_seen = set()
    for q in live_forecasts:
        all_engines_seen.add(q.get("engine", ""))
    engines_active = [e for e in ENGINE_ORDER if e in all_engines_seen]

    # Engine analytics live on /benchmark — predictions page focuses on live forecasts

    # Summary stats
    unique_markets = len(set(q.get("ticker") or q["display_question"] for q in live_forecasts))
    summary = {
        "unique_markets": unique_markets,
        "engines_active": len(engines_active),
        "total_forecasts": len(live_forecasts),
        "pending": len(pending_raw),
        "resolved": len(resolved_raw),
        "best_engine": None,
        "best_run_mode": None,
        "best_by_run": {},
        "market_mean_brier": None,
        "beat_market": None,
    }

    return templates.TemplateResponse(request, "predictions.html", {
        "featured_markets": featured_markets,
        "all_active_events": all_active_events,
        "pending_events": pending_events,
        "resolved_by_source": resolved_by_source,
        "analytics_anchored": [],
        "analytics_independent": [],
        "summary": summary,
        "engines_active": engines_active,
        "engine_info": ENGINE_INFO,
        "engine_order": ENGINE_ORDER,
        "production_engines": PRODUCTION_ENGINES,
    })
