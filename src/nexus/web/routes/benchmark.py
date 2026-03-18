"""Benchmark page — engine comparison on resolved predictions.

Separated from predictions page to keep live predictions clean.
Shows benchmark fixtures, hindcast results (dev mode), and engine Brier scores.
"""

from collections import OrderedDict
from datetime import date, timedelta

from fastapi import APIRouter, Request

from nexus.web.app import get_store, get_templates
from nexus.web.routes.predictions import (
    ENGINE_INFO,
    ENGINE_ORDER,
    _enrich_forecast,
)

router = APIRouter()

BENCHMARK_LOOKBACK_DAYS = 180


@router.get("/benchmark")
async def benchmark_page(request: Request):
    """Benchmark results page. Add ?dev=1 for hindcast/internal data."""
    store = get_store(request)
    templates = get_templates(request)
    today = date.today()
    dev_mode = request.query_params.get("dev") == "1"

    end = today
    start = end - timedelta(days=BENCHMARK_LOOKBACK_DAYS)
    questions = await store.get_forecast_questions_between(start=start, end=end)

    all_forecasts = [_enrich_forecast(q, today) for q in questions]

    # Filter to benchmark sources only
    benchmark_sources = {"benchmark"}
    if dev_mode:
        benchmark_sources.add("hindcast")

    benchmark_forecasts = [
        q for q in all_forecasts
        if q["source_type"] in benchmark_sources
    ]

    # Only resolved items
    resolved = [q for q in benchmark_forecasts if q.get("outcome_status") == "resolved"]

    # Per-engine stats
    engine_stats: dict[str, dict] = {}
    for q in resolved:
        eng = q.get("engine", "")
        if not eng:
            continue
        if eng not in engine_stats:
            engine_stats[eng] = {
                "brier_sum": 0, "count": 0, "hits": 0,
                "by_run_label": {},
            }
        bs = q.get("brier_score")
        if bs is not None:
            engine_stats[eng]["brier_sum"] += bs
            engine_stats[eng]["count"] += 1
            if bs < 0.25:
                engine_stats[eng]["hits"] += 1
            rl = q.get("run_label") or "unknown"
            if rl not in engine_stats[eng]["by_run_label"]:
                engine_stats[eng]["by_run_label"][rl] = {"brier_sum": 0, "count": 0, "hits": 0}
            engine_stats[eng]["by_run_label"][rl]["brier_sum"] += bs
            engine_stats[eng]["by_run_label"][rl]["count"] += 1
            if bs < 0.25:
                engine_stats[eng]["by_run_label"][rl]["hits"] += 1

    for stats in engine_stats.values():
        stats["mean_brier"] = round(stats["brier_sum"] / stats["count"], 4) if stats["count"] else None
        stats["hit_rate"] = round(stats["hits"] / stats["count"] * 100, 1) if stats["count"] else None
        for sub in stats["by_run_label"].values():
            sub["mean_brier"] = round(sub["brier_sum"] / sub["count"], 4) if sub["count"] else None
            sub["hit_rate"] = round(sub.get("hits", 0) / sub["count"] * 100, 1) if sub["count"] else None

    # Market baseline Brier
    market_briers = []
    for q in resolved:
        mp = q.get("market_prob")
        if mp is not None and q.get("resolved_bool") is not None:
            outcome = 1.0 if q["resolved_bool"] else 0.0
            market_briers.append((float(mp) - outcome) ** 2)
    market_mean_brier = round(sum(market_briers) / len(market_briers), 4) if market_briers else None

    # Build engine comparison table
    def _build_table(run_label: str) -> list[dict]:
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

    analytics_anchored = _build_table("anchored")
    analytics_independent = _build_table("independent")

    # Per-question detail table (dedup by ticker, keep most recent cutoff)
    seen_tickers: dict[str, dict] = {}
    for q in resolved:
        ticker = q.get("ticker") or q.get("display_question", "")
        eng = q.get("engine", "")
        key = f"{ticker}|{eng}|{q.get('run_label', '')}"
        existing = seen_tickers.get(key)
        if not existing or (q.get("generated_for", "") > existing.get("generated_for", "")):
            seen_tickers[key] = q
    # Sort by ticker then brier so hits and misses are interleaved naturally
    question_detail = sorted(seen_tickers.values(), key=lambda q: (q.get("ticker") or "", q.get("brier_score") or 1.0))

    # Unique tickers and settlement date range
    unique_tickers = len(set(q.get("ticker") for q in resolved if q.get("ticker")))
    settlement_dates = [q.get("resolution_date_str") for q in resolved if q.get("resolution_date_str")]
    date_range = f"{min(settlement_dates)} — {max(settlement_dates)}" if settlement_dates else ""

    # Best engine
    all_tables = analytics_anchored + analytics_independent
    best_engine = min(all_tables, key=lambda r: r["brier"]) if all_tables else None

    summary = {
        "total_resolved": len(resolved),
        "unique_tickers": unique_tickers,
        "date_range": date_range,
        "market_mean_brier": market_mean_brier,
        "best_engine": best_engine,
        "engines_tested": len(engine_stats),
    }

    return templates.TemplateResponse(request, "benchmark.html", {
        "summary": summary,
        "analytics_anchored": analytics_anchored,
        "analytics_independent": analytics_independent,
        "question_detail": question_detail[:200],  # cap for page size
        "engine_info": ENGINE_INFO,
        "engine_order": ENGINE_ORDER,
        "dev_mode": dev_mode,
        "market_mean_brier": market_mean_brier,
    })
