"""Cost dashboard and API routes."""

from datetime import date, timedelta

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from nexus.web.app import get_store, get_templates

router = APIRouter()


@router.get("/cost")
async def cost_page(request: Request):
    """Cost dashboard page."""
    store = get_store(request)
    templates = get_templates(request)
    today = date.today()

    # Get last 30 days of usage
    since = today - timedelta(days=30)
    daily_costs = await store.get_usage_summary(since.isoformat())
    today_cost = await store.get_daily_cost(today.isoformat())

    return templates.TemplateResponse(request, "cost.html", {
        "today_cost": today_cost,
        "daily_costs": daily_costs,
        "today": today,
    })


@router.get("/api/cost")
async def cost_api(request: Request):
    """JSON endpoint for HTMX cost badge polling."""
    store = get_store(request)
    today_cost = await store.get_daily_cost(date.today().isoformat())

    # Also get LLM client cost summary if available
    llm = getattr(request.app.state, "llm", None)
    session_cost = {}
    if llm:
        session_cost = llm.usage.cost_summary()

    return JSONResponse({
        "today_usd": round(today_cost, 4),
        "session": session_cost,
    })
