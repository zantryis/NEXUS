"""Web chat widget — Q&A via the dashboard, reusing the Telegram Q&A pipeline."""

import logging
import time
from collections import defaultdict

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from nexus.web.app import get_store

logger = logging.getLogger(__name__)

router = APIRouter()

DAILY_LIMIT = 5  # max Q&A requests per IP per day
DAY_SECONDS = 86400
MAX_QUESTION_LENGTH = 2000


def _check_rate_limit(request: Request) -> tuple[bool, int]:
    """Check IP-based rate limit. Returns (allowed, remaining)."""
    if not hasattr(request.app.state, "chat_rate_limits"):
        request.app.state.chat_rate_limits = defaultdict(list)

    limits = request.app.state.chat_rate_limits
    ip = request.client.host if request.client else "unknown"
    now = time.time()

    # Clean old entries
    limits[ip] = [t for t in limits[ip] if now - t < DAY_SECONDS]

    remaining = DAILY_LIMIT - len(limits[ip])
    if remaining <= 0:
        return False, 0

    return True, remaining


def _record_request(request: Request):
    """Record a Q&A request for rate limiting."""
    limits = request.app.state.chat_rate_limits
    ip = request.client.host if request.client else "unknown"
    limits[ip].append(time.time())


@router.post("/api/chat")
async def chat_ask(request: Request):
    """Accept a question, run Q&A pipeline, return answer as JSON."""
    allowed, remaining = _check_rate_limit(request)
    if not allowed:
        return JSONResponse(
            {"error": "Rate limit exceeded. Try again tomorrow.", "remaining": 0},
            status_code=429,
        )

    body = await request.json()
    question = (body.get("question") or "").strip()
    if not question:
        return JSONResponse({"error": "Question cannot be empty."}, status_code=400)
    if len(question) > MAX_QUESTION_LENGTH:
        return JSONResponse(
            {"error": f"Question too long (max {MAX_QUESTION_LENGTH} characters)."},
            status_code=400,
        )

    store = get_store(request)
    llm = getattr(request.app.state, "llm", None)
    config = getattr(request.app.state, "config", None)

    if not llm or not config:
        return JSONResponse(
            {"error": "Service not fully initialized. Try again later."},
            status_code=503,
        )

    try:
        from nexus.agent.qa import answer_question
        answer = await answer_question(llm, store, config, question)
        _record_request(request)
        _, new_remaining = _check_rate_limit(request)
        return JSONResponse({"answer": answer, "remaining": new_remaining})
    except Exception as e:
        logger.error(f"Chat Q&A failed: {e}", exc_info=True)
        return JSONResponse({"error": "Failed to generate answer."}, status_code=500)
