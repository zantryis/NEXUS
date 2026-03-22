"""Casefile web routes and background rebuild endpoints."""

from __future__ import annotations

import asyncio
import html
import logging
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse

from nexus.casefiles.builder import build_casefile
from nexus.casefiles.models import (
    BuildStatus,
    CaseAssessment,
    CaseHypothesis,
    EvidenceItem,
    ExtractedDocument,
)
from nexus.casefiles.runtime import (
    build_case_divergence_models,
    build_case_graph_summary,
    build_case_thread_models,
    build_recent_changes,
)
from nexus.casefiles.qa import answer_case_question
from nexus.casefiles.storage import case_dir, load_case, list_cases
from nexus.web.app import get_store, get_templates

logger = logging.getLogger(__name__)

router = APIRouter()

MAX_QUESTION_LENGTH = 2000
CASE_CHAT_DAILY_LIMIT = 10
DAY_SECONDS = 86400


def _track_background_task(request: Request, task: asyncio.Task) -> asyncio.Task:
    tasks = getattr(request.app.state, "background_tasks", None)
    if tasks is None:
        request.app.state.background_tasks = set()
        tasks = request.app.state.background_tasks
    tasks.add(task)
    task.add_done_callback(tasks.discard)
    return task


def _case_status_store(request: Request) -> dict[str, BuildStatus]:
    if not hasattr(request.app.state, "casefile_statuses"):
        request.app.state.casefile_statuses = {}
    return request.app.state.casefile_statuses


def _set_status(
    request: Request,
    slug: str,
    *,
    status: str,
    label: str,
    error: str | None = None,
    presentable: bool | None = None,
) -> BuildStatus:
    payload = BuildStatus(
        slug=slug,
        status=status,
        label=label,
        error=error,
        presentable=presentable,
        updated_at=datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    )
    _case_status_store(request)[slug] = payload
    return payload


def _status_fragment(status: BuildStatus) -> HTMLResponse:
    classes = {
        "running": "status-running",
        "failed": "status-error",
        "completed": "status-done",
        "idle": "",
    }
    message = html.escape(status.label)
    if status.error:
        message = f"{message}: {html.escape(status.error)}"

    attrs = ""
    if status.status == "running":
        attrs = ' hx-get="/api/casefiles/{slug}/status" hx-trigger="every 3s" hx-swap="outerHTML"'.format(
            slug=status.slug
        )
        message = f'<div class="pipeline-spinner"></div> {message}'

    badge = ""
    if status.status == "completed" and status.presentable is not None:
        badge_text = "Presentable" if status.presentable else "Draft"
        badge_class = "casefile-pill-presentable" if status.presentable else "casefile-pill-draft"
        badge = f' <span class="casefile-pill {badge_class}">{badge_text}</span>'

    return HTMLResponse(
        f'<div id="casefile-build-status" class="pipeline-controls-status {classes.get(status.status, "")}"{attrs}>{message}{badge}</div>'
    )


def _check_case_chat_limit(request: Request, slug: str) -> tuple[bool, int]:
    if not hasattr(request.app.state, "case_chat_limits"):
        request.app.state.case_chat_limits = defaultdict(list)

    limits = request.app.state.case_chat_limits
    ip = request.client.host if request.client else "unknown"
    key = f"{slug}:{ip}"
    now = time.time()
    limits[key] = [entry for entry in limits[key] if now - entry < DAY_SECONDS]
    remaining = CASE_CHAT_DAILY_LIMIT - len(limits[key])
    if remaining <= 0:
        return False, 0
    return True, remaining


def _record_case_chat(request: Request, slug: str) -> None:
    limits = request.app.state.case_chat_limits
    ip = request.client.host if request.client else "unknown"
    key = f"{slug}:{ip}"
    limits[key].append(time.time())


async def _overlay_bundle_from_store(store, loaded):
    """Overlay live DB-backed case state onto the cached bundle snapshot."""
    case_row = await store.get_case(loaded.case.slug)
    if case_row is None or loaded.bundle is None:
        return loaded.bundle

    case_id = case_row["id"]
    bundle = loaded.bundle.model_copy(deep=True)
    documents = await store.get_case_documents(case_id)
    evidence = await store.get_case_evidence(case_id)
    hypotheses = await store.get_case_hypotheses(case_id)
    assessments = await store.get_case_assessments(case_id)
    open_questions = await store.get_case_open_questions(case_id)
    threads = await store.get_threads_for_case(case_id)
    divergence = await store.get_case_divergence(case_id)
    convergence = await store.get_case_convergence(case_id)
    graph_data = await store.get_case_graph_data(case_id)

    if documents:
        bundle.documents = [ExtractedDocument.model_validate(item) for item in documents]
    if evidence:
        bundle.evidence = [EvidenceItem.model_validate(item) for item in evidence]
    if hypotheses:
        bundle.hypotheses = [CaseHypothesis.model_validate(item) for item in hypotheses]
    if assessments:
        bundle.assessments = [CaseAssessment.model_validate(item) for item in assessments]
    if open_questions:
        bundle.open_questions = open_questions

    bundle.threads = build_case_thread_models(threads, divergence, convergence)
    bundle.divergence = build_case_divergence_models(divergence)
    bundle.graph = build_case_graph_summary(graph_data)
    bundle.recent_changes = build_recent_changes(
        loaded.case,
        bundle.assessments,
        bundle.threads,
        bundle.divergence,
    )
    bundle.metadata.document_count = len(bundle.documents)
    bundle.metadata.evidence_count = len(bundle.evidence)
    return bundle


async def _render_casefile_detail(request: Request, slug: str, *, initial_tab: str = "overview"):
    templates = get_templates(request)
    data_dir = getattr(request.app.state, "data_dir", Path("data"))

    try:
        loaded = load_case(data_dir, slug)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    bundle = await _overlay_bundle_from_store(get_store(request), loaded)
    evidence_index = {item.id: item for item in bundle.evidence} if bundle else {}
    entity_index = {item.id: item for item in bundle.entities} if bundle else {}
    status = _case_status_store(request).get(slug)

    return templates.TemplateResponse(
        request,
        "casefile_detail.html",
        {
            "case": loaded.case,
            "bundle": bundle,
            "seeds": loaded.seeds.sources if loaded.seeds else [],
            "evidence_index": evidence_index,
            "entity_index": entity_index,
            "build_status": status,
            "initial_tab": initial_tab,
        },
    )


@router.get("/casefiles/")
async def casefile_index(request: Request):
    templates = get_templates(request)
    data_dir = getattr(request.app.state, "data_dir", Path("data"))
    cases = list_cases(data_dir)
    return templates.TemplateResponse(
        request,
        "casefiles_index.html",
        {
            "cases": cases,
        },
    )


@router.get("/casefiles/{slug}")
async def casefile_detail(request: Request, slug: str):
    return await _render_casefile_detail(request, slug, initial_tab="overview")


@router.get("/casefiles/{slug}/graph")
async def casefile_graph_page(request: Request, slug: str):
    return await _render_casefile_detail(request, slug, initial_tab="graph")


@router.get("/casefiles/{slug}/threads/{thread_slug}")
async def casefile_thread_detail(request: Request, slug: str, thread_slug: str):
    store = get_store(request)
    templates = get_templates(request)

    case_row = await store.get_case(slug)
    if case_row is None:
        raise HTTPException(status_code=404, detail="Case not found")

    thread = await store.get_thread(thread_slug)
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")

    if slug not in await store.get_cases_for_thread(thread["id"]):
        raise HTTPException(status_code=404, detail="Thread is not linked to this case")

    events = await store.get_events_for_thread(thread["id"])
    convergence = await store.get_convergence_for_thread(thread["id"])
    divergence = await store.get_divergence_for_thread(thread["id"])
    causal_links = await store.get_causal_links_for_thread(thread["id"])
    projection_items = await store.get_projection_items_for_thread(thread["id"])
    page = await store.get_page(f"thread:{thread_slug}")

    return templates.TemplateResponse(
        request,
        "thread.html",
        {
            "thread": thread,
            "events": events,
            "convergence": convergence,
            "divergence": divergence,
            "causal_links": causal_links,
            "topics": [f"case:{slug}"],
            "projection_items": projection_items,
            "page": page,
        },
    )


@router.post("/api/casefiles/{slug}/rebuild")
async def rebuild_casefile(request: Request, slug: str):
    data_dir = getattr(request.app.state, "data_dir", Path("data"))
    llm = getattr(request.app.state, "llm", None)
    if llm is None:
        return _status_fragment(
            _set_status(
                request,
                slug,
                status="failed",
                label="Casefile rebuild unavailable",
                error="LLM client is not initialized.",
            )
        )

    case_path = case_dir(data_dir, slug)
    if not (case_path / "case.yaml").exists():
        return _status_fragment(
            _set_status(
                request,
                slug,
                status="failed",
                label="Case not found",
                error="Missing case.yaml",
            )
        )

    existing = _case_status_store(request).get(slug)
    if existing and existing.status == "running":
        return _status_fragment(existing)

    _set_status(request, slug, status="running", label="Starting casefile rebuild")

    async def _run():
        try:
            def _progress(message: str) -> None:
                _set_status(request, slug, status="running", label=message)

            bundle = await build_casefile(case_path, llm=llm, store=get_store(request), progress=_progress)
            label = "Casefile rebuilt"
            if not bundle.review.presentable:
                label = "Casefile rebuilt with readiness blockers"
            _set_status(
                request,
                slug,
                status="completed",
                label=label,
                presentable=bundle.review.presentable,
            )
        except Exception as exc:
            logger.error("Casefile rebuild failed for %s: %s", slug, exc, exc_info=True)
            _set_status(
                request,
                slug,
                status="failed",
                label="Casefile rebuild failed",
                error=str(exc)[:240],
            )

    _track_background_task(request, asyncio.create_task(_run()))
    return _status_fragment(_case_status_store(request)[slug])


@router.get("/api/casefiles/{slug}/status")
async def casefile_status(request: Request, slug: str):
    data_dir = getattr(request.app.state, "data_dir", Path("data"))
    status = _case_status_store(request).get(slug)
    if status is not None:
        return _status_fragment(status)

    try:
        loaded = load_case(data_dir, slug)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if loaded.bundle is None:
        return _status_fragment(
            BuildStatus(slug=slug, status="idle", label="No built casefile yet")
        )

    presentable = loaded.bundle.review.presentable
    label = f"Last built {loaded.bundle.metadata.last_updated}"
    return _status_fragment(
        BuildStatus(
            slug=slug,
            status="completed",
            label=label,
            presentable=presentable,
        )
    )


@router.post("/api/casefiles/{slug}/chat")
async def casefile_chat(request: Request, slug: str):
    allowed, remaining = _check_case_chat_limit(request, slug)
    if not allowed:
        return JSONResponse(
            {"error": "Rate limit exceeded. Try again tomorrow.", "remaining": 0},
            status_code=429,
        )

    llm = getattr(request.app.state, "llm", None)
    if llm is None:
        return JSONResponse(
            {"error": "LLM client is not initialized."},
            status_code=503,
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

    data_dir = getattr(request.app.state, "data_dir", Path("data"))
    try:
        loaded = load_case(data_dir, slug)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    bundle = await _overlay_bundle_from_store(get_store(request), loaded)
    if bundle is None:
        return JSONResponse({"error": "Casefile has not been built yet."}, status_code=404)

    try:
        response = await answer_case_question(llm, bundle, question)
    except Exception as exc:
        logger.error("Casefile chat failed for %s: %s", slug, exc, exc_info=True)
        return JSONResponse({"error": "Failed to answer question."}, status_code=500)

    _record_case_chat(request, slug)
    _, remaining = _check_case_chat_limit(request, slug)
    response["remaining"] = remaining
    return JSONResponse(response)


@router.get("/api/casefiles/{slug}/graph")
async def casefile_graph(request: Request, slug: str):
    store = get_store(request)
    case_row = await store.get_case(slug)
    if case_row is None:
        raise HTTPException(status_code=404, detail="Case not found")
    return JSONResponse(await store.get_case_graph_data(case_row["id"]))


@router.get("/api/casefiles/{slug}/assessments")
async def casefile_assessments(request: Request, slug: str):
    store = get_store(request)
    case_row = await store.get_case(slug)
    if case_row is None:
        raise HTTPException(status_code=404, detail="Case not found")
    return JSONResponse({"items": await store.get_case_assessments(case_row["id"])})


@router.get("/api/casefiles/{slug}/updates")
async def casefile_updates(request: Request, slug: str):
    data_dir = getattr(request.app.state, "data_dir", Path("data"))
    try:
        loaded = load_case(data_dir, slug)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    bundle = await _overlay_bundle_from_store(get_store(request), loaded)
    if bundle is None:
        return JSONResponse({"items": []})
    return JSONResponse({"items": bundle.recent_changes})
