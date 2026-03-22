"""Acquisition pipeline for casefiles: seeds, IR expansion, HTML, and PDFs."""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import logging
import re
from pathlib import Path
from typing import Awaitable, Callable
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import httpx
import trafilatura

from nexus.agent.websearch import web_search
from nexus.casefiles.models import (
    AcquisitionQuery,
    AcquisitionResult,
    CandidateDocument,
    CaseConfig,
    FetchedDocument,
    SeedSource,
)

logger = logging.getLogger(__name__)

try:
    from pypdf import PdfReader
except ImportError:  # pragma: no cover - exercised in runtime, not logic tests
    PdfReader = None  # type: ignore[assignment]

SearchFn = Callable[[str, int], Awaitable[list[dict]]]

TRACKING_QUERY_PREFIXES = ("utm_", "fbclid", "gclid", "mc_cid", "mc_eid")
TEXT_CHAR_FALLBACK = 12000


def canonicalize_url(url: str) -> str:
    """Normalize URLs for dedupe without destroying archived or file URLs."""
    if url.startswith("file://"):
        return url

    split = urlsplit(url)
    query_items = [
        (k, v)
        for k, v in parse_qsl(split.query, keep_blank_values=True)
        if not any(k.lower().startswith(prefix) for prefix in TRACKING_QUERY_PREFIXES)
    ]
    normalized_query = urlencode(sorted(query_items))
    return urlunsplit(
        (
            split.scheme.lower(),
            split.netloc.lower(),
            split.path.rstrip("/") or split.path,
            normalized_query,
            "",
        )
    )


def _title_fingerprint(title: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()
    return re.sub(r"\s+", " ", normalized)


def infer_kind(url: str, title: str | None = None) -> str:
    """Infer source kind when search results do not declare one."""
    target = f"{url} {title or ''}".lower()
    if ".pdf" in target:
        return "report"
    if "webcache" in target or "archive" in target or "wayback" in target:
        return "archive"
    return "article"


def build_ir_queries(case: CaseConfig) -> list[AcquisitionQuery]:
    """Build a bounded retrieval plan around the case question and hypotheses."""
    title = case.title.strip()
    years = [y for y in (case.time_bounds.start, case.time_bounds.end) if y]
    year_hint = ""
    if years:
        year_hint = f" {years[0][:4]}"

    queries: list[AcquisitionQuery] = [
        AcquisitionQuery(
            query=f'"{title}" official report pdf{year_hint}',
            source_class="official",
            rationale="Primary official documentation",
        ),
        AcquisitionQuery(
            query=f'"{title}" investigation report pdf',
            source_class="investigation",
            rationale="Investigative or inquiry material",
        ),
        AcquisitionQuery(
            query=f'"{title}" technical analysis',
            source_class="analysis",
            rationale="Independent technical analysis",
        ),
        AcquisitionQuery(
            query=f'"{title}" latest developments',
            source_class="media",
            rationale="Recent reporting and updates",
        ),
        AcquisitionQuery(
            query=f'"{title}" timeline key evidence',
            source_class="analysis",
            rationale="Timeline reconstruction",
        ),
    ]

    for seed in case.hypothesis_seeds[:4]:
        queries.append(
            AcquisitionQuery(
                query=f'"{title}" {seed}',
                source_class="analysis",
                rationale=f"Hypothesis expansion: {seed}",
            )
        )

    return queries[: case.build.max_queries]


def seed_candidates(seeds: list[SeedSource]) -> list[CandidateDocument]:
    """Convert optional seed manifest entries to acquisition candidates."""
    return [
        CandidateDocument(
            id=source.id,
            label=source.label,
            url=source.url,
            kind=source.kind,
            role=source.role,
            source_class=source.source_class,
            priority=source.priority,
            notes=source.notes,
            discovered_via="seed",
        )
        for source in seeds
    ]


async def search_candidates(
    case: CaseConfig,
    *,
    search_fn: SearchFn = web_search,
) -> tuple[list[AcquisitionQuery], list[CandidateDocument]]:
    """Run bounded web search expansion for the case."""
    queries = build_ir_queries(case)

    async def _run(query: AcquisitionQuery) -> list[CandidateDocument]:
        raw_results = await asyncio.wait_for(
            search_fn(query.query, case.build.max_search_results_per_query),
            timeout=15.0,
        )
        candidates: list[CandidateDocument] = []
        for idx, result in enumerate(raw_results, start=1):
            url = result.get("url", "").strip()
            if not url:
                continue
            label = result.get("title", "").strip() or f"{query.source_class.title()} result {idx}"
            digest = hashlib.sha1(f"{query.query}|{url}".encode()).hexdigest()[:12]
            candidates.append(
                CandidateDocument(
                    id=f"search-{digest}",
                    label=label,
                    title_hint=label,
                    url=url,
                    kind=infer_kind(url, label),
                    role="secondary",
                    source_class=query.source_class,
                    priority=4,
                    discovered_via=query.query,
                    search_snippet=result.get("snippet", "").strip() or None,
                    notes=query.rationale,
                )
            )
        return candidates

    results = await asyncio.gather(*[_run(query) for query in queries], return_exceptions=True)
    candidates: list[CandidateDocument] = []
    for result in results:
        if isinstance(result, Exception):
            logger.warning("Casefile search expansion failed: %s", result)
            continue
        candidates.extend(result)
    return queries, candidates


def dedupe_candidates(candidates: list[CandidateDocument]) -> list[CandidateDocument]:
    """Remove duplicates by canonical URL, then title fingerprint."""
    seen_urls: set[str] = set()
    seen_titles: set[str] = set()
    deduped: list[CandidateDocument] = []

    for candidate in sorted(candidates, key=_candidate_sort_key, reverse=True):
        url_key = canonicalize_url(candidate.url)
        if url_key in seen_urls:
            continue

        title_key = _title_fingerprint(candidate.title_hint or candidate.label)
        if title_key and title_key in seen_titles:
            continue

        seen_urls.add(url_key)
        if title_key:
            seen_titles.add(title_key)
        deduped.append(candidate)

    return deduped


def _candidate_sort_key(candidate: CandidateDocument) -> tuple[int, int, int]:
    role_bonus = 3 if candidate.role == "primary" else 0
    kind_bonus = 1 if candidate.kind == "report" else 0
    class_bonus = {
        "official": 3,
        "investigation": 2,
        "analysis": 1,
        "media": 0,
    }.get(candidate.source_class, 0)
    return (candidate.priority, role_bonus + class_bonus, kind_bonus)


def select_balanced_candidates(
    candidates: list[CandidateDocument],
    max_documents: int,
) -> list[CandidateDocument]:
    """Select candidates while preserving source-class spread."""
    ordered = sorted(candidates, key=_candidate_sort_key, reverse=True)
    buckets: dict[str, list[CandidateDocument]] = {
        "official": [],
        "investigation": [],
        "analysis": [],
        "media": [],
    }
    for candidate in ordered:
        buckets.setdefault(candidate.source_class, []).append(candidate)

    selected: list[CandidateDocument] = []
    selected_ids: set[str] = set()

    for source_class in ("official", "investigation", "analysis", "media"):
        if buckets.get(source_class):
            choice = buckets[source_class][0]
            selected.append(choice)
            selected_ids.add(choice.id)
            if len(selected) >= max_documents:
                return selected

    for candidate in ordered:
        if candidate.id in selected_ids:
            continue
        selected.append(candidate)
        selected_ids.add(candidate.id)
        if len(selected) >= max_documents:
            break
    return selected


def _local_path_from_url(url: str) -> Path | None:
    if url.startswith("file://"):
        return Path(url[7:])
    path = Path(url)
    if path.exists():
        return path
    return None


def extract_pdf_text(data: bytes, *, max_chars: int = TEXT_CHAR_FALLBACK) -> tuple[str, str | None, str | None]:
    """Extract text from a text-based PDF. Returns (text, title, error)."""
    if PdfReader is None:
        return "", None, "pypdf not installed"

    try:
        reader = PdfReader(io.BytesIO(data))
        chunks = [(page.extract_text() or "").strip() for page in reader.pages]
        text = "\n\n".join(chunk for chunk in chunks if chunk).strip()
        title = getattr(getattr(reader, "metadata", None), "title", None)
        if len(text) < 200:
            return text[:max_chars], title, "pdf_text_too_thin"
        return text[:max_chars], title, None
    except Exception as exc:
        return "", None, f"pdf_extract_failed:{exc}"


def _truncate_excerpt(text: str, limit: int = 240) -> str | None:
    compact = " ".join(text.split())
    if not compact:
        return None
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1].rstrip() + "…"


def _extract_html_payload(html: str, url: str, title_hint: str | None) -> tuple[str, str, str | None]:
    result_json = trafilatura.extract(html, output_format="json", url=url)
    if result_json:
        try:
            data = json.loads(result_json)
            title = data.get("title") or title_hint or url
            text = (data.get("text") or "").strip()
            published_at = data.get("date")
            return text, title, published_at
        except (json.JSONDecodeError, TypeError):
            pass

    text = (trafilatura.extract(html) or "").strip()
    return text, title_hint or url, None


async def fetch_candidate_document(
    candidate: CandidateDocument,
    *,
    client: httpx.AsyncClient | None = None,
    max_text_chars: int = TEXT_CHAR_FALLBACK,
) -> FetchedDocument | None:
    """Fetch one candidate as HTML or PDF and return extracted raw text."""
    local_path = _local_path_from_url(candidate.url)
    if local_path is not None:
        if local_path.suffix.lower() == ".pdf":
            text, title, error = extract_pdf_text(local_path.read_bytes(), max_chars=max_text_chars)
            if error and len(text) < 50:
                return FetchedDocument(
                    id=candidate.id,
                    label=candidate.label,
                    url=candidate.url,
                    canonical_url=candidate.url,
                    kind=candidate.kind,
                    role=candidate.role,
                    source_class=candidate.source_class,
                    priority=candidate.priority,
                    notes=candidate.notes,
                    discovered_via=candidate.discovered_via,
                    title=title or candidate.title_hint or candidate.label,
                    raw_text=text,
                    ingestion_status="unsupported_pdf",
                    ingestion_error=error,
                )
            return FetchedDocument(
                id=candidate.id,
                label=candidate.label,
                url=candidate.url,
                canonical_url=candidate.url,
                kind=candidate.kind,
                role=candidate.role,
                source_class=candidate.source_class,
                priority=candidate.priority,
                notes=candidate.notes,
                discovered_via=candidate.discovered_via,
                title=title or candidate.title_hint or candidate.label,
                raw_text=text,
                excerpt=_truncate_excerpt(text),
                ingestion_error=error,
                ingestion_status="ok" if not error else "unsupported_pdf",
            )

        html = local_path.read_text(encoding="utf-8")
        text, title, published_at = _extract_html_payload(html, candidate.url, candidate.title_hint or candidate.label)
        if not text:
            return None
        return FetchedDocument(
            id=candidate.id,
            label=candidate.label,
            url=candidate.url,
            canonical_url=candidate.url,
            kind=candidate.kind,
            role=candidate.role,
            source_class=candidate.source_class,
            priority=candidate.priority,
            notes=candidate.notes,
            discovered_via=candidate.discovered_via,
            title=title,
            published_at=published_at,
            raw_text=text[:max_text_chars],
            excerpt=_truncate_excerpt(text),
        )

    owns_client = client is None
    if owns_client:
        client = httpx.AsyncClient(
            follow_redirects=True,
            headers={"User-Agent": "NexusCasefiles/0.1"},
        )
    assert client is not None

    try:
        response = await client.get(candidate.url, timeout=30.0)
        response.raise_for_status()
        final_url = canonicalize_url(str(response.url))
        content_type = response.headers.get("content-type", "").lower()

        is_pdf = final_url.lower().endswith(".pdf") or "application/pdf" in content_type
        if is_pdf:
            text, title, error = extract_pdf_text(response.content, max_chars=max_text_chars)
            if error and len(text) < 50:
                return FetchedDocument(
                    id=candidate.id,
                    label=candidate.label,
                    url=candidate.url,
                    canonical_url=final_url,
                    kind=candidate.kind,
                    role=candidate.role,
                    source_class=candidate.source_class,
                    priority=candidate.priority,
                    notes=candidate.notes,
                    discovered_via=candidate.discovered_via,
                    title=title or candidate.title_hint or candidate.label,
                    raw_text=text,
                    ingestion_status="unsupported_pdf",
                    ingestion_error=error,
                )
            return FetchedDocument(
                id=candidate.id,
                label=candidate.label,
                url=candidate.url,
                canonical_url=final_url,
                kind=candidate.kind,
                role=candidate.role,
                source_class=candidate.source_class,
                priority=candidate.priority,
                notes=candidate.notes,
                discovered_via=candidate.discovered_via,
                title=title or candidate.title_hint or candidate.label,
                raw_text=text,
                excerpt=_truncate_excerpt(text),
                ingestion_error=error,
                ingestion_status="ok" if not error else "unsupported_pdf",
            )

        html = response.text
        text, title, published_at = _extract_html_payload(html, final_url, candidate.title_hint or candidate.label)
        if not text:
            return None
        return FetchedDocument(
            id=candidate.id,
            label=candidate.label,
            url=candidate.url,
            canonical_url=final_url,
            kind=candidate.kind,
            role=candidate.role,
            source_class=candidate.source_class,
            priority=candidate.priority,
            notes=candidate.notes,
            discovered_via=candidate.discovered_via,
            title=title,
            published_at=published_at,
            raw_text=text[:max_text_chars],
            excerpt=_truncate_excerpt(text),
        )
    except Exception as exc:
        logger.warning("Failed to fetch casefile source %s: %s", candidate.url, exc)
        return None
    finally:
        if owns_client:
            await client.aclose()


async def acquire_case_documents(
    case: CaseConfig,
    *,
    seeds: list[SeedSource] | None = None,
    search_fn: SearchFn = web_search,
) -> AcquisitionResult:
    """Acquire a bounded document set using seeds plus IR expansion."""
    seed_list = seeds or []
    queries, search_results = await search_candidates(case, search_fn=search_fn)
    candidates = dedupe_candidates(seed_candidates(seed_list) + search_results)

    fetch_plan = select_balanced_candidates(
        candidates,
        max(case.build.max_documents * 2, min(len(candidates), case.build.max_documents + 4)),
    )

    async with httpx.AsyncClient(
        follow_redirects=True,
        headers={"User-Agent": "NexusCasefiles/0.1"},
    ) as client:
        fetched = await asyncio.gather(
            *[
                fetch_candidate_document(
                    candidate,
                    client=client,
                    max_text_chars=case.build.max_text_chars,
                )
                for candidate in fetch_plan
            ]
        )

    seen_urls: set[str] = set()
    seen_titles: set[str] = set()
    documents: list[FetchedDocument] = []
    for document in fetched:
        if document is None:
            continue
        url_key = canonicalize_url(document.canonical_url)
        title_key = _title_fingerprint(document.title)
        if url_key in seen_urls or (title_key and title_key in seen_titles):
            continue
        seen_urls.add(url_key)
        if title_key:
            seen_titles.add(title_key)
        if document.raw_text:
            documents.append(document)

    balanced_documents = select_balanced_candidates(
        [
            CandidateDocument(
                id=document.id,
                label=document.label,
                title_hint=document.title,
                url=document.canonical_url,
                kind=document.kind,
                role=document.role,
                source_class=document.source_class,
                priority=document.priority,
                notes=document.notes,
                discovered_via=document.discovered_via,
            )
            for document in documents
        ],
        case.build.max_documents,
    )
    selected_ids = {candidate.id for candidate in balanced_documents}

    return AcquisitionResult(
        queries=queries,
        candidates=candidates,
        documents=[document for document in documents if document.id in selected_ids],
    )
