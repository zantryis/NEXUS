"""Grounded Q&A over a built casefile bundle."""

from __future__ import annotations

import json
import logging
import re

from nexus.casefiles.builder import _parse_json_payload
from nexus.casefiles.models import CasefileBundle

logger = logging.getLogger(__name__)

QA_SYSTEM_PROMPT = (
    "Answer questions using only the supplied casefile context. "
    "Do not fill gaps from general knowledge. "
    "If the casefile does not support an answer, say that directly. "
    "Return valid JSON only."
)


def _keywords(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]{3,}", text.lower())
        if token not in {"what", "when", "where", "which", "with", "from", "that", "have", "this"}
    }


def _pick_context(bundle: CasefileBundle, question: str) -> dict:
    terms = _keywords(question)

    evidence_scored: list[tuple[int, dict]] = []
    for item in bundle.evidence:
        haystack = " ".join(
            [
                item.claim,
                item.summary,
                item.document_title,
                " ".join(item.related_hypotheses),
                " ".join(item.time_anchors),
            ]
        ).lower()
        score = sum(1 for term in terms if term in haystack)
        if score:
            evidence_scored.append((score, item.model_dump()))

    doc_scored: list[tuple[int, dict]] = []
    for item in bundle.documents:
        haystack = " ".join(
            [
                item.title,
                item.summary,
                " ".join(item.time_anchors),
                " ".join(entity.name for entity in item.entities),
            ]
        ).lower()
        score = sum(1 for term in terms if term in haystack)
        if score:
            doc_scored.append(
                (
                    score,
                    {
                        "id": item.id,
                        "title": item.title,
                        "summary": item.summary,
                        "source_class": item.source_class,
                        "role": item.role,
                    },
                )
            )

    evidence = [item for _, item in sorted(evidence_scored, key=lambda pair: pair[0], reverse=True)[:8]]
    documents = [item for _, item in sorted(doc_scored, key=lambda pair: pair[0], reverse=True)[:4]]

    if not evidence:
        evidence = [item.model_dump() for item in bundle.evidence[:6]]
    if not documents:
        documents = [
            {
                "id": item.id,
                "title": item.title,
                "summary": item.summary,
                "source_class": item.source_class,
                "role": item.role,
            }
            for item in bundle.documents[:4]
        ]

    return {
        "overview": bundle.overview.model_dump(),
        "hypotheses": [item.model_dump() for item in bundle.hypotheses[:4]],
        "evidence": evidence,
        "documents": documents,
        "open_questions": bundle.open_questions[:5],
    }


async def answer_case_question(llm, bundle: CasefileBundle, question: str) -> dict:
    """Answer a case question using only the built casefile."""
    context = _pick_context(bundle, question)
    valid_refs = {item.id for item in bundle.evidence} | {item.id for item in bundle.documents}

    raw = await llm.complete(
        "agent",
        QA_SYSTEM_PROMPT,
        json.dumps(
            {
                "question": question,
                "context": context,
                "output_schema": {
                    "answer": "direct answer grounded in case context",
                    "citations": ["evidence or document ids like E001 or doc ids"],
                },
                "constraints": [
                    "Use only the provided casefile context.",
                    "Citations are required.",
                    "If the context is insufficient, say so plainly and cite the closest relevant items.",
                ],
            },
            ensure_ascii=True,
        ),
        json_response=True,
        timeout_s=90.0,
    )

    try:
        data = _parse_json_payload(raw)
    except Exception:
        logger.warning("Casefile QA returned invalid JSON")
        data = {"answer": "The casefile could not generate a grounded answer.", "citations": []}

    citations = [citation for citation in (data.get("citations") or []) if citation in valid_refs]
    if not citations:
        citations = [item.id for item in bundle.evidence[:2]]
    return {
        "answer": data.get("answer") or "The casefile does not support a grounded answer yet.",
        "citations": citations,
    }
