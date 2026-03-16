"""Structural prediction engine — 3 LLM calls with research-backed prompting.

Architecture (adapted from AIA Forecaster + Halawi et al. + ForecastBench):
  Call 1: Base rate + evidence assessment
  Call 2: Contrarian pass
  Call 3: Supervisor reconciliation

Always forecasts — LLM world knowledge as base, KG evidence raises confidence.
"""

from __future__ import annotations

import json
import logging
from datetime import date

from nexus.engine.projection.evidence import (
    EvidencePackage,
    assemble_evidence_package,
    format_evidence_section,
)
from nexus.engine.projection.models import (
    EvidenceFactor,
    StructuralAssessment,
)

logger = logging.getLogger(__name__)

# ── Prompt templates ─────────────────────────────────────────────────

_BASE_RATE_SYSTEM = """\
You are a superforecaster — calibrated, base-rate aware, and evidence-driven.
You produce structured assessments, not vague hedges. When evidence is weak,
you still make a call at low confidence rather than abstaining.

Today's date: {as_of}

Respond ONLY with valid JSON matching this schema:
{{
  "verdict": "yes" | "no" | "uncertain",
  "confidence": "high" | "medium" | "low",
  "factors": [
    {{"factor": "description", "direction": "supports_yes" | "supports_no" | "ambiguous",
      "weight": "strong" | "moderate" | "weak", "source_type": "trajectory | convergence | causal_chain | relationship_change | cross_topic | divergence | world_knowledge"}}
  ],
  "reasoning": "2-4 sentence synthesis",
  "base_rate_reasoning": "What reference class applies? What is the base rate?",
  "key_uncertainties": ["..."],
  "signposts": ["What to watch for..."]
}}

Rules:
- "uncertain" is ONLY for genuine 50/50 toss-ups, NOT for lack of evidence.
- Low confidence ≠ uncertain. If you lean yes/no but weakly, use verdict=yes/no + confidence=low.
- HIGH confidence: 3+ strong aligned factors, or KG evidence strongly directional.
- MEDIUM: mix of factors, or world-knowledge-only with clear lean.
- LOW: weak evidence, or world-knowledge-only with slight lean.
"""

_BASE_RATE_USER = """\
## Question
{question}

{evidence_section}

## Instructions
1. Identify the best reference class for this type of event. What is the base rate?
2. Using your world knowledge AND the intelligence evidence above, list reasons \
for YES and reasons for NO. Rate each reason's strength (strong/moderate/weak).
3. Aggregate like a superforecaster. Consider base rates, then adjust for evidence.
4. Produce your verdict and confidence.
"""

_CONTRARIAN_SYSTEM = """\
You are a contrarian analyst. Your job is to challenge the obvious answer.
You must argue the OTHER side, find wildcards, and critique base-rate reasoning.

Today's date: {as_of}

Respond ONLY with valid JSON:
{{
  "verdict": "yes" | "no" | "uncertain",
  "confidence": "high" | "medium" | "low",
  "contrarian_argument": "The strongest case against the obvious answer",
  "wildcards": ["Events that could flip the outcome"],
  "base_rate_critique": "Is the base rate being over/under-weighted?"
}}
"""

_CONTRARIAN_USER = """\
## Question
{question}

{evidence_section}

A base-rate analyst assessed this question and found:
- Verdict: {analyst1_verdict} (confidence: {analyst1_confidence})
- Reasoning: {analyst1_reasoning}

## Your task
1. What's the strongest case AGAINST their answer?
2. What wildcard events could flip the outcome?
3. Is the base rate being over/under-weighted?
4. Your own independent verdict and confidence.
"""

_SUPERVISOR_SYSTEM = """\
You are a supervising forecaster reconciling two independent analysts.
Produce the final assessment by weighing the strength of each analyst's reasoning.

Today's date: {as_of}

Rules:
- If they agree: adopt the verdict. If reasoning aligns, confidence = max(both).
- If they disagree: determine whose reasoning is more compelling. Use the stronger \
verdict but cap confidence at MEDIUM unless evidence is overwhelming.
- Always produce factors, uncertainties, and signposts.

Respond ONLY with valid JSON:
{{
  "verdict": "yes" | "no" | "uncertain",
  "confidence": "high" | "medium" | "low",
  "factors": [
    {{"factor": "...", "direction": "supports_yes" | "supports_no" | "ambiguous",
      "weight": "strong" | "moderate" | "weak", "source_type": "..."}}
  ],
  "reasoning": "2-4 sentence reconciliation",
  "contrarian_view": "Summary of the contrarian's best point",
  "key_uncertainties": ["..."],
  "signposts": ["..."]
}}
"""

_SUPERVISOR_USER = """\
## Question
{question}

## Analyst 1 — Base Rate + Evidence
- Verdict: {analyst1_verdict} ({analyst1_confidence})
- Reasoning: {analyst1_reasoning}
- Base rate: {analyst1_base_rate}

## Analyst 2 — Contrarian
- Verdict: {analyst2_verdict} ({analyst2_confidence})
- Contrarian argument: {analyst2_argument}
- Wildcards: {analyst2_wildcards}
- Base rate critique: {analyst2_critique}

## Reconcile
Produce your final structured assessment.
"""


# ── JSON parsing helpers ─────────────────────────────────────────────


def _parse_json(text: str) -> dict:
    """Extract JSON from LLM response, handling markdown code fences."""
    text = text.strip()
    if text.startswith("```"):
        # Remove code fence
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)
    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON object in text
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            result = json.loads(text[start:end])
        else:
            raise
    # Ensure we always return a dict
    if isinstance(result, list):
        return result[0] if result and isinstance(result[0], dict) else {}
    return result if isinstance(result, dict) else {}


# ── Core prediction function ─────────────────────────────────────────


async def predict_structural(
    llm,
    evidence: EvidencePackage,
    *,
    config_key: str = "knowledge_summary",
) -> StructuralAssessment:
    """Run the 3-call structural prediction pipeline.

    Call 1: Base rate + evidence assessment
    Call 2: Contrarian pass
    Call 3: Supervisor reconciliation

    Returns a StructuralAssessment with verdict, confidence, and structured factors.
    """
    as_of_str = evidence.as_of.isoformat()
    evidence_section = format_evidence_section(evidence)
    has_kg = bool(evidence.entities or evidence.threads or evidence.recent_events)

    # ── Call 1: Base rate analyst ────────────────────────────────────
    call1_response = await llm.complete(
        config_key,
        _BASE_RATE_SYSTEM.format(as_of=as_of_str),
        _BASE_RATE_USER.format(
            question=evidence.question,
            evidence_section=evidence_section,
        ),
        json_response=True,
    )
    analyst1 = _parse_json(call1_response)

    # ── Call 2: Contrarian analyst ──────────────────────────────────
    call2_response = await llm.complete(
        config_key,
        _CONTRARIAN_SYSTEM.format(as_of=as_of_str),
        _CONTRARIAN_USER.format(
            question=evidence.question,
            evidence_section=evidence_section,
            analyst1_verdict=analyst1.get("verdict", "uncertain"),
            analyst1_confidence=analyst1.get("confidence", "low"),
            analyst1_reasoning=analyst1.get("reasoning", ""),
        ),
        json_response=True,
    )
    analyst2 = _parse_json(call2_response)

    # ── Call 3: Supervisor reconciliation ───────────────────────────
    call3_response = await llm.complete(
        config_key,
        _SUPERVISOR_SYSTEM.format(as_of=as_of_str),
        _SUPERVISOR_USER.format(
            question=evidence.question,
            analyst1_verdict=analyst1.get("verdict", "uncertain"),
            analyst1_confidence=analyst1.get("confidence", "low"),
            analyst1_reasoning=analyst1.get("reasoning", ""),
            analyst1_base_rate=analyst1.get("base_rate_reasoning", ""),
            analyst2_verdict=analyst2.get("verdict", "uncertain"),
            analyst2_confidence=analyst2.get("confidence", "low"),
            analyst2_argument=analyst2.get("contrarian_argument", ""),
            analyst2_wildcards=", ".join(analyst2.get("wildcards", [])),
            analyst2_critique=analyst2.get("base_rate_critique", ""),
        ),
        json_response=True,
    )
    final = _parse_json(call3_response)

    # ── Build StructuralAssessment ──────────────────────────────────
    factors = []
    for f in final.get("factors", []):
        try:
            factors.append(EvidenceFactor(
                factor=f.get("factor", ""),
                direction=f.get("direction", "ambiguous"),
                weight=f.get("weight", "moderate"),
                source_type=f.get("source_type", "world_knowledge"),
            ))
        except Exception:
            logger.debug("Skipping malformed factor: %s", f)

    return StructuralAssessment(
        question=evidence.question,
        verdict=final.get("verdict", "uncertain"),
        confidence=final.get("confidence", "low"),
        factors=factors,
        reasoning=final.get("reasoning", ""),
        contrarian_view=final.get("contrarian_view", ""),
        key_uncertainties=final.get("key_uncertainties", []),
        signposts=final.get("signposts", []),
        base_rate_reasoning=analyst1.get("base_rate_reasoning", ""),
        has_kg_evidence=has_kg,
    )


# ── BenchmarkEngine adapter ─────────────────────────────────────────


class StructuralBenchmarkEngine:
    """Adapter satisfying the BenchmarkEngine protocol."""

    engine_name = "structural"

    async def predict_probability(
        self,
        question: str,
        *,
        llm=None,
        store=None,
        market_prob: float | None = None,
        as_of: date | None = None,
    ) -> float:
        """Assemble evidence, run structural prediction, return implied probability.

        market_prob is intentionally ignored — this engine is independent.
        """
        cutoff = as_of or date.today()

        # Assemble evidence from knowledge store
        evidence = await assemble_evidence_package(store, question, as_of=cutoff)

        # Run structural prediction
        assessment = await predict_structural(llm, evidence)

        return assessment.implied_probability
