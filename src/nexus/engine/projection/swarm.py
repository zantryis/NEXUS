"""Swarm forecasting: multi-persona LLM ensemble with proper aggregation.

Architecture inspired by the "Wisdom of the Silicon Crowd" finding that
ensembles of diverse LLM forecasters match human prediction tournament
performance. Uses geometric mean of odds for aggregation (provably
optimal for log-scoring) and extremization to correct for the systematic
centrist bias in LLM probability estimates.

References:
- Schoenegger et al., "Wisdom of the Silicon Crowd" (Science Advances, 2024)
- Neyman & Roughgarden, "When pooling forecasts, use geometric mean of odds"
- Satopaa et al., "Combining multiple probability predictions using a
  simple logit model" (IJF, 2014) — extremization theory
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
from dataclasses import dataclass, field
from datetime import date

from nexus.engine.projection.forecasting import (
    FAMILY_TARGET_SPECS,
    ForecastEngineInput,
    ForecastQuestion,
    ForecastRun,
    _build_candidate_catalog,
    _clip_probability,
)
from nexus.engine.projection.models import CrossTopicSignal
from nexus.engine.synthesis.knowledge import NarrativeThread
from nexus.llm.client import LLMClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Persona definitions
# ---------------------------------------------------------------------------

PERSONAS: dict[str, dict] = {
    "base_rate_analyst": {
        "system": (
            "You are a calibration-focused forecasting analyst. Your PRIMARY tool is the "
            "base rate. Start from the EXACT base rate given and adjust ONLY for extraordinary "
            "evidence. Most adjustments should be TINY (2-8 percentage points). Adjustments "
            ">15pp require multiple independent confirming sources. IMPORTANT: LLMs are "
            "systematically overconfident — most events you think are 80% likely actually "
            "happen about 50% of the time. Err toward the base rate, not toward certainty."
        ),
        "weight": 1.5,
    },
    "momentum_tracker": {
        "system": (
            "You are a trend-focused analyst who reads momentum and acceleration signals. "
            "You pay close attention to event velocity, trajectory changes, and whether "
            "activity is building or fading. When momentum is strong and accelerating, you "
            "assign higher probabilities than the base rate. When momentum is decelerating "
            "or the thread has gone dormant, you assign lower probabilities. You trust "
            "trajectory data and recent event patterns more than historical averages."
        ),
        "weight": 1.0,
    },
    "evidence_skeptic": {
        "system": (
            "You are a skeptical analyst who assumes predictions are overconfident until "
            "proven otherwise. You apply regression to the mean aggressively. MOST threads "
            "do NOT produce new events within 7 days. MOST partnerships announced do NOT "
            "materialize quickly. Narrative momentum is usually noise. Your starting point "
            "is ALWAYS the base rate, and you rarely adjust more than 5pp up. You actively "
            "look for reasons the base rate is already too high. Single-source narratives "
            "get discounted to near the base rate."
        ),
        "weight": 1.2,
    },
    "cross_domain_synthesizer": {
        "system": (
            "You are a cross-domain analyst who sees connections between different topic "
            "areas. You look at cross-topic signals, shared entities, and how developments "
            "in one domain catalyze events in another. When you see strong cross-domain "
            "bridges with shared accelerating entities, you weight this heavily. When topics "
            "are isolated with no cross-domain activity, you shade toward the base rate. "
            "You specialize in seeing what single-topic analysts miss."
        ),
        "weight": 1.0,
    },
}

SWARM_PROMPT_TEMPLATE = """\
## Forecast Question
{question}

## Target
Type: {target_variable}
Horizon: {horizon_days} days (resolves {resolution_date})
Resolution: {resolution_criteria}

## Base Rates
Prior (hardcoded): {prior_base_rate:.0%}
{empirical_line}

## Thread Context
{thread_context}

## Evidence Quality
{evidence_section}

## Recent Events
{recent_events_section}

## Cross-Topic Signals
{cross_topic_section}

---

CRITICAL CALIBRATION NOTE: LLM forecasters are systematically overconfident.
Events you think are 80% likely typically happen ~50% of the time.
Your base rate anchor should be your STARTING probability.
Only adjust if you have SPECIFIC, CONCRETE evidence — not general narrative momentum.

Reason step by step:
1. State the base rate — this is your starting probability
2. List 1-2 specific factors that might adjust UP (max +10pp each, with justification)
3. List 1-2 specific factors that might adjust DOWN (max -10pp each, with justification)
4. Apply adjustments to the base rate to get your final probability (stay within ±20pp of base rate)

Return JSON only: {{"reasoning": "your step-by-step analysis", "probability": 0.XX, "confidence": "low|medium|high"}}"""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PersonaForecast:
    persona: str
    probability: float
    reasoning: str
    confidence: str = "medium"


@dataclass
class SwarmResult:
    question: str
    forecasts: list[PersonaForecast]
    aggregated_probability: float
    extremized_probability: float
    final_probability: float
    aggregation_method: str = "geometric_mean_of_odds"
    extremization_gamma: float = 2.5


# ---------------------------------------------------------------------------
# Aggregation math
# ---------------------------------------------------------------------------

def geometric_mean_of_odds(
    probabilities: list[float],
    weights: list[float] | None = None,
) -> float:
    """Aggregate probabilities via weighted geometric mean of log-odds.

    Converts each probability to log-odds, takes the weighted mean,
    then converts back. This is provably optimal under logarithmic scoring
    and naturally handles asymmetric information.
    """
    if not probabilities:
        return 0.5

    # Clip to avoid log(0)
    clipped = [max(0.02, min(0.98, p)) for p in probabilities]
    w = weights or [1.0] * len(clipped)

    total_weight = sum(w)
    if total_weight == 0:
        return 0.5

    # Convert to log-odds, take weighted mean
    log_odds = [math.log(p / (1.0 - p)) for p in clipped]
    weighted_mean = sum(lo * wi for lo, wi in zip(log_odds, w)) / total_weight

    # Convert back to probability
    result = 1.0 / (1.0 + math.exp(-weighted_mean))
    return max(0.02, min(0.98, result))


def extremize(probability: float, gamma: float = 2.5) -> float:
    """Push probability away from 0.5 toward 0 or 1.

    Uses the standard extremization formula:
        p_ext = p^gamma / (p^gamma + (1-p)^gamma)

    gamma=1.0 is the identity. gamma>1 extremizes.
    Recommended gamma ~ 2.5 based on Satopaa et al.
    """
    p = max(0.02, min(0.98, probability))

    if gamma == 1.0:
        return p

    p_gamma = p ** gamma
    q_gamma = (1.0 - p) ** gamma
    denom = p_gamma + q_gamma

    if denom == 0:
        return 0.5

    result = p_gamma / denom
    return max(0.02, min(0.98, round(result, 4)))


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------

def _find_source_thread(
    question: ForecastQuestion,
    payload: ForecastEngineInput,
) -> NarrativeThread | None:
    """Find the thread most relevant to a forecast question."""
    # Check evidence_thread_ids first
    thread_ids = set(question.evidence_thread_ids or [])
    for thread in payload.threads:
        if thread.thread_id and thread.thread_id in thread_ids:
            return thread

    # Check target_metadata thread_id
    meta_tid = (question.target_metadata or {}).get("thread_id")
    if meta_tid:
        for thread in payload.threads:
            if thread.thread_id == meta_tid:
                return thread

    # Fall back to first thread with matching entities
    anchor_entities = {
        e.lower()
        for e in (question.target_metadata or {}).get("anchor_entities", [])
    }
    if anchor_entities:
        for thread in payload.threads:
            thread_entities = {e.lower() for e in (thread.key_entities or [])}
            if anchor_entities & thread_entities:
                return thread

    return payload.threads[0] if payload.threads else None


def build_forecast_context(
    question: ForecastQuestion,
    payload: ForecastEngineInput,
    calibration_data: list[dict] | None = None,
) -> dict:
    """Extract rich context from all available signals for persona prompts."""
    thread = _find_source_thread(question, payload)

    # Empirical base rate from calibration data
    empirical_hit_rate = None
    empirical_n = 0
    if calibration_data:
        matching = [
            r for r in calibration_data
            if r.get("target_variable") == question.target_variable
        ]
        empirical_n = len(matching)
        if empirical_n > 0:
            empirical_hit_rate = sum(
                1 for r in matching if r["resolved_bool"]
            ) / empirical_n

    # Source diversity from recent events
    all_sources = []
    all_countries = set()
    all_languages = set()
    for event in payload.recent_events:
        for source in (event.sources or []):
            all_sources.append(source)
            if source.get("country"):
                all_countries.add(source["country"])
            if source.get("language"):
                all_languages.add(source["language"])

    # Convergence & divergence from thread
    convergence_facts = []
    divergence_facts = []
    if thread:
        convergence_facts = getattr(thread, "convergence", None) or []
        divergence_facts = getattr(thread, "divergence", None) or []

    return {
        "question": question.question,
        "target_variable": question.target_variable,
        "horizon_days": question.horizon_days,
        "resolution_date": question.resolution_date.isoformat() if question.resolution_date else "unknown",
        "resolution_criteria": question.resolution_criteria or "",
        "prior_base_rate": question.base_rate or 0.5,
        "empirical_hit_rate": empirical_hit_rate,
        "empirical_n": empirical_n,
        # Thread signals
        "thread_headline": thread.headline if thread else None,
        "thread_status": getattr(thread, "status", None) if thread else None,
        "thread_trajectory": getattr(thread, "trajectory_label", None) if thread else None,
        "thread_momentum": getattr(thread, "momentum_score", None) if thread else None,
        "thread_velocity": getattr(thread, "velocity_7d", None) if thread else None,
        "thread_acceleration": getattr(thread, "acceleration_7d", None) if thread else None,
        "thread_snapshot_count": getattr(thread, "snapshot_count", None) if thread else None,
        "thread_event_count": len(thread.events) if thread else 0,
        "key_entities": (thread.key_entities or []) if thread else [],
        # Evidence quality
        "convergence_facts": convergence_facts,
        "divergence_facts": divergence_facts,
        "source_count": len(all_sources),
        "country_count": len(all_countries),
        "language_count": len(all_languages),
        # Cross-topic
        "cross_topic_signals": [
            {
                "entity": s.shared_entity,
                "related_topic": s.related_topic_slug,
                "note": s.note,
            }
            for s in payload.cross_topic_signals[:5]
        ],
        # Recent events
        "recent_events": [
            {
                "date": e.date.isoformat() if isinstance(e.date, date) else str(e.date),
                "summary": e.summary[:200],
                "significance": e.significance,
                "entity_count": len(e.entities or []),
            }
            for e in payload.recent_events[:8]
        ],
    }


def _render_swarm_prompt(question: ForecastQuestion, context: dict) -> str:
    """Build the user prompt for a swarm persona from context dict."""
    # Empirical line
    if context["empirical_hit_rate"] is not None:
        empirical_line = (
            f"Empirical (observed): {context['empirical_hit_rate']:.0%} "
            f"(n={context['empirical_n']})"
        )
    else:
        empirical_line = "Empirical: no resolved data yet"

    # Thread context
    if context["thread_headline"]:
        thread_lines = [
            f"- Thread: {context['thread_headline']}",
            f"- Status: {context['thread_status']} | Trajectory: {context['thread_trajectory']}",
            f"- Momentum: {context['thread_momentum']} | Velocity: {context['thread_velocity']} events/7d",
            f"- Acceleration: {context['thread_acceleration']} | Snapshots: {context['thread_snapshot_count']}",
            f"- Events in thread: {context['thread_event_count']}",
            f"- Key entities: {', '.join(context['key_entities'][:8])}",
        ]
        thread_context = "\n".join(thread_lines)
    else:
        thread_context = "No thread context available."

    # Evidence section
    evidence_lines = []
    if context["convergence_facts"]:
        evidence_lines.append(f"Convergence ({len(context['convergence_facts'])} multi-source confirmed facts):")
        for fact in context["convergence_facts"][:3]:
            text = fact.get("fact", fact.get("fact_text", str(fact)))
            evidence_lines.append(f"  + {text}")
    if context["divergence_facts"]:
        evidence_lines.append(f"Divergence ({len(context['divergence_facts'])} conflicting framings):")
        for div in context["divergence_facts"][:3]:
            if isinstance(div, dict):
                evidence_lines.append(
                    f"  ~ {div.get('source_a', '?')}: {div.get('framing_a', '?')} vs "
                    f"{div.get('source_b', '?')}: {div.get('framing_b', '?')}"
                )
    evidence_lines.append(
        f"Source diversity: {context['source_count']} sources, "
        f"{context['country_count']} countries, {context['language_count']} languages"
    )
    evidence_section = "\n".join(evidence_lines) if evidence_lines else "No evidence quality data."

    # Recent events
    if context["recent_events"]:
        event_lines = []
        for ev in context["recent_events"][:6]:
            event_lines.append(
                f"- [{ev['date']}] (sig={ev['significance']}) {ev['summary']}"
            )
        recent_events_section = "\n".join(event_lines)
    else:
        recent_events_section = "No recent events."

    # Cross-topic signals
    if context["cross_topic_signals"]:
        signal_lines = []
        for sig in context["cross_topic_signals"][:4]:
            signal_lines.append(
                f"- {sig['entity']} bridges to {sig['related_topic']}: {sig.get('note', 'entity co-occurrence')}"
            )
        cross_topic_section = "\n".join(signal_lines)
    else:
        cross_topic_section = "No cross-topic signals detected."

    return SWARM_PROMPT_TEMPLATE.format(
        question=question.question,
        target_variable=question.target_variable,
        horizon_days=question.horizon_days,
        resolution_date=context["resolution_date"],
        resolution_criteria=context["resolution_criteria"],
        prior_base_rate=context["prior_base_rate"],
        empirical_line=empirical_line,
        thread_context=thread_context,
        evidence_section=evidence_section,
        recent_events_section=recent_events_section,
        cross_topic_section=cross_topic_section,
    )


# ---------------------------------------------------------------------------
# Swarm runner
# ---------------------------------------------------------------------------

def anchor_blend(
    swarm_probability: float,
    anchor_probability: float,
    *,
    swarm_weight: float = 0.4,
) -> float:
    """Blend the swarm's LLM estimate with the deterministic anchor.

    Uses the deterministic (well-calibrated) probability as an anchor
    and lets the swarm shift it. This prevents the swarm from overriding
    proven calibration. The swarm_weight controls how much influence
    the LLM reasoning has (0.4 = swarm moves the probability by 40%
    of the gap between anchor and swarm estimate).
    """
    blended = anchor_probability + swarm_weight * (swarm_probability - anchor_probability)
    return max(0.02, min(0.98, round(blended, 4)))


async def run_swarm(
    llm: LLMClient,
    question: ForecastQuestion,
    payload: ForecastEngineInput,
    calibration_data: list[dict] | None = None,
    *,
    personas: dict[str, dict] | None = None,
    gamma: float = 0.8,
) -> SwarmResult:
    """Run multi-persona ensemble: parallel LLM calls, aggregate, compress.

    Uses gamma < 1.0 to COMPRESS overconfident LLM estimates back toward
    0.5 (LLMs are systematically overconfident, unlike human crowds which
    are underconfident). Then blends with the deterministic anchor to
    produce the final probability.
    """
    active_personas = personas or PERSONAS
    context = build_forecast_context(question, payload, calibration_data)
    prompt = _render_swarm_prompt(question, context)

    async def call_persona(name: str, spec: dict) -> PersonaForecast | None:
        try:
            raw = await llm.complete(
                config_key="knowledge_summary",
                system_prompt=spec["system"],
                user_prompt=prompt,
                json_response=True,
            )
            data = json.loads(raw)
            prob = float(data.get("probability", 0.5))
            return PersonaForecast(
                persona=name,
                probability=max(0.02, min(0.98, prob)),
                reasoning=data.get("reasoning", ""),
                confidence=data.get("confidence", "medium"),
            )
        except Exception as exc:
            logger.warning("Swarm persona %s failed: %s", name, exc)
            return None

    results = await asyncio.gather(*[
        call_persona(name, spec)
        for name, spec in active_personas.items()
    ])

    forecasts = [r for r in results if r is not None]

    if not forecasts:
        # Total failure → fall back to deterministic probability
        return SwarmResult(
            question=question.question,
            forecasts=[],
            aggregated_probability=question.probability,
            extremized_probability=question.probability,
            final_probability=question.probability,
            extremization_gamma=gamma,
        )

    probs = [f.probability for f in forecasts]
    weights = [active_personas[f.persona]["weight"] for f in forecasts]

    aggregated = geometric_mean_of_odds(probs, weights)
    compressed = extremize(aggregated, gamma=gamma)
    # Blend with the deterministic anchor (the question's pre-swarm probability)
    final = anchor_blend(compressed, question.probability)

    return SwarmResult(
        question=question.question,
        forecasts=forecasts,
        aggregated_probability=round(aggregated, 4),
        extremized_probability=round(compressed, 4),
        final_probability=round(final, 4),
        extremization_gamma=gamma,
    )


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class SwarmForecastEngine:
    """Multi-persona LLM ensemble forecast engine.

    Generates candidate questions using the native deterministic catalog,
    then runs a swarm of diverse LLM personas on each question to produce
    well-calibrated, wide-range probability estimates.
    """

    engine_name = "swarm"

    async def generate(
        self,
        llm: LLMClient | None,
        payload: ForecastEngineInput,
        *,
        critic_pass: bool = True,
        max_questions: int = 4,
        calibration_data: list[dict] | None = None,
    ) -> ForecastRun:
        candidates = _build_candidate_catalog(
            payload, "native", max_questions, calibration_data=calibration_data,
        )

        if llm is not None:
            for candidate in candidates:
                result = await run_swarm(
                    llm, candidate, payload, calibration_data,
                )
                candidate.probability = result.final_probability
                # Annotate with swarm metadata
                candidate.signals_cited = (candidate.signals_cited or []) + [
                    f"swarm:aggregated={result.aggregated_probability:.3f}",
                    f"swarm:extremized={result.extremized_probability:.3f}",
                    f"swarm:personas={len(result.forecasts)}",
                ] + [
                    f"swarm:{f.persona}={f.probability:.3f}"
                    for f in result.forecasts
                ]

        return ForecastRun(
            topic_slug=payload.topic_slug,
            topic_name=payload.topic_name,
            engine=self.engine_name,
            generated_for=payload.run_date,
            summary=f"Swarm ensemble forecast ({len(PERSONAS)} personas) for {payload.topic_name}.",
            questions=candidates,
            metadata={"swarm": True, "persona_count": len(PERSONAS)},
        )
