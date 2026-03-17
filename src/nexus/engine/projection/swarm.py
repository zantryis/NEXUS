"""Forecast aggregation math: geometric mean of odds, extremization, anchor blending.

Provably optimal aggregation methods for combining probability estimates.
These utilities are used by forecast engines to calibrate LLM outputs.

References:
- Neyman & Roughgarden, "When pooling forecasts, use geometric mean of odds"
- Satopaa et al., "Combining multiple probability predictions using a
  simple logit model" (IJF, 2014) — extremization theory
- Schoenegger et al., "Wisdom of the Silicon Crowd" (Science Advances, 2024)
"""

from __future__ import annotations

import logging
import math

logger = logging.getLogger(__name__)


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


def extremize(probability: float, gamma: float = 0.8) -> float:
    """Push probability away from or toward 0.5.

    Uses the standard extremization formula:
        p_ext = p^gamma / (p^gamma + (1-p)^gamma)

    gamma=1.0 is the identity. gamma>1 extremizes (pushes away from 0.5).
    gamma<1 compresses (pushes toward 0.5) — the production default (0.8)
    corrects for LLM overconfidence.
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


def derive_verdict(probability: float) -> tuple[str, str]:
    """Derive verdict and confidence from a calibrated probability.

    Returns (verdict, confidence) where:
    - verdict: "yes" | "no" | "uncertain"
    - confidence: "high" | "medium" | "low"
    """
    if probability >= 0.80:
        return "yes", "high"
    elif probability >= 0.65:
        return "yes", "medium"
    elif probability >= 0.55:
        return "yes", "low"
    elif probability <= 0.20:
        return "no", "high"
    elif probability <= 0.35:
        return "no", "medium"
    elif probability <= 0.45:
        return "no", "low"
    else:
        return "uncertain", "low"


def anchor_blend(
    swarm_probability: float,
    anchor_probability: float,
    *,
    swarm_weight: float = 0.4,
) -> float:
    """Blend an LLM estimate with a deterministic anchor.

    Uses the anchor (well-calibrated) probability as the base
    and lets the LLM shift it. The swarm_weight controls how much
    influence the LLM reasoning has (0.4 = LLM moves the probability
    by 40% of the gap between anchor and LLM estimate).
    """
    blended = anchor_probability + swarm_weight * (swarm_probability - anchor_probability)
    return max(0.02, min(0.98, round(blended, 4)))


# ── Auto-calibration: fit gamma + swarm_weight from resolved forecasts ──


MIN_RESOLVED_FOR_TUNING = 20  # Don't tune with fewer samples


def _brier(probability: float, outcome: bool) -> float:
    """Brier score for a single forecast."""
    target = 1.0 if outcome else 0.0
    return (probability - target) ** 2


def fit_calibration_params(
    resolved: list[dict],
    *,
    gamma_range: tuple[float, float, float] = (0.3, 1.5, 0.1),
    weight_range: tuple[float, float, float] = (0.1, 0.8, 0.05),
) -> dict:
    """Grid search over gamma and swarm_weight to minimize mean Brier score.

    Each entry in *resolved* must have:
      - "raw_probability": the LLM's raw output (before calibration)
      - "market_probability": the anchor (Kalshi price or base rate), or None
      - "resolved_bool": True/False outcome

    Returns {"gamma", "swarm_weight", "mean_brier", "n_samples", "improved"}.
    If too few samples, returns current defaults with improved=False.
    """
    usable = [
        r for r in resolved
        if r.get("raw_probability") is not None and r.get("resolved_bool") is not None
    ]
    if len(usable) < MIN_RESOLVED_FOR_TUNING:
        return {
            "gamma": 0.8,
            "swarm_weight": 0.45,
            "mean_brier": None,
            "n_samples": len(usable),
            "improved": False,
            "reason": f"need {MIN_RESOLVED_FOR_TUNING} resolved, have {len(usable)}",
        }

    # Compute baseline Brier with current defaults
    baseline_brier = _mean_brier_for_params(usable, gamma=0.8, swarm_weight=0.45)

    # Grid search
    best_gamma = 0.8
    best_weight = 0.45
    best_brier = baseline_brier

    g_start, g_end, g_step = gamma_range
    w_start, w_end, w_step = weight_range

    g = g_start
    while g <= g_end:
        w = w_start
        while w <= w_end:
            score = _mean_brier_for_params(usable, gamma=g, swarm_weight=w)
            if score < best_brier:
                best_brier = score
                best_gamma = round(g, 2)
                best_weight = round(w, 2)
            w += w_step
        g += g_step

    improved = best_brier < baseline_brier - 0.001  # Meaningful improvement threshold
    if improved:
        logger.info(
            "Calibration improved: gamma=%.2f→%.2f, weight=%.2f→%.2f, "
            "Brier=%.4f→%.4f (n=%d)",
            0.8, best_gamma, 0.45, best_weight,
            baseline_brier, best_brier, len(usable),
        )

    return {
        "gamma": best_gamma,
        "swarm_weight": best_weight,
        "mean_brier": round(best_brier, 4),
        "baseline_brier": round(baseline_brier, 4),
        "n_samples": len(usable),
        "improved": improved,
    }


def _mean_brier_for_params(
    samples: list[dict], *, gamma: float, swarm_weight: float,
) -> float:
    """Compute mean Brier score for a set of calibration parameters."""
    total = 0.0
    for s in samples:
        raw = float(s["raw_probability"])
        outcome = s["resolved_bool"]
        calibrated = extremize(raw, gamma=gamma)
        market_prob = s.get("market_probability")
        if market_prob is not None:
            calibrated = anchor_blend(calibrated, float(market_prob), swarm_weight=swarm_weight)
        calibrated = max(0.05, min(0.95, calibrated))
        total += _brier(calibrated, outcome)
    return total / len(samples)
