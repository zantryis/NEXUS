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

import math


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
    gamma<1 compresses (pushes toward 0.5) — use this for
    LLM overconfidence correction.
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
