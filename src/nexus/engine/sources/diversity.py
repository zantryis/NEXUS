"""Source diversity scoring — evaluate geographic, affiliation, and language balance."""

import math
import logging
from collections import Counter
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class DiversityMetrics:
    """Diversity assessment for a set of source feeds."""
    geographic_score: float = 0.0   # Shannon entropy across countries (0–3)
    affiliation_score: float = 0.0  # Shannon entropy across affiliations (0–2.5)
    language_score: float = 0.0     # Shannon entropy across languages (0–3)
    overall: float = 0.0            # Weighted average
    warnings: list[str] = field(default_factory=list)
    country_dist: dict[str, int] = field(default_factory=dict)
    affiliation_dist: dict[str, int] = field(default_factory=dict)
    language_dist: dict[str, int] = field(default_factory=dict)


def _shannon_entropy(counts: dict[str, int]) -> float:
    """Shannon entropy in bits. Higher = more diverse."""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)
    return round(entropy, 3)


def compute_diversity(feeds: list[dict]) -> DiversityMetrics:
    """Compute diversity metrics for a set of feeds."""
    if not feeds:
        return DiversityMetrics(warnings=["No feeds to assess"])

    countries = Counter(f.get("country", "unknown") for f in feeds)
    affiliations = Counter(f.get("affiliation", "unknown") for f in feeds)
    languages = Counter(f.get("language", "unknown") for f in feeds)

    geo = _shannon_entropy(countries)
    aff = _shannon_entropy(affiliations)
    lang = _shannon_entropy(languages)
    overall = round(geo * 0.4 + aff * 0.4 + lang * 0.2, 3)

    warnings = []

    # Check for obvious imbalances
    total = len(feeds)
    for country, count in countries.items():
        if count / total > 0.6 and total >= 5:
            warnings.append(f"Geographic concentration: {count}/{total} feeds from {country}")

    unknown_aff = affiliations.get("unknown", 0)
    if unknown_aff / total > 0.5 and total >= 5:
        warnings.append(f"Many unclassified sources: {unknown_aff}/{total} have unknown affiliation")

    aff_types = {k for k in affiliations if k != "unknown"}
    if len(aff_types) == 1 and total >= 5:
        warnings.append(f"Single perspective: all sources are '{list(aff_types)[0]}'")

    if len(languages) == 1 and total >= 10:
        warnings.append(f"Single language: all sources in '{list(languages.keys())[0]}'")

    return DiversityMetrics(
        geographic_score=geo,
        affiliation_score=aff,
        language_score=lang,
        overall=overall,
        warnings=warnings,
        country_dist=dict(countries.most_common()),
        affiliation_dist=dict(affiliations.most_common()),
        language_dist=dict(languages.most_common()),
    )


def suggest_improvements(
    current_feeds: list[dict],
    global_sources: list[dict],
    max_suggestions: int = 5,
) -> list[dict]:
    """Suggest feeds from the global registry that would improve diversity.

    Prioritizes underrepresented countries, affiliations, and languages.
    """
    current_urls = {f.get("url") for f in current_feeds}
    current_countries = Counter(f.get("country", "unknown") for f in current_feeds)
    current_affiliations = Counter(f.get("affiliation", "unknown") for f in current_feeds)

    # Score each global source by how much it fills a gap
    candidates = []
    for source in global_sources:
        if source.get("url") in current_urls:
            continue

        country = source.get("country", "unknown")
        affiliation = source.get("affiliation", "unknown")

        # Lower counts = higher priority (fills a gap)
        country_score = 1.0 / (1 + current_countries.get(country, 0))
        affil_score = 1.0 / (1 + current_affiliations.get(affiliation, 0))
        tier_bonus = {"A": 1.0, "B": 0.7, "C": 0.4}.get(source.get("tier", "C"), 0.4)

        gap_score = (country_score * 0.4 + affil_score * 0.4 + tier_bonus * 0.2)
        candidates.append((gap_score, source))

    candidates.sort(key=lambda x: x[0], reverse=True)
    return [source for _, source in candidates[:max_suggestions]]
