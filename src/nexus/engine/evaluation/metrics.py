"""Automated metrics — computed every pipeline run, no LLM needed."""

import math
from collections import Counter
from datetime import date
from pathlib import Path

import yaml

from nexus.engine.knowledge.events import Event
from nexus.engine.sources.polling import ContentItem
from nexus.engine.synthesis.knowledge import TopicSynthesis


def source_diversity_index(articles: list[ContentItem]) -> float:
    """Shannon entropy across affiliations. Higher = more diverse."""
    if not articles:
        return 0.0
    affiliations = [a.source_affiliation or "unknown" for a in articles]
    counts = Counter(affiliations)
    total = sum(counts.values())
    entropy = 0.0
    for count in counts.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)
    return round(entropy, 3)


def convergence_ratio(events: list[Event]) -> float:
    """Fraction of events confirmed by 2+ sources."""
    if not events:
        return 0.0
    multi_source = sum(1 for e in events if len(e.sources) >= 2)
    return round(multi_source / len(events), 3)


def language_coverage(articles: list[ContentItem]) -> dict:
    """Percentage of articles from each detected language."""
    if not articles:
        return {}
    langs = [a.detected_language or a.source_language or "unknown" for a in articles]
    counts = Counter(langs)
    total = sum(counts.values())
    return {lang: round(count / total, 3) for lang, count in counts.most_common()}


def extraction_stats(articles: list[ContentItem]) -> dict:
    """Per-status breakdown of extraction results."""
    statuses = Counter(a.extraction_status for a in articles)
    return dict(statuses.most_common())


def event_dedup_ratio(extracted_count: int, final_count: int) -> float:
    """Ratio of final events to extracted events (1.0 = no dedup, lower = more merging)."""
    if extracted_count == 0:
        return 1.0
    return round(final_count / extracted_count, 3)


def compute_run_metrics(
    syntheses: list[TopicSynthesis],
    all_articles: list[ContentItem],
    all_events: list[Event],
    extracted_event_count: int,
) -> dict:
    """Compute all automated metrics for a pipeline run."""
    return {
        "date": date.today().isoformat(),
        "source_diversity_index": source_diversity_index(all_articles),
        "convergence_ratio": convergence_ratio(all_events),
        "language_coverage": language_coverage(all_articles),
        "extraction_stats": extraction_stats(all_articles),
        "event_dedup_ratio": event_dedup_ratio(extracted_event_count, len(all_events)),
        "article_count": len(all_articles),
        "event_count": len(all_events),
        "thread_count": sum(len(s.threads) for s in syntheses),
        "divergence_count": sum(
            len(t.divergence) for s in syntheses for t in s.threads
        ),
        "topics": {
            s.topic_name: {
                "threads": len(s.threads),
                "source_balance": s.source_balance,
                "languages": s.languages_represented,
            }
            for s in syntheses
        },
    }


def save_metrics(data_dir: Path, metrics: dict) -> Path:
    """Save metrics to data/metrics/{date}.yaml."""
    metrics_dir = data_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    path = metrics_dir / f"{metrics['date']}.yaml"
    path.write_text(yaml.dump(metrics, default_flow_style=False, allow_unicode=True))
    return path
