"""Tests for the benchmark module — report structure, naive baseline, style comparison."""

import pytest
from datetime import date

from nexus.engine.evaluation.benchmark import (
    BenchmarkReport,
    TopicBenchmark,
    StyleResult,
    build_naive_synthesis,
    _compute_aggregate,
    _compute_style_ranking,
)
from nexus.engine.sources.polling import ContentItem
from nexus.engine.synthesis.knowledge import TopicSynthesis


def _make_articles(n: int = 10) -> list[ContentItem]:
    """Create minimal ContentItem objects for testing."""
    items = []
    for i in range(n):
        items.append(ContentItem(
            url=f"https://example.com/{i}",
            title=f"Article {i}: Test headline about topic",
            source_id=f"source-{i % 3}",
            source_affiliation=["private", "state", "public"][i % 3],
            source_country=["US", "GB", "RU"][i % 3],
            source_language="en",
        ))
    return items


def test_benchmark_report_structure():
    """BenchmarkReport has correct structure and to_json/to_markdown work."""
    tb = TopicBenchmark(
        topic_name="AI Research",
        topic_slug="ai-research",
        nexus_scores={"overall": 7.5, "completeness": 8, "source_balance": 7,
                      "convergence_accuracy": 8, "divergence_detection": 6,
                      "entity_coverage": 8, "strengths": ["good coverage"],
                      "weaknesses": ["weak divergence"]},
        naive_scores={"overall": 4.2, "completeness": 5, "source_balance": 4,
                      "convergence_accuracy": 3, "divergence_detection": 3,
                      "entity_coverage": 6},
        style_results=[
            StyleResult(style="analytical", briefing_text="test", scores={"overall": 7.0}),
            StyleResult(style="editorial", briefing_text="test", scores={"overall": 8.2}),
        ],
        improvement_pct=78.6,
        article_count=25,
        event_count=8,
        source_count=12,
    )
    report = BenchmarkReport(
        topics=[tb],
        aggregate={"nexus": {"overall": 7.5}, "naive": {"overall": 4.2}},
        style_ranking=[("editorial", 8.2), ("analytical", 7.0)],
        timestamp="2026-03-13T10:00:00",
        judge_model="test-model",
        duration_s=120.0,
    )

    # JSON round-trip
    j = report.to_json()
    assert j["topics"][0]["topic_name"] == "AI Research"
    assert j["style_ranking"][0] == ("editorial", 8.2)

    # Markdown output
    md = report.to_markdown()
    assert "# Nexus Benchmark Report" in md
    assert "AI Research" in md
    assert "editorial" in md
    assert "+79%" in md or "79" in md


def test_naive_baseline_generation():
    """Naive synthesis has flat structure: one thread, no convergence/divergence."""
    articles = _make_articles(20)
    syn = build_naive_synthesis("Test Topic", articles, max_articles=10)

    assert syn.topic_name == "Test Topic"
    assert len(syn.threads) == 1
    assert syn.threads[0].convergence == []
    assert syn.threads[0].divergence == []
    assert len(syn.threads[0].events) == 10  # max_articles cap
    assert syn.metadata.get("naive") is True


def test_naive_baseline_respects_max():
    """Naive baseline doesn't exceed max_articles."""
    articles = _make_articles(5)
    syn = build_naive_synthesis("Test", articles, max_articles=30)
    assert len(syn.threads[0].events) == 5  # only 5 available


def test_naive_baseline_source_balance():
    """Naive baseline correctly computes source balance from sampled articles."""
    articles = _make_articles(9)
    syn = build_naive_synthesis("Test", articles, max_articles=9)
    assert "private" in syn.source_balance
    assert "state" in syn.source_balance
    assert "public" in syn.source_balance


def test_funnel_stats_from_filter_log():
    """Placeholder — funnel stats require a live store. Just verify import works."""
    from nexus.engine.evaluation.benchmark import get_funnel_stats
    assert callable(get_funnel_stats)


def test_aggregate_computation():
    """Aggregate averages across topics correctly."""
    topics = [
        TopicBenchmark(
            topic_name="A", topic_slug="a",
            nexus_scores={"overall": 8.0, "completeness": 8, "source_balance": 7,
                          "convergence_accuracy": 9, "divergence_detection": 7, "entity_coverage": 9},
            naive_scores={"overall": 4.0, "completeness": 5, "source_balance": 3,
                          "convergence_accuracy": 4, "divergence_detection": 3, "entity_coverage": 5},
        ),
        TopicBenchmark(
            topic_name="B", topic_slug="b",
            nexus_scores={"overall": 6.0, "completeness": 6, "source_balance": 6,
                          "convergence_accuracy": 6, "divergence_detection": 6, "entity_coverage": 6},
            naive_scores={"overall": 5.0, "completeness": 5, "source_balance": 5,
                          "convergence_accuracy": 5, "divergence_detection": 5, "entity_coverage": 5},
        ),
    ]
    agg = _compute_aggregate(topics)
    assert agg["nexus"]["overall"] == 7.0  # (8+6)/2
    assert agg["naive"]["overall"] == 4.5  # (4+5)/2


def test_style_ranking():
    """Style ranking sorts by average overall score."""
    topics = [
        TopicBenchmark(
            topic_name="A", topic_slug="a",
            style_results=[
                StyleResult(style="analytical", briefing_text="", scores={"overall": 6.0}),
                StyleResult(style="editorial", briefing_text="", scores={"overall": 8.0}),
                StyleResult(style="conversational", briefing_text="", scores={"overall": 7.0}),
            ],
        ),
    ]
    ranking = _compute_style_ranking(topics)
    assert ranking[0][0] == "editorial"
    assert ranking[0][1] == 8.0
    assert ranking[-1][0] == "analytical"


def test_style_ranking_skips_errors():
    """Style ranking ignores results with errors."""
    topics = [
        TopicBenchmark(
            topic_name="A", topic_slug="a",
            style_results=[
                StyleResult(style="analytical", briefing_text="", scores={"overall": 7.0}),
                StyleResult(style="editorial", briefing_text="", scores={"error": "failed"}),
            ],
        ),
    ]
    ranking = _compute_style_ranking(topics)
    assert len(ranking) == 1
    assert ranking[0][0] == "analytical"
