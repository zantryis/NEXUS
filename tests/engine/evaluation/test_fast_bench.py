"""Tests for fast benchmark runner."""

import json
from datetime import date
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nexus.config.models import NexusConfig, TopicConfig, UserConfig
from nexus.engine.evaluation.experiment import CachedTopicData
from nexus.engine.knowledge.events import Event
from nexus.engine.sources.polling import ContentItem
from nexus.engine.synthesis.knowledge import NarrativeThread, TopicSynthesis


def _make_fake_synth(topic_name: str = "Test Topic") -> TopicSynthesis:
    """Build a minimal valid TopicSynthesis for mocking."""
    event = Event(date=date(2026, 3, 15), summary="Test event", significance=7)
    return TopicSynthesis(
        topic_name=topic_name,
        threads=[NarrativeThread(
            headline="Test thread",
            events=[event],
            convergence=["Sources agree"],
            divergence=[],
            key_entities=["Entity A"],
            significance=7,
        )],
        background=[],
    )


def _make_cached(slug: str, n_articles: int = 5) -> CachedTopicData:
    """Build a CachedTopicData with fake articles."""
    topic_cfg = TopicConfig(name=slug.replace("-", " ").title(), subtopics=["test"])
    articles = [
        ContentItem(
            title=f"Article {i}",
            url=f"https://example.com/{slug}/{i}",
            source_id=f"src-{i}",
            text=f"Body text for article {i} about {slug}.",
        )
        for i in range(n_articles)
    ]
    return CachedTopicData(
        topic_cfg=topic_cfg,
        slug=slug,
        raw_articles=articles,
        ingested_articles=articles,
        recent_events=[],
    )


def _write_fixtures(tmp_path: Path, slugs: list[str]) -> Path:
    """Write CachedTopicData fixtures to a temp dir and return it."""
    fixture_dir = tmp_path / "2026-03-15"
    fixture_dir.mkdir(parents=True)
    for slug in slugs:
        cached = _make_cached(slug)
        cached.to_json(fixture_dir / f"{slug}.json")
    return fixture_dir


class TestLoadFixtures:
    def test_load_fixtures_from_dir(self, tmp_path):
        """Loading fixtures from a specific directory returns CachedTopicData list."""
        from nexus.engine.evaluation.fast_bench import load_fixtures

        fixture_dir = _write_fixtures(tmp_path, ["iran-us-relations", "ai-ml-research"])
        result = load_fixtures(fixture_dir)

        assert len(result) == 2
        slugs = {c.slug for c in result}
        assert "iran-us-relations" in slugs
        assert "ai-ml-research" in slugs
        assert all(len(c.ingested_articles) == 5 for c in result)

    def test_load_fixtures_missing_dir_raises(self, tmp_path):
        """Missing fixture directory raises FileNotFoundError."""
        from nexus.engine.evaluation.fast_bench import load_fixtures

        with pytest.raises(FileNotFoundError):
            load_fixtures(tmp_path / "nonexistent")

    def test_load_fixtures_empty_dir_raises(self, tmp_path):
        """Directory with no JSON files raises ValueError."""
        from nexus.engine.evaluation.fast_bench import load_fixtures

        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with pytest.raises(ValueError, match="No fixture"):
            load_fixtures(empty_dir)


class TestLatestFixtureDir:
    def test_finds_most_recent_date_dir(self, tmp_path):
        """Auto-detects the most recent dated subdirectory."""
        from nexus.engine.evaluation.fast_bench import find_latest_fixture_dir

        benchmarks = tmp_path / "benchmarks"
        for d in ["2026-03-10", "2026-03-15", "2026-03-12"]:
            subdir = benchmarks / d
            subdir.mkdir(parents=True)
            cached = _make_cached("test")
            cached.to_json(subdir / "test.json")

        result = find_latest_fixture_dir(benchmarks)
        assert result.name == "2026-03-15"

    def test_no_dated_dirs_raises(self, tmp_path):
        """No valid dated directories raises FileNotFoundError."""
        from nexus.engine.evaluation.fast_bench import find_latest_fixture_dir

        benchmarks = tmp_path / "benchmarks"
        benchmarks.mkdir()
        (benchmarks / "results").mkdir()  # not a date dir

        with pytest.raises(FileNotFoundError, match="No benchmark"):
            find_latest_fixture_dir(benchmarks)


class TestThresholdOverride:
    def test_threshold_applied_to_topic_configs(self, tmp_path):
        """Threshold override modifies topic configs in loaded fixtures."""
        from nexus.engine.evaluation.fast_bench import load_fixtures, apply_threshold_override

        fixture_dir = _write_fixtures(tmp_path, ["test-topic"])
        fixtures = load_fixtures(fixture_dir)

        # Default threshold should be whatever TopicConfig defaults to
        original = fixtures[0].topic_cfg.filter_threshold

        apply_threshold_override(fixtures, 3.0)
        assert fixtures[0].topic_cfg.filter_threshold == 3.0
        assert fixtures[0].topic_cfg.filter_threshold != original or original == 3.0


class TestRunFastBenchmark:
    @pytest.mark.asyncio
    async def test_runs_suite_a_on_fixtures(self, tmp_path):
        """Fast benchmark runs 3 variants (full, naive, no_filter) per topic."""
        from nexus.engine.evaluation.fast_bench import run_fast_benchmark

        fixture_dir = _write_fixtures(tmp_path, ["test-topic"])

        mock_llm = AsyncMock()
        mock_store = AsyncMock()

        fake_synth = _make_fake_synth()
        fake_stats = {"articles_in": 5, "pass_out": 3, "pass_rate": 0.6, "events": 2}

        with patch("nexus.engine.evaluation.fast_bench.run_full_pipeline_variant",
                    return_value=(fake_synth, fake_stats)) as mock_full, \
             patch("nexus.engine.evaluation.fast_bench.run_no_filter_variant",
                    return_value=(fake_synth, fake_stats)) as mock_nf, \
             patch("nexus.engine.evaluation.fast_bench.build_naive_synthesis",
                    return_value=fake_synth) as mock_naive, \
             patch("nexus.engine.evaluation.fast_bench.judge_synthesis",
                    return_value={"completeness": 7, "source_balance": 6,
                                  "convergence_accuracy": 5, "divergence_detection": 3,
                                  "entity_coverage": 7, "overall": 5.6}):

            report = await run_fast_benchmark(
                llm=mock_llm,
                store=mock_store,
                fixture_dir=fixture_dir,
            )

        assert "A" in report.suites
        suite_a = report.suites["A"]
        # 3 variants × 1 topic = 3 results
        assert len(suite_a.results) == 3
        variant_names = {r.variant_name for r in suite_a.results}
        assert variant_names == {"full_pipeline", "naive_baseline", "no_filter"}

    @pytest.mark.asyncio
    async def test_threshold_override_passed_to_pipeline(self, tmp_path):
        """Threshold override is passed through to run_full_pipeline_variant."""
        from nexus.engine.evaluation.fast_bench import run_fast_benchmark

        fixture_dir = _write_fixtures(tmp_path, ["test-topic"])

        mock_llm = AsyncMock()
        mock_store = AsyncMock()

        fake_synth = _make_fake_synth()

        with patch("nexus.engine.evaluation.fast_bench.run_full_pipeline_variant",
                    return_value=(fake_synth, {"articles_in": 5, "pass_out": 3})) as mock_full, \
             patch("nexus.engine.evaluation.fast_bench.run_no_filter_variant",
                    return_value=(fake_synth, {})), \
             patch("nexus.engine.evaluation.fast_bench.build_naive_synthesis",
                    return_value=fake_synth), \
             patch("nexus.engine.evaluation.fast_bench.judge_synthesis",
                    return_value={"completeness": 7, "overall": 5.6}):

            await run_fast_benchmark(
                llm=mock_llm,
                store=mock_store,
                fixture_dir=fixture_dir,
                threshold_override=3.5,
            )

        # Verify threshold was passed to the full pipeline variant
        call_kwargs = mock_full.call_args
        assert call_kwargs[1].get("threshold") == 3.5 or call_kwargs.kwargs.get("threshold") == 3.5


class TestCaptureFixtures:
    @pytest.mark.asyncio
    async def test_capture_saves_fixture_files(self, tmp_path):
        """Capture polls topics and saves CachedTopicData JSONs."""
        from nexus.engine.evaluation.fast_bench import capture_benchmark

        config = NexusConfig(
            user=UserConfig(name="Test"),
            topics=[
                TopicConfig(name="Iran US Relations", subtopics=["sanctions"]),
                TopicConfig(name="AI ML Research", subtopics=["llm"]),
            ],
        )

        fake_cached = _make_cached("iran-us-relations", n_articles=10)

        with patch("nexus.engine.evaluation.fast_bench.cache_topic_articles",
                    return_value=fake_cached) as mock_cache:

            output_dir = await capture_benchmark(
                config=config,
                llm=AsyncMock(),
                store=AsyncMock(),
                data_dir=tmp_path,
            )

        # Should have called cache_topic_articles for each topic
        assert mock_cache.call_count == 2

        # Output dir should be date-stamped under data_dir/benchmarks/
        assert output_dir.parent.name == "benchmarks" or output_dir.parent.parent == tmp_path
        assert output_dir.exists()

        # Should have saved JSON files
        json_files = list(output_dir.glob("*.json"))
        assert len(json_files) == 2


class TestResultSaving:
    @pytest.mark.asyncio
    async def test_saves_results_json(self, tmp_path):
        """Fast benchmark saves results to data/benchmarks/results/."""
        from nexus.engine.evaluation.fast_bench import run_fast_benchmark

        fixture_dir = _write_fixtures(tmp_path, ["test-topic"])
        results_dir = tmp_path / "benchmarks" / "results"

        fake_synth = _make_fake_synth()

        with patch("nexus.engine.evaluation.fast_bench.run_full_pipeline_variant",
                    return_value=(fake_synth, {"articles_in": 5})), \
             patch("nexus.engine.evaluation.fast_bench.run_no_filter_variant",
                    return_value=(fake_synth, {})), \
             patch("nexus.engine.evaluation.fast_bench.build_naive_synthesis",
                    return_value=fake_synth), \
             patch("nexus.engine.evaluation.fast_bench.judge_synthesis",
                    return_value={"completeness": 7, "overall": 5.6}):

            report = await run_fast_benchmark(
                llm=AsyncMock(),
                store=AsyncMock(),
                fixture_dir=fixture_dir,
                results_dir=results_dir,
            )

        # Results should be saved
        result_files = list(results_dir.glob("*.json"))
        assert len(result_files) == 1

        # Should be valid JSON with suite A
        data = json.loads(result_files[0].read_text())
        assert "suites" in data
        assert "A" in data["suites"]
