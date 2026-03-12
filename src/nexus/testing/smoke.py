"""E2E smoke test runner — minimal pipeline run with real APIs, verifies outputs."""

import logging
import time
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

from nexus.config.models import NexusConfig, TopicConfig, UserConfig, BudgetConfig
from nexus.engine.pipeline import run_pipeline
from nexus.llm.client import LLMClient

logger = logging.getLogger(__name__)

# Override ingestion cap for smoke tests
SMOKE_MAX_INGEST = 20


@dataclass
class SmokeTestConfig:
    """Configuration for a smoke test run."""
    topic_name: str = "AI/ML Research"
    max_articles: int = 20
    max_feeds: int = 8
    test_telegram: bool = False
    test_audio: bool = False
    data_dir: Optional[Path] = None


@dataclass
class SmokeCheck:
    """A single verification check."""
    name: str
    passed: bool
    detail: str = ""


@dataclass
class SmokeTestResult:
    """Result of a smoke test run."""
    success: bool = False
    checks: list[SmokeCheck] = field(default_factory=list)
    events_found: int = 0
    threads_created: int = 0
    briefing_chars: int = 0
    cost_usd: float = 0.0
    timing_s: float = 0.0
    errors: list[str] = field(default_factory=list)


def _build_smoke_config(
    smoke_cfg: SmokeTestConfig,
    data_dir: Path,
) -> NexusConfig:
    """Build a minimal NexusConfig for smoke testing."""
    topic = TopicConfig(
        name=smoke_cfg.topic_name,
        priority="high",
        subtopics=["latest developments"],
        scope="narrow",
        max_events=5,
    )
    return NexusConfig(
        user=UserConfig(name="SmokeTest", timezone="UTC"),
        topics=[topic],
        budget=BudgetConfig(daily_limit_usd=0.50, warning_threshold_usd=0.25),
    )


async def run_smoke_test(config: SmokeTestConfig) -> SmokeTestResult:
    """Run the smoke test pipeline and verify outputs.

    This hits real APIs — requires API keys in environment.
    """
    import os
    from nexus.engine.knowledge.store import KnowledgeStore

    result = SmokeTestResult()
    t0 = time.monotonic()

    # Use provided data_dir or create temp
    data_dir = config.data_dir
    if data_dir is None:
        import tempfile
        data_dir = Path(tempfile.mkdtemp(prefix="nexus-smoke-"))

    data_dir.mkdir(parents=True, exist_ok=True)

    # Check 1: API keys present
    api_key = os.getenv("GEMINI_API_KEY")
    deepseek_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("deepseek")
    has_key = bool(api_key or deepseek_key)
    result.checks.append(SmokeCheck(
        "api_keys_present", has_key,
        f"GEMINI={'set' if api_key else 'missing'}, DEEPSEEK={'set' if deepseek_key else 'missing'}",
    ))
    if not has_key:
        result.errors.append("No API keys available — cannot run smoke test")
        result.timing_s = time.monotonic() - t0
        return result

    # Build config
    nexus_cfg = _build_smoke_config(config, data_dir)

    llm = LLMClient(
        nexus_cfg.models,
        api_key=api_key,
        deepseek_api_key=deepseek_key,
        budget_config=nexus_cfg.budget,
    )

    # Check 2: Source registry exists (or discover)
    from nexus.engine.pipeline import load_source_registry
    sources = load_source_registry(data_dir, nexus_cfg.topics[0])
    if not sources:
        # Try to copy from main data dir or discover
        main_data = Path("data")
        slug = nexus_cfg.topics[0].name.lower().replace(" ", "-").replace("/", "-")
        main_registry = main_data / "sources" / slug / "registry.yaml"
        smoke_registry = data_dir / "sources" / slug / "registry.yaml"
        smoke_registry.parent.mkdir(parents=True, exist_ok=True)

        if main_registry.exists():
            import yaml
            reg = yaml.safe_load(main_registry.read_text())
            # Limit feeds for speed
            if reg and "sources" in reg:
                reg["sources"] = reg["sources"][:config.max_feeds]
            smoke_registry.write_text(yaml.dump(reg, default_flow_style=False))
            sources = reg.get("sources", [])
        else:
            # Run discovery
            from nexus.engine.sources.discovery import discover_sources
            disc_result = await discover_sources(
                llm, nexus_cfg.topics[0].name,
                subtopics=nexus_cfg.topics[0].subtopics,
                max_feeds=config.max_feeds,
                data_dir=data_dir,
            )
            if disc_result.feeds:
                import yaml
                smoke_registry.write_text(
                    yaml.dump({"sources": disc_result.feeds}, default_flow_style=False)
                )
                sources = disc_result.feeds

    result.checks.append(SmokeCheck(
        "sources_available", len(sources) > 0,
        f"{len(sources)} feeds available",
    ))

    if not sources:
        result.errors.append("No source feeds available")
        result.timing_s = time.monotonic() - t0
        return result

    try:
        # Run pipeline with capped ingestion for speed
        briefing_path = await run_pipeline(
            nexus_cfg, llm, data_dir,
            gemini_api_key=api_key,
            max_ingest=SMOKE_MAX_INGEST,
        )

        # Check 3: Briefing generated
        briefing_exists = briefing_path.exists()
        briefing_text = briefing_path.read_text() if briefing_exists else ""
        result.briefing_chars = len(briefing_text)
        result.checks.append(SmokeCheck(
            "briefing_generated", briefing_exists and len(briefing_text) > 100,
            f"{len(briefing_text)} chars",
        ))

        # Check 4: Events in store
        store = KnowledgeStore(data_dir / "knowledge.db")
        await store.initialize()
        try:
            slug = nexus_cfg.topics[0].name.lower().replace(" ", "-").replace("/", "-")
            events = await store.get_events(slug)
            result.events_found = len(events)
            result.checks.append(SmokeCheck(
                "events_extracted", len(events) > 0,
                f"{len(events)} events",
            ))

            # Check 5: Events are recent
            cutoff = date.today() - timedelta(days=3)
            recent = [e for e in events if e.date >= cutoff]
            pct = (len(recent) / len(events) * 100) if events else 0
            result.checks.append(SmokeCheck(
                "events_are_recent", pct >= 50,
                f"{len(recent)}/{len(events)} within 3 days ({pct:.0f}%)",
            ))

            # Check 6: Cost tracking
            cost = llm.usage.cost_summary()
            result.cost_usd = cost.get("total_usd", 0)
            result.checks.append(SmokeCheck(
                "cost_tracked", cost.get("total_usd", 0) > 0,
                f"${cost.get('total_usd', 0):.4f}",
            ))

            # Check 7: Cost persisted to SQLite
            daily_cost = await store.get_daily_cost(date.today().isoformat())
            result.checks.append(SmokeCheck(
                "cost_persisted", daily_cost > 0,
                f"${daily_cost:.4f} in store",
            ))

            # Check 8: Metrics file
            metrics_path = data_dir / "metrics" / f"{date.today().isoformat()}.yaml"
            result.checks.append(SmokeCheck(
                "metrics_saved", metrics_path.exists(),
                str(metrics_path) if metrics_path.exists() else "missing",
            ))
        finally:
            await store.close()

    except Exception as e:
        result.errors.append(f"Pipeline failed: {e}")
        logger.exception("Smoke test pipeline failed")

    result.timing_s = round(time.monotonic() - t0, 1)
    result.success = all(c.passed for c in result.checks) and not result.errors
    return result
