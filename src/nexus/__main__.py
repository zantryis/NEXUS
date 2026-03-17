"""CLI entry point: python -m nexus <command>."""

import asyncio
import json
import logging
import os
import sys
from datetime import date
from pathlib import Path

from dotenv import load_dotenv


def _is_loopback_host(host: str) -> bool:
    """Return True when host clearly binds only to localhost."""
    return host in {"127.0.0.1", "::1", "localhost"}


def _log_bind_warning(host: str, port: int) -> None:
    """Warn when the dashboard is intentionally exposed beyond localhost."""
    if _is_loopback_host(host):
        return
    logging.getLogger(__name__).warning(
        "Binding Nexus to http://%s:%s exposes the dashboard on your network. "
        "Setup/settings stay localhost-only unless NEXUS_ADMIN_TOKEN is set.",
        host,
        port,
    )


def _parse_models_override(args: list[str]) -> dict[str, str]:
    """Parse --models-override key=value pairs from CLI args."""
    overrides = {}
    i = 0
    while i < len(args):
        if args[i] == "--models-override" and i + 1 < len(args):
            key, _, value = args[i + 1].partition("=")
            if key and value:
                overrides[key] = value
            i += 2
        else:
            i += 1
    return overrides


def run_engine():
    """Run the daily engine pipeline."""
    load_dotenv()
    from nexus.config.loader import load_config
    from nexus.engine.pipeline import run_pipeline
    from nexus.llm.client import LLMClient

    data_dir = Path("data")
    config_path = data_dir / "config.yaml"

    if not config_path.exists():
        print(f"Config not found at {config_path}.")
        print(f"  Run: python -m nexus setup")
        print(f"  Or:  cp data/config.example.yaml data/config.yaml")
        sys.exit(1)

    config = load_config(config_path)

    api_key = os.getenv("GEMINI_API_KEY")
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("deepseek")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    litellm_base_url = os.getenv("LITELLM_BASE_URL") or os.getenv("LITELLM_PROXY_URL")
    litellm_api_key = os.getenv("LITELLM_API_KEY") or os.getenv("LITELLM_PROXY_API_KEY")

    # Free/Ollama preset doesn't need API keys.
    # Hosted LiteLLM proxy credentials also count as a valid provider path.
    if (not api_key and not anthropic_api_key and not deepseek_api_key
            and not openai_api_key and not (litellm_base_url and litellm_api_key)
            and config.preset != "free"):
        print("No API key found. Set one in .env:")
        print("  GEMINI_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY, or DEEPSEEK_API_KEY")
        print("  Or LiteLLM proxy creds: LITELLM_BASE_URL/LITELLM_API_KEY")
        print("  Or hosted proxy creds: LITELLM_PROXY_URL/LITELLM_PROXY_API_KEY")
        print("  Or use preset: free for local Ollama")
        sys.exit(1)

    # Apply model overrides from CLI
    overrides = _parse_models_override(sys.argv)
    if overrides:
        for key, value in overrides.items():
            if hasattr(config.models, key):
                setattr(config.models, key, value)
                print(f"  Model override: {key}={value}")
            else:
                print(f"  Warning: unknown model key '{key}'")

    ollama_base_url = os.getenv("OLLAMA_BASE_URL")
    llm = LLMClient(
        config.models, api_key=api_key,
        anthropic_api_key=anthropic_api_key,
        deepseek_api_key=deepseek_api_key,
        openai_api_key=openai_api_key,
        ollama_base_url=ollama_base_url,
        budget_config=config.budget,
        litellm_base_url=litellm_base_url,
        litellm_api_key=litellm_api_key,
    )

    do_capture = "--capture" in sys.argv
    do_backtest = "--backtest" in sys.argv

    if do_backtest:
        from nexus.engine.pipeline import run_backtest
        label = None
        if "--label" in sys.argv:
            idx = sys.argv.index("--label")
            if idx + 1 < len(sys.argv):
                label = sys.argv[idx + 1]

        # --topic: restrict to a single topic
        if "--topic" in sys.argv:
            idx = sys.argv.index("--topic")
            if idx + 1 < len(sys.argv):
                slug = sys.argv[idx + 1]
                matched = [t for t in config.topics
                           if t.name.lower().replace(" ", "-").replace("/", "-") == slug]
                if not matched:
                    print(f"Topic '{slug}' not found. Available:")
                    for t in config.topics:
                        print(f"  {t.name.lower().replace(' ', '-').replace('/', '-')}")
                    sys.exit(1)
                config.topics = matched
                print(f"  Backtest restricted to topic: {matched[0].name}")

        # --days: restrict to last N days of fixture data
        max_days = None
        if "--days" in sys.argv:
            idx = sys.argv.index("--days")
            if idx + 1 < len(sys.argv):
                max_days = int(sys.argv[idx + 1])
                print(f"  Backtest restricted to last {max_days} days")

        asyncio.run(run_backtest(config, llm, data_dir, label=label, max_days=max_days))
    else:
        elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")
        briefing_path = asyncio.run(run_pipeline(
            config, llm, data_dir, capture=do_capture,
            gemini_api_key=api_key,
            openai_api_key=openai_api_key,
            elevenlabs_api_key=elevenlabs_api_key,
        ))
        print(f"Briefing generated: {briefing_path}")


def run_sources():
    """Source registry management: check, build."""
    from nexus.engine.sources.registry import (
        load_global_registry, sources_for_topic, check_feed_health,
        build_topic_registry,
    )
    import yaml

    data_dir = Path("data")
    registry_path = data_dir / "sources" / "global_registry.yaml"
    sources = load_global_registry(registry_path)

    if not sources:
        print(f"No global registry at {registry_path}")
        sys.exit(1)

    subcommand = sys.argv[2] if len(sys.argv) > 2 else "check"

    if subcommand == "check":
        lang_filter = None
        if len(sys.argv) > 3:
            lang_filter = sys.argv[3]
        print(f"Checking {len(sources)} feeds...")
        for s in sources:
            if lang_filter and s.language != lang_filter:
                continue
            result = check_feed_health(s)
            status = result["status"].upper()
            print(f"  {status:5s} {result['entries']:4d}  [{s.language}] {s.id:25s} {s.name}")

    elif subcommand == "build":
        if len(sys.argv) < 4:
            print("Usage: python -m nexus sources build <topic-slug> <tag1,tag2,...>")
            sys.exit(1)
        slug = sys.argv[3]
        tags = sys.argv[4].split(",") if len(sys.argv) > 4 else []

        # Load config to get topic definition
        from nexus.config.loader import load_config
        config = load_config(data_dir / "config.yaml")
        topic = None
        for t in config.topics:
            topic_slug = t.name.lower().replace(" ", "-").replace("/", "-")
            if topic_slug == slug:
                topic = t
                break
        if not topic:
            print(f"Topic '{slug}' not found in config. Available:")
            for t in config.topics:
                print(f"  {t.name.lower().replace(' ', '-').replace('/', '-')}")
            sys.exit(1)

        matched = sources_for_topic(sources, topic, tag_hints=tags)
        if not matched:
            print(f"No sources matched tags={tags} for languages={topic.source_languages}")
            sys.exit(1)

        registry = build_topic_registry(matched)
        out_path = data_dir / "sources" / slug / "registry.yaml"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(yaml.dump(registry, default_flow_style=False))
        print(f"Built {out_path} with {len(matched)} sources:")
        for s in matched:
            print(f"  [{s.language}] {s.id} — {s.name}")

    elif subcommand == "discover":
        load_dotenv()
        import yaml as _yaml
        from nexus.config.loader import load_config
        from nexus.llm.client import LLMClient
        from nexus.engine.sources.discovery import discover_sources

        if len(sys.argv) < 4:
            print("Usage: python -m nexus sources discover <topic-slug>")
            sys.exit(1)

        slug = sys.argv[3]
        config = load_config(data_dir / "config.yaml")

        topic = None
        for t in config.topics:
            topic_slug = t.name.lower().replace(" ", "-").replace("/", "-")
            if topic_slug == slug:
                topic = t
                break
        if not topic:
            print(f"Topic '{slug}' not found in config.")
            sys.exit(1)

        api_key = os.getenv("GEMINI_API_KEY")
        anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        deepseek_api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("deepseek")

        llm = LLMClient(
            config.models, api_key=api_key,
            anthropic_api_key=anthropic_api_key,
            deepseek_api_key=deepseek_api_key,
        )

        # Get existing URLs to avoid duplicates
        existing_urls: set[str] = set()
        existing_path = data_dir / "sources" / slug / "registry.yaml"
        if existing_path.exists():
            existing_reg = _yaml.safe_load(existing_path.read_text()) or {}
            for s_entry in existing_reg.get("sources", []):
                existing_urls.add(s_entry.get("url", ""))

        result = asyncio.run(discover_sources(
            llm, topic.name, subtopics=topic.subtopics,
            existing_urls=existing_urls,
            data_dir=data_dir,
        ))

        if not result.feeds:
            print(f"No new feeds discovered for '{slug}'.")
            sys.exit(0)

        print(f"\nDiscovered {len(result.feeds)} feeds for '{slug}':")
        print(f"  From registry: {result.sources_from_registry}")
        print(f"  From Google News: {result.sources_from_google_news}")
        print(f"  From web search: {result.sources_from_web}")
        print()

        for d in result.feeds:
            affil = d.get('affiliation', '?')[:8]
            country = d.get('country', '?')
            print(f"  [{d['language']}] [{affil:8s}] [{country}] {d.get('name', d['id'])}")
            print(f"      {d['url']}")

        # Diversity metrics
        dm = result.diversity
        print(f"\nDiversity: geo={dm.geographic_score:.2f} affil={dm.affiliation_score:.2f} "
              f"lang={dm.language_score:.2f} overall={dm.overall:.2f}")
        for w in dm.warnings:
            print(f"  ⚠ {w}")

        # Semi-supervised: ask for approval
        print()
        choice = input("[A]ccept all / [R]eview individually / [C]ancel: ").strip().lower()
        if choice == "c":
            print("Cancelled.")
            sys.exit(0)

        approved = result.feeds
        if choice == "r":
            approved = []
            for d in result.feeds:
                name = d.get('name', d['id'])
                resp = input(f"  Keep '{name}'? [y/n]: ").strip().lower()
                if resp != "n":
                    approved.append(d)
            print(f"\nApproved {len(approved)} of {len(result.feeds)} feeds.")

        if not approved:
            print("No feeds approved.")
            sys.exit(0)

        # Append to existing registry
        existing_path.parent.mkdir(parents=True, exist_ok=True)
        if existing_path.exists():
            reg = _yaml.safe_load(existing_path.read_text()) or {}
        else:
            reg = {"sources": []}
        reg["sources"].extend(approved)
        existing_path.write_text(_yaml.dump(reg, default_flow_style=False))
        print(f"Updated {existing_path} with {len(approved)} sources")

    elif subcommand == "list":
        for s in sources:
            affil = f"{s.affiliation}/{s.country}" if s.country else s.affiliation
            print(f"  [{s.language}] {s.tier} {affil:12s} {s.id:25s} {','.join(s.tags)}")

    else:
        print(f"Unknown subcommand: {subcommand}. Use: check, build, list, discover")
        sys.exit(1)


def run_evaluate():
    """Evaluate synthesis quality via LLM judge."""
    load_dotenv()
    import yaml
    from nexus.engine.synthesis.knowledge import TopicSynthesis
    from nexus.engine.evaluation.judge import judge_synthesis, compare_syntheses
    from nexus.config.models import ModelsConfig
    from nexus.llm.client import LLMClient

    api_key = os.getenv("GEMINI_API_KEY")
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("deepseek")
    if not api_key and not anthropic_api_key and not deepseek_api_key:
        print("Set GEMINI_API_KEY, ANTHROPIC_API_KEY, or DEEPSEEK_API_KEY in .env")
        sys.exit(1)

    models = ModelsConfig()
    overrides = _parse_models_override(sys.argv)
    for key, value in overrides.items():
        if hasattr(models, key):
            setattr(models, key, value)

    llm = LLMClient(
        models, api_key=api_key,
        anthropic_api_key=anthropic_api_key,
        deepseek_api_key=deepseek_api_key,
    )

    subcommand = sys.argv[2] if len(sys.argv) > 2 else ""

    if subcommand == "synthesis":
        if len(sys.argv) < 4:
            print("Usage: python -m nexus evaluate synthesis <path-to-synthesis.yaml>")
            sys.exit(1)
        path = Path(sys.argv[3])
        raw = yaml.safe_load(path.read_text())
        synthesis = TopicSynthesis(**raw)
        scores = asyncio.run(judge_synthesis(llm, synthesis))
        print(yaml.dump(scores, default_flow_style=False))

    elif subcommand == "compare":
        if len(sys.argv) < 5:
            print("Usage: python -m nexus evaluate compare <synthesis-A.yaml> <synthesis-B.yaml>")
            sys.exit(1)
        path_a, path_b = Path(sys.argv[3]), Path(sys.argv[4])
        syn_a = TopicSynthesis(**yaml.safe_load(path_a.read_text()))
        syn_b = TopicSynthesis(**yaml.safe_load(path_b.read_text()))
        result = asyncio.run(compare_syntheses(
            llm, syn_a, syn_b,
            label_a=path_a.stem, label_b=path_b.stem,
        ))
        print(yaml.dump(result, default_flow_style=False))

    else:
        print("Usage: python -m nexus evaluate <synthesis|compare> [args...]")
        sys.exit(1)


def run_serve():
    """Start the dashboard web server."""
    import uvicorn
    from nexus.web.app import create_app

    db_path = Path("data/knowledge.db")
    host = "127.0.0.1"
    port = 8080

    # Parse --port and --host from argv
    if "--port" in sys.argv:
        idx = sys.argv.index("--port")
        if idx + 1 < len(sys.argv):
            port = int(sys.argv[idx + 1])
    if "--host" in sys.argv:
        idx = sys.argv.index("--host")
        if idx + 1 < len(sys.argv):
            host = sys.argv[idx + 1]

    app = create_app(db_path)
    _log_bind_warning(host, port)
    print(f"Starting Nexus dashboard at http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)


def run_all_services():
    """Run the unified always-on service: dashboard + scheduler + Telegram bot."""
    load_dotenv()
    from nexus.config.loader import load_config
    from nexus.runner import run_all

    data_dir = Path("data")
    config_path = data_dir / "config.yaml"

    host = "127.0.0.1"
    port = 8080
    if "--port" in sys.argv:
        idx = sys.argv.index("--port")
        if idx + 1 < len(sys.argv):
            port = int(sys.argv[idx + 1])
    if "--host" in sys.argv:
        idx = sys.argv.index("--host")
        if idx + 1 < len(sys.argv):
            host = sys.argv[idx + 1]

    if not config_path.exists():
        # No config yet — start dashboard only so the web setup wizard can run
        logging.getLogger(__name__).info(
            "No config found — starting dashboard for web setup wizard at http://%s:%s",
            host, port,
        )
        import uvicorn
        from nexus.web.app import create_app
        app = create_app(data_dir / "knowledge.db")
        _log_bind_warning(host, port)
        uvicorn.run(app, host=host, port=port)
        return

    config = load_config(config_path)
    _log_bind_warning(host, port)
    asyncio.run(run_all(config, data_dir, host=host, port=port))


def run_benchmark():
    """Fast benchmark: Suite A on saved fixtures (~3-5 min).

    Usage:
        python -m nexus benchmark --capture           # Poll 2 topics, save fixtures
        python -m nexus benchmark                     # Run Suite A on latest fixtures
        python -m nexus benchmark --threshold 4.0     # Override filter threshold
        python -m nexus benchmark --fixtures DIR      # Specific fixture snapshot
    """
    load_dotenv()
    from nexus.config.loader import load_config
    from nexus.engine.evaluation.fast_bench import (
        capture_benchmark, find_latest_fixture_dir, run_fast_benchmark,
    )
    from nexus.engine.knowledge.store import KnowledgeStore
    from nexus.llm.client import LLMClient

    data_dir = Path("data")
    config_path = data_dir / "config.yaml"

    if not config_path.exists():
        print(f"Config not found at {config_path}. Run: python -m nexus setup")
        sys.exit(1)

    config = load_config(config_path)

    api_key = os.getenv("GEMINI_API_KEY")
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("deepseek")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    litellm_base_url = os.getenv("LITELLM_BASE_URL")
    litellm_api_key = os.getenv("LITELLM_API_KEY")

    llm = LLMClient(
        config.models,
        api_key=api_key,
        anthropic_api_key=anthropic_api_key,
        deepseek_api_key=deepseek_api_key,
        openai_api_key=openai_api_key,
        budget_config=config.budget,
        litellm_base_url=litellm_base_url,
        litellm_api_key=litellm_api_key,
    )

    # Parse CLI args
    do_capture = "--capture" in sys.argv
    threshold = None
    fixture_path = None

    args = sys.argv[2:]
    i = 0
    while i < len(args):
        if args[i] == "--threshold" and i + 1 < len(args):
            threshold = float(args[i + 1])
            i += 2
        elif args[i] == "--fixtures" and i + 1 < len(args):
            fixture_path = Path(args[i + 1])
            i += 2
        elif args[i] == "--capture":
            i += 1
        else:
            i += 1

    async def _run():
        store = KnowledgeStore(data_dir / "knowledge.db")
        await store.initialize()

        if do_capture:
            print("Capturing benchmark fixtures...")
            output_dir = await capture_benchmark(config, llm, store, data_dir)
            print(f"Fixtures saved: {output_dir}")
            return

        # Determine fixture directory
        if fixture_path:
            fdir = fixture_path
        else:
            benchmarks_dir = data_dir / "benchmarks"
            if not benchmarks_dir.exists():
                print("No benchmark fixtures found. Run: python -m nexus benchmark --capture")
                sys.exit(1)
            fdir = find_latest_fixture_dir(benchmarks_dir)

        print(f"Running fast benchmark on fixtures: {fdir}")
        if threshold:
            print(f"  Threshold override: {threshold}")
        print()

        report = await run_fast_benchmark(
            llm=llm,
            store=store,
            fixture_dir=fdir,
            threshold_override=threshold,
            results_dir=data_dir / "benchmarks" / "results",
        )

        # Print markdown report
        md = report.to_markdown()
        print(md)
        print(f"\nDuration: {report.duration_s:.0f}s")

    asyncio.run(_run())


def run_experiment():
    """Run controlled experiment suites for README quality claims."""
    load_dotenv()
    from nexus.config.loader import load_config
    from nexus.engine.evaluation.experiment import run_experiments
    from nexus.llm.client import LLMClient

    data_dir = Path("data")
    config_path = data_dir / "config.yaml"

    if not config_path.exists():
        print(f"Config not found at {config_path}. Run: python -m nexus setup")
        sys.exit(1)

    config = load_config(config_path)

    api_key = os.getenv("GEMINI_API_KEY")
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("deepseek")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    litellm_base_url = os.getenv("LITELLM_BASE_URL")
    litellm_api_key = os.getenv("LITELLM_API_KEY")

    llm = LLMClient(
        config.models,
        api_key=api_key,
        anthropic_api_key=anthropic_api_key,
        deepseek_api_key=deepseek_api_key,
        openai_api_key=openai_api_key,
        budget_config=config.budget,
        litellm_base_url=litellm_base_url,
        litellm_api_key=litellm_api_key,
    )

    # Parse CLI args
    suites = None
    topics = None
    budget_usd = 15.0
    export_fixtures = None
    load_fixtures = None
    rejudge = None
    env = "local"

    args = sys.argv[2:]
    i = 0
    while i < len(args):
        if args[i] == "--suite" and i + 1 < len(args):
            suites = [s.strip().upper() for s in args[i + 1].split(",")]
            i += 2
        elif args[i] == "--topics" and i + 1 < len(args):
            topics = [t.strip() for t in args[i + 1].split(",")]
            i += 2
        elif args[i] == "--budget" and i + 1 < len(args):
            budget_usd = float(args[i + 1])
            i += 2
        elif args[i] == "--export-fixtures" and i + 1 < len(args):
            export_fixtures = Path(args[i + 1])
            i += 2
        elif args[i] == "--load-fixtures" and i + 1 < len(args):
            load_fixtures = Path(args[i + 1])
            i += 2
        elif args[i] == "--rejudge" and i + 1 < len(args):
            rejudge = Path(args[i + 1])
            i += 2
        elif args[i] == "--env" and i + 1 < len(args):
            env = args[i + 1]
            i += 2
        else:
            i += 1

    print("Running experiment suites...")
    if suites:
        print(f"  Suites: {', '.join(suites)}")
    else:
        print("  Suites: A, B, C, D, E, F, G (all)")
    if topics:
        print(f"  Topics: {', '.join(topics)}")
    print(f"  Budget cap: ${budget_usd:.2f}")
    print(f"  Environment: {env}")
    if load_fixtures:
        print(f"  Loading fixtures from: {load_fixtures}")
    if export_fixtures:
        print(f"  Exporting fixtures to: {export_fixtures}")
    if rejudge:
        print(f"  Re-judging syntheses from: {rejudge}")
    print()

    report = asyncio.run(run_experiments(
        config, llm, data_dir,
        suites=suites,
        topics=topics,
        budget_usd=budget_usd,
        export_fixtures=export_fixtures,
        load_fixtures=load_fixtures,
        rejudge=rejudge,
        env=env,
    ))

    # Output full markdown report
    md = report.to_markdown()
    print(md)

    # Save JSON
    exp_dir = data_dir / "experiments"
    exp_dir.mkdir(parents=True, exist_ok=True)
    timestamp = report.timestamp.replace(":", "-").split(".")[0]
    json_path = exp_dir / f"{timestamp}.json"
    json_path.write_text(json.dumps(report.to_json(), indent=2, default=str))
    print(f"\nJSON saved: {json_path}")

    # Save full markdown report
    md_path = Path("docs") / "benchmark-results.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(md)
    print(f"Full report: {md_path}")

    # Print README snippet
    print("\n--- README SNIPPET ---\n")
    print(report.to_readme_snippet())


def run_projection():
    """Future projection utilities: generate, backfill, evaluate, compare."""
    load_dotenv()
    from nexus.config.loader import load_config
    from nexus.engine.knowledge.store import KnowledgeStore
    from nexus.engine.projection.evaluation import (
        auto_evaluate_projections,
        compare_projection_engines,
        cross_topic_bridge_report,
        trajectory_lift_report,
    )
    from nexus.engine.projection.service import (
        backfill_thread_snapshots,
        generate_projections_from_store,
    )
    from nexus.llm.client import LLMClient

    data_dir = Path("data")
    db_path = data_dir / "knowledge.db"
    config_path = data_dir / "config.yaml"
    subcommand = sys.argv[2] if len(sys.argv) > 2 else ""

    start = None
    end = None
    target_date = None
    engine = None
    engines = ["native"]
    min_thread_snapshots = None
    topic_slug_filter = None

    args = sys.argv[3:]
    i = 0
    while i < len(args):
        if args[i] == "--start" and i + 1 < len(args):
            start = date.fromisoformat(args[i + 1])
            i += 2
        elif args[i] == "--end" and i + 1 < len(args):
            end = date.fromisoformat(args[i + 1])
            i += 2
        elif args[i] == "--date" and i + 1 < len(args):
            target_date = date.fromisoformat(args[i + 1])
            i += 2
        elif args[i] == "--engine" and i + 1 < len(args):
            engine = args[i + 1]
            i += 2
        elif args[i] == "--engines" and i + 1 < len(args):
            engines = [part.strip() for part in args[i + 1].split(",") if part.strip()]
            i += 2
        elif args[i] == "--min-thread-snapshots" and i + 1 < len(args):
            min_thread_snapshots = int(args[i + 1])
            i += 2
        elif args[i] == "--topic" and i + 1 < len(args):
            topic_slug_filter = args[i + 1]
            i += 2
        else:
            i += 1

    async def _run():
        store = KnowledgeStore(db_path)
        await store.initialize()
        try:
            config = load_config(config_path) if config_path.exists() else None

            if subcommand == "generate":
                if config is None:
                    raise SystemExit(
                        "Usage: python -m nexus projection generate [--date YYYY-MM-DD] "
                        "[--engine actor|native] [--min-thread-snapshots N] [--topic topic-slug]"
                    )

                if topic_slug_filter:
                    filtered_topics = [
                        topic for topic in config.topics
                        if topic.name.lower().replace(" ", "-").replace("/", "-") == topic_slug_filter
                    ]
                    if not filtered_topics:
                        raise SystemExit(f"Topic '{topic_slug_filter}' not found in config.")
                    config.topics = filtered_topics

                selected_engine = engine or config.future_projection.engine
                llm = None
                if selected_engine in ("native", "actor"):
                    api_key = os.getenv("GEMINI_API_KEY")
                    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
                    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("deepseek")
                    openai_api_key = os.getenv("OPENAI_API_KEY")
                    ollama_base_url = os.getenv("OLLAMA_BASE_URL")
                    litellm_base_url = os.getenv("LITELLM_BASE_URL") or os.getenv("LITELLM_PROXY_URL")
                    litellm_api_key = os.getenv("LITELLM_API_KEY") or os.getenv("LITELLM_PROXY_API_KEY")
                    has_provider = any([
                        api_key,
                        anthropic_api_key,
                        deepseek_api_key,
                        openai_api_key,
                        ollama_base_url,
                        litellm_base_url and litellm_api_key,
                    ])
                    if has_provider or config.preset == "free":
                        llm = LLMClient(
                            config.models,
                            api_key=api_key,
                            anthropic_api_key=anthropic_api_key,
                            deepseek_api_key=deepseek_api_key,
                            openai_api_key=openai_api_key,
                            ollama_base_url=ollama_base_url,
                            budget_config=config.budget,
                            litellm_base_url=litellm_base_url,
                            litellm_api_key=litellm_api_key,
                        )

                result = await generate_projections_from_store(
                    store,
                    llm,
                    config,
                    target_date=target_date,
                    min_thread_snapshots_override=min_thread_snapshots,
                    engine_override=engine,
                    experiments_dir=data_dir / "experiments",
                )
                print(json.dumps(result, indent=2, default=str))
                return

            if subcommand == "backfill":
                result = await backfill_thread_snapshots(store)
                print(json.dumps(result, indent=2))
                return

            if subcommand in ("evaluate", "compare"):
                # Create LLM client for semantic judge if any provider is configured
                eval_llm = None
                if config is not None:
                    api_key = os.getenv("GEMINI_API_KEY")
                    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
                    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("deepseek")
                    openai_api_key = os.getenv("OPENAI_API_KEY")
                    ollama_base_url = os.getenv("OLLAMA_BASE_URL")
                    litellm_base_url = os.getenv("LITELLM_BASE_URL") or os.getenv("LITELLM_PROXY_URL")
                    litellm_api_key = os.getenv("LITELLM_API_KEY") or os.getenv("LITELLM_PROXY_API_KEY")
                    has_provider = any([
                        api_key, anthropic_api_key, deepseek_api_key, openai_api_key,
                        ollama_base_url, litellm_base_url and litellm_api_key,
                    ])
                    if has_provider or config.preset == "free":
                        eval_llm = LLMClient(
                            config.models, api_key=api_key,
                            anthropic_api_key=anthropic_api_key,
                            deepseek_api_key=deepseek_api_key,
                            openai_api_key=openai_api_key,
                            ollama_base_url=ollama_base_url,
                            budget_config=config.budget,
                            litellm_base_url=litellm_base_url,
                            litellm_api_key=litellm_api_key,
                        )

            if subcommand == "evaluate":
                if start is None or end is None:
                    raise SystemExit("Usage: python -m nexus projection evaluate --start YYYY-MM-DD --end YYYY-MM-DD [--engine native]")
                report = {
                    "projection_hit_rate": await auto_evaluate_projections(store, start=start, end=end, engine=engine, llm=eval_llm),
                    "trajectory_lift": await trajectory_lift_report(store, start=start, end=end),
                    "cross_topic_bridge_utility": await cross_topic_bridge_report(store, start=start, end=end),
                }
                print(json.dumps(report, indent=2, default=str))
                return

            if subcommand == "compare":
                if start is None or end is None:
                    raise SystemExit(
                        "Usage: python -m nexus projection compare --engines actor,native --start YYYY-MM-DD --end YYYY-MM-DD"
                    )
                report = await compare_projection_engines(store, start=start, end=end, engines=engines, llm=eval_llm)
                print(json.dumps(report, indent=2, default=str))
                return

            if subcommand == "consolidate":
                from nexus.engine.synthesis.threads import find_merge_candidates
                dry_run = "--dry-run" in sys.argv
                jaccard_only = "--jaccard-only" in sys.argv
                threshold = 0.5
                for idx, arg in enumerate(args):
                    if arg == "--threshold" and idx + 1 < len(args):
                        threshold = float(args[idx + 1])

                all_threads = await store.get_active_threads(topic_slug_filter)
                print(f"Found {len(all_threads)} active/emerging threads" +
                      (f" for {topic_slug_filter}" if topic_slug_filter else ""))

                # Build LLM for ambiguous pairs if config available
                consolidate_llm = None
                if config is not None:
                    api_key = os.getenv("GEMINI_API_KEY")
                    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
                    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("deepseek")
                    openai_api_key = os.getenv("OPENAI_API_KEY")
                    ollama_base_url = os.getenv("OLLAMA_BASE_URL")
                    litellm_base_url = os.getenv("LITELLM_BASE_URL") or os.getenv("LITELLM_PROXY_URL")
                    litellm_api_key = os.getenv("LITELLM_API_KEY") or os.getenv("LITELLM_PROXY_API_KEY")
                    has_provider = any([
                        api_key, anthropic_api_key, deepseek_api_key, openai_api_key,
                        ollama_base_url, litellm_base_url and litellm_api_key,
                    ])
                    if has_provider or config.preset == "free":
                        consolidate_llm = LLMClient(
                            config.models, api_key=api_key,
                            anthropic_api_key=anthropic_api_key,
                            deepseek_api_key=deepseek_api_key,
                            openai_api_key=openai_api_key,
                            ollama_base_url=ollama_base_url,
                            budget_config=config.budget,
                            litellm_base_url=litellm_base_url,
                            litellm_api_key=litellm_api_key,
                        )

                pairs = await find_merge_candidates(
                    all_threads, None if jaccard_only else consolidate_llm,
                    high_threshold=threshold,
                )
                if not pairs:
                    print("No merge candidates found.")
                    return

                thread_map = {t["id"]: t for t in all_threads}
                for keep_id, absorb_id in pairs:
                    k = thread_map.get(keep_id, {})
                    a = thread_map.get(absorb_id, {})
                    print(f"  MERGE: \"{a.get('headline', '?')}\" (id={absorb_id}) "
                          f"→ \"{k.get('headline', '?')}\" (id={keep_id})")

                if dry_run:
                    print(f"\nDry run: {len(pairs)} pairs would be merged.")
                    return

                for keep_id, absorb_id in pairs:
                    result = await store.merge_threads(keep_id, absorb_id)
                    print(f"  Merged {absorb_id} → {keep_id}: {result}")
                print(f"\nConsolidated {len(pairs)} thread pairs.")
                return

            if subcommand == "purge-templates":
                confirm = "--confirm" in sys.argv
                result = await store.purge_template_projections(dry_run=not confirm)
                print(f"Template projection items found: {result['items_found']}")
                if result.get("dry_run"):
                    print("Dry run — no changes made. Use --confirm to delete.")
                else:
                    print(f"Deleted: {result['items_deleted']} items, {result['projections_deleted']} empty projections")
                return

            raise SystemExit(
                "Usage: python -m nexus projection <generate|backfill|evaluate|compare|consolidate|purge-templates> [args...]"
            )
        finally:
            await store.close()

    asyncio.run(_run())


def run_forecast():
    """Quantified forecast utilities: generate, replay, resolve, benchmark, backfill, readiness."""
    load_dotenv()
    import shutil
    import sqlite3
    import tempfile
    from nexus.config.models import KalshiBenchmarkConfig
    from nexus.config.loader import load_config
    from nexus.engine.knowledge.store import KnowledgeStore
    from nexus.engine.projection.evaluation import (
        audit_forecast_leakage,
        export_graph_bundles,
    )
    from nexus.engine.projection.kalshi import (
        KalshiClient,
        KalshiLedger,
        bootstrap_kalshi_credentials,
        compare_forecasts_to_kalshi,
        sync_kalshi_tickers,
    )
    from nexus.engine.projection.service import backfill_signal_rich_profile, backfill_syntheses, generate_forecasts_from_store, topic_slug_from_name
    from nexus.llm.client import LLMClient

    data_dir = Path("data")
    db_path = data_dir / "knowledge.db"
    config_path = data_dir / "config.yaml"
    subcommand = sys.argv[2] if len(sys.argv) > 2 else ""

    start = None
    end = None
    target_date = None
    _through = None  # noqa: F841 — reserved for future CLI wiring
    engine = None
    engines = ["actor", "native"]
    min_thread_snapshots = None
    topic_slug_filter = None
    profile = "signal-rich"
    _mode = "audit"  # noqa: F841 — reserved for future CLI wiring
    strict = True
    tickers: list[str] = []
    mapping_file = None
    cred_file = None
    key_path = None
    horizon_days = 7
    max_cases = 30
    min_significance = 7

    args = sys.argv[3:]
    i = 0
    while i < len(args):
        if args[i] == "--start" and i + 1 < len(args):
            start = date.fromisoformat(args[i + 1])
            i += 2
        elif args[i] == "--end" and i + 1 < len(args):
            end = date.fromisoformat(args[i + 1])
            i += 2
        elif args[i] == "--through" and i + 1 < len(args):
            _through = date.fromisoformat(args[i + 1])  # noqa: F841
            i += 2
        elif args[i] == "--date" and i + 1 < len(args):
            target_date = date.fromisoformat(args[i + 1])
            i += 2
        elif args[i] == "--engine" and i + 1 < len(args):
            engine = args[i + 1]
            i += 2
        elif args[i] == "--engines" and i + 1 < len(args):
            engines = [part.strip() for part in args[i + 1].split(",") if part.strip()]
            i += 2
        elif args[i] == "--min-thread-snapshots" and i + 1 < len(args):
            min_thread_snapshots = int(args[i + 1])
            i += 2
        elif args[i] == "--topic" and i + 1 < len(args):
            topic_slug_filter = args[i + 1]
            i += 2
        elif args[i] == "--profile" and i + 1 < len(args):
            profile = args[i + 1]
            i += 2
        elif args[i] == "--mode" and i + 1 < len(args):
            _mode = args[i + 1]  # noqa: F841
            i += 2
        elif args[i] == "--tickers" and i + 1 < len(args):
            tickers = [part.strip() for part in args[i + 1].split(",") if part.strip()]
            i += 2
        elif args[i] == "--mapping-file" and i + 1 < len(args):
            mapping_file = args[i + 1]
            i += 2
        elif args[i] == "--cred-file" and i + 1 < len(args):
            cred_file = args[i + 1]
            i += 2
        elif args[i] == "--key-path" and i + 1 < len(args):
            key_path = args[i + 1]
            i += 2
        elif args[i] == "--horizon" and i + 1 < len(args):
            horizon_days = int(args[i + 1])
            i += 2
        elif args[i] == "--max-cases" and i + 1 < len(args):
            max_cases = int(args[i + 1])
            i += 2
        elif args[i] == "--min-significance" and i + 1 < len(args):
            min_significance = int(args[i + 1])
            i += 2
        elif args[i] == "--strict":
            strict = True  # noqa: F841
            i += 1
        elif args[i] == "--no-strict":
            strict = False  # noqa: F841
            i += 1
        else:
            i += 1

    def _snapshot_sqlite_db(source: Path, target: Path) -> None:
        src = sqlite3.connect(f"file:{source}?mode=ro", uri=True)
        dst = sqlite3.connect(str(target))
        try:
            src.backup(dst)
        finally:
            dst.close()
            src.close()

    def _optional_llm(config, selected_engine: str):
        if selected_engine in {"market"}:
            return None
        api_key = os.getenv("GEMINI_API_KEY")
        anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        deepseek_api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("deepseek")
        openai_api_key = os.getenv("OPENAI_API_KEY")
        ollama_base_url = os.getenv("OLLAMA_BASE_URL")
        litellm_base_url = os.getenv("LITELLM_BASE_URL") or os.getenv("LITELLM_PROXY_URL")
        litellm_api_key = os.getenv("LITELLM_API_KEY") or os.getenv("LITELLM_PROXY_API_KEY")
        has_provider = any([
            api_key,
            anthropic_api_key,
            deepseek_api_key,
            openai_api_key,
            ollama_base_url,
            litellm_base_url and litellm_api_key,
        ])
        if not has_provider and config.preset != "free":
            return None
        return LLMClient(
            config.models,
            api_key=api_key,
            anthropic_api_key=anthropic_api_key,
            deepseek_api_key=deepseek_api_key,
            openai_api_key=openai_api_key,
            ollama_base_url=ollama_base_url,
            budget_config=config.budget,
            litellm_base_url=litellm_base_url,
            litellm_api_key=litellm_api_key,
        )

    async def _run():
        store = None
        temp_dir = None
        kalshi_ledger = None
        try:
            config = load_config(config_path) if config_path.exists() else None
            kalshi_config = config.future_projection.kalshi if config else KalshiBenchmarkConfig()
            if config and topic_slug_filter:
                filtered_topics = [
                    topic for topic in config.topics
                    if topic.name.lower().replace(" ", "-").replace("/", "-") == topic_slug_filter
                ]
                if not filtered_topics:
                    raise SystemExit(f"Topic '{topic_slug_filter}' not found in config.")
                config.topics = filtered_topics

            if subcommand in {"kalshi-sync", "kalshi-bootstrap", "kalshi-auth-check"}:
                store = None
            elif subcommand in {"benchmark", "replay", "audit-leakage", "export-graph", "readiness", "kalshi-compare", "kalshi-benchmark"}:
                temp_dir = Path(tempfile.mkdtemp(prefix="nexus-forecast-benchmark-"))
                snapshot_path = temp_dir / "knowledge.db"
                _snapshot_sqlite_db(db_path, snapshot_path)
                store = KnowledgeStore(snapshot_path)
            else:
                store = KnowledgeStore(db_path)

            if store is not None:
                await store.initialize()

            if subcommand == "generate":
                if config is None:
                    raise SystemExit(
                        "Usage: python -m nexus forecast generate [--date YYYY-MM-DD] "
                        "[--engine actor|native] [--min-thread-snapshots N] [--topic topic-slug]"
                    )
                llm = _optional_llm(config, engine or "actor")
                result = await generate_forecasts_from_store(
                    store,
                    llm,
                    config,
                    target_date=target_date,
                    min_thread_snapshots_override=min_thread_snapshots,
                    engine_override=engine or "actor",
                    experiments_dir=data_dir / "experiments",
                )
                print(json.dumps(result, indent=2, default=str))
                return

            if subcommand == "export-graph":
                if config is None or start is None or end is None:
                    raise SystemExit(
                        "Usage: python -m nexus forecast export-graph --start YYYY-MM-DD --end YYYY-MM-DD "
                        "[--profile signal-rich]"
                    )
                export_dir = Path(config.future_projection.graph_sidecars.export_dir)
                report = await export_graph_bundles(
                    store,
                    config,
                    start=start,
                    end=end,
                    profile=profile,
                    target_dir=export_dir,
                )
                print(json.dumps(report, indent=2, default=str))
                return

            if subcommand == "audit-leakage":
                if config is None or start is None or end is None:
                    raise SystemExit(
                        "Usage: python -m nexus forecast audit-leakage --start YYYY-MM-DD --end YYYY-MM-DD "
                        "[--profile signal-rich]"
                    )
                report = await audit_forecast_leakage(
                    store,
                    config,
                    start=start,
                    end=end,
                    profile=profile,
                )
                out_dir = data_dir / "benchmarks"
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"forecast-audit-leakage-{start.isoformat()}-{end.isoformat()}.json"
                out_path.write_text(json.dumps(report, indent=2, default=str))
                print(json.dumps(report, indent=2, default=str))
                print(f"\nSaved: {out_path}")
                return

            if subcommand == "backfill":
                if config is None:
                    raise SystemExit("Usage: python -m nexus forecast backfill --profile signal-rich")
                if profile != "signal-rich":
                    raise SystemExit(f"Unsupported backfill profile: {profile}")
                result = await backfill_signal_rich_profile(
                    store,
                    config,
                    target_dir=data_dir / "benchmarks" / "forecast_signal_rich",
                )
                print(json.dumps(result, indent=2, default=str))
                return

            if subcommand == "backfill-keys":
                result = await store.backfill_forecast_keys(
                    start=start,
                    end=end,
                    engine=engine,
                )
                print(json.dumps(result, indent=2, default=str))
                return

            if subcommand == "backfill-syntheses":
                if config is None or not topic_slug_filter:
                    raise SystemExit(
                        "Usage: python -m nexus forecast backfill-syntheses --topic SLUG "
                        "[--start YYYY-MM-DD] [--end YYYY-MM-DD]"
                    )
                llm = _optional_llm(config, "native")
                if llm is None:
                    raise SystemExit("LLM provider required for synthesis backfill. Set GEMINI_API_KEY or similar.")
                result = await backfill_syntheses(
                    store,
                    llm,
                    config,
                    topic_slug=topic_slug_filter,
                    start=start,
                    end=end,
                )
                print(json.dumps(result, indent=2, default=str))
                return

            if subcommand == "kalshi-bootstrap":
                result = bootstrap_kalshi_credentials(
                    cred_path=Path(cred_file or "kalshi-cred"),
                    config=kalshi_config,
                    target_key_path=Path(key_path) if key_path else None,
                )
                print(json.dumps(result, indent=2, default=str))
                return

            if subcommand == "kalshi-auth-check":
                report = await KalshiClient(kalshi_config).auth_check()
                out_dir = data_dir / "benchmarks"
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / "forecast-kalshi-auth-check.json"
                out_path.write_text(json.dumps(report, indent=2, default=str))
                print(json.dumps(report, indent=2, default=str))
                print(f"\nSaved: {out_path}")
                return

            if subcommand == "kalshi-sync":
                if config is None or start is None or end is None or not tickers:
                    raise SystemExit(
                        "Usage: python -m nexus forecast kalshi-sync --tickers T1,T2 --start YYYY-MM-DD --end YYYY-MM-DD"
                    )
                kalshi_ledger = KalshiLedger(Path(config.future_projection.kalshi.ledger_path))
                await kalshi_ledger.initialize()
                result = await sync_kalshi_tickers(
                    kalshi_ledger,
                    KalshiClient(config.future_projection.kalshi),
                    tickers=tickers,
                    start=start,
                    end=end,
                )
                print(json.dumps(result, indent=2, default=str))
                return

            if subcommand == "kalshi-compare":
                if config is None or start is None or end is None:
                    raise SystemExit(
                        "Usage: python -m nexus forecast kalshi-compare --start YYYY-MM-DD --end YYYY-MM-DD "
                        "[--mapping-file PATH]"
                    )
                kalshi_ledger = KalshiLedger(Path(config.future_projection.kalshi.ledger_path))
                await kalshi_ledger.initialize()
                report = await compare_forecasts_to_kalshi(
                    store,
                    kalshi_ledger,
                    start=start,
                    end=end,
                    mapping_path=Path(mapping_file or config.future_projection.kalshi.mapping_file),
                    engine=engine,
                )
                out_dir = data_dir / "benchmarks"
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"forecast-kalshi-compare-{start.isoformat()}-{end.isoformat()}.json"
                out_path.write_text(json.dumps(report, indent=2, default=str))
                print(json.dumps(report, indent=2, default=str))
                print(f"\nSaved: {out_path}")
                return

            if subcommand == "kalshi-loop":
                if config is None:
                    raise SystemExit(
                        "Usage: python -m nexus forecast kalshi-loop [--date YYYY-MM-DD] "
                        "[--engine structural|actor|naked|graphrag|perspective|debate] "
                        "[--engines all|structural,actor,...] [--topic topic-slug]"
                    )
                from nexus.engine.projection.service import run_kalshi_loop
                from nexus.engine.synthesis.knowledge import TopicSynthesis

                ALL_ENGINES = ["structural", "actor", "naked", "graphrag", "perspective", "debate"]
                engine_list = engines if engines != ["actor", "native"] else None  # default wasn't changed
                if engine_list and "all" in engine_list:
                    engine_list = ALL_ENGINES
                elif engine_list:
                    pass  # use as-is from --engines arg
                elif engine:
                    engine_list = [engine]
                else:
                    engine_list = ["structural"]

                kalshi_client = KalshiClient(kalshi_config)
                loop_llm = _optional_llm(config, "structural")
                loop_date = target_date or date.today()

                # Initialize Kalshi ledger for price snapshots
                kalshi_ledger = KalshiLedger(Path(kalshi_config.ledger_path))
                await kalshi_ledger.initialize()

                # Load latest syntheses for each topic
                loop_syntheses = []
                for topic in config.topics:
                    slug = topic_slug_from_name(topic.name)
                    synthesis_dates = [date.fromisoformat(raw) for raw in await store.get_synthesis_dates(slug)]
                    syn_date = next((d for d in synthesis_dates if d <= loop_date), None) if synthesis_dates else None
                    if not syn_date:
                        continue
                    raw_syn = await store.get_synthesis(slug, syn_date)
                    if raw_syn:
                        from nexus.engine.projection.service import hydrate_synthesis_threads
                        loop_syntheses.append(
                            await hydrate_synthesis_threads(store, TopicSynthesis(**raw_syn), topic_slug=slug)
                        )

                all_results = {}
                for engine_name in engine_list:
                    print(f"\n{'='*60}")
                    print(f"Running engine: {engine_name}")
                    print(f"{'='*60}")
                    result = await run_kalshi_loop(
                        store,
                        loop_llm,
                        loop_syntheses,
                        run_date=loop_date,
                        kalshi_client=kalshi_client,
                        kalshi_config=kalshi_config,
                        engine=engine_name,
                        topic_configs=config.topics,
                        ledger=kalshi_ledger if engine_name == engine_list[0] else None,  # snapshot once
                    )
                    all_results[engine_name] = result
                    print(f"  Matched: {result['markets_matched']}, Generated: {result['questions_generated']}")

                # Print comparison if multiple engines
                if len(engine_list) > 1:
                    print(f"\n{'='*60}")
                    print("ENGINE COMPARISON")
                    print(f"{'='*60}")
                    # Collect all tickers across engines
                    all_tickers: dict[str, dict] = {}
                    for eng_name, res in all_results.items():
                        for div in res.get("divergences", []):
                            ticker = div["ticker"]
                            if ticker not in all_tickers:
                                all_tickers[ticker] = {"question": div["question"][:60], "market": div["kalshi_probability"]}
                            all_tickers[ticker][eng_name] = div["our_probability"]

                    header = f"{'Ticker':>30s}  {'Market':>6s}"
                    for eng_name in engine_list:
                        header += f"  {eng_name:>10s}"
                    print(header)
                    print("-" * len(header))
                    for ticker, data in all_tickers.items():
                        row = f"{ticker:>30s}  {data['market']:>5.0%}"
                        for eng_name in engine_list:
                            prob = data.get(eng_name)
                            row += f"  {prob:>9.0%}" if prob is not None else f"  {'—':>10s}"
                        print(row)

                out_dir = data_dir / "benchmarks"
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"forecast-kalshi-loop-{loop_date.isoformat()}.json"
                output = all_results if len(engine_list) > 1 else all_results[engine_list[0]]
                out_path.write_text(json.dumps(output, indent=2, default=str))
                print(f"\nSaved: {out_path}")
                return

            if subcommand == "resolve":
                if config is None:
                    raise SystemExit("Usage: python -m nexus forecast resolve [--date YYYY-MM-DD]")
                from nexus.engine.projection.kalshi_resolution import resolve_kalshi_forecasts
                kalshi_client = KalshiClient(kalshi_config)
                resolve_date = target_date or date.today()
                result = await resolve_kalshi_forecasts(
                    store, kalshi_client, as_of=resolve_date,
                )
                print(json.dumps(result, indent=2, default=str))
                if result["resolved"]:
                    print(f"\nResolved {result['resolved']} predictions, mean Brier: {result['mean_brier']}")
                else:
                    print(f"\nNo settled markets yet ({result['still_open']} still open, {result['errors']} errors)")
                return

            if subcommand == "calibrate":
                from nexus.engine.projection.swarm import fit_calibration_params
                calibration_data = await store.get_historical_calibration()
                # Reshape: store returns {probability, resolved_bool}, we need {raw_probability, ...}
                samples = []
                for row in calibration_data:
                    samples.append({
                        "raw_probability": row["probability"],
                        "resolved_bool": row["resolved_bool"],
                        "market_probability": row.get("base_rate"),
                    })
                result = fit_calibration_params(samples)
                print(json.dumps(result, indent=2, default=str))
                if result["improved"]:
                    print(
                        f"\nRecommended: gamma={result['gamma']}, "
                        f"swarm_weight={result['swarm_weight']} "
                        f"(Brier: {result['baseline_brier']} → {result['mean_brier']})"
                    )
                elif result.get("reason"):
                    print(f"\n{result['reason']}")
                else:
                    print(f"\nCurrent defaults are optimal (Brier={result['mean_brier']}, n={result['n_samples']})")
                return

            if subcommand == "kalshi-scan":
                kalshi_client = KalshiClient(kalshi_config)
                keywords_str = None
                for i, arg in enumerate(sys.argv):
                    if arg == "--keywords" and i + 1 < len(sys.argv):
                        keywords_str = sys.argv[i + 1]
                keyword_list = [k.strip().lower() for k in keywords_str.split(",")] if keywords_str else []
                all_events: list[dict] = []
                cursor = None
                for _ in range(10):  # max 10 pages
                    page = await kalshi_client.list_events(status="open", limit=200, cursor=cursor)
                    events_list = page.get("events", [])
                    if not events_list:
                        break
                    all_events.extend(events_list)
                    cursor = page.get("cursor")
                    if not cursor:
                        break

                # Group by category/series and count markets
                categories: dict[str, dict] = {}
                for event in all_events:
                    cat = event.get("category", "unknown")
                    title = event.get("title", "")
                    markets = event.get("markets", [])
                    market_count = len(markets) if isinstance(markets, list) else 0
                    if keyword_list:
                        haystack = (title + " " + event.get("subtitle", "")).lower()
                        if not any(kw in haystack for kw in keyword_list):
                            continue
                    if cat not in categories:
                        categories[cat] = {"events": 0, "markets": 0, "sample_titles": []}
                    categories[cat]["events"] += 1
                    categories[cat]["markets"] += market_count
                    if len(categories[cat]["sample_titles"]) < 3:
                        categories[cat]["sample_titles"].append(title[:100])

                ranked = sorted(categories.items(), key=lambda x: x[1]["markets"], reverse=True)
                report = {
                    "total_events": len(all_events),
                    "matched_events": sum(c["events"] for c in categories.values()),
                    "keywords": keyword_list or "(none)",
                    "categories": [
                        {"category": cat, **data}
                        for cat, data in ranked
                    ],
                }
                print(json.dumps(report, indent=2, default=str))
                return

            if subcommand == "kalshi-benchmark":
                from nexus.engine.projection.kalshi_benchmark import (
                    build_benchmark_dataset,
                    build_benchmark_from_metadata,
                    discover_settled_markets,
                    load_benchmark_dataset,
                    run_benchmark,
                    save_benchmark_dataset,
                    save_benchmark_report,
                )

                bench_dir = data_dir / "benchmarks"
                bench_dir.mkdir(parents=True, exist_ok=True)
                dataset_path = bench_dir / "kalshi_benchmark_dataset.json"
                report_path = bench_dir / "kalshi_engine_comparison.json"

                bench_action = args[0] if args else ""

                if bench_action == "--discover":
                    if config is None:
                        raise SystemExit(
                            "Usage: python -m nexus forecast kalshi-benchmark --discover"
                        )
                    kalshi_client = KalshiClient(kalshi_config)
                    kalshi_ledger = KalshiLedger(
                        Path(kalshi_config.ledger_path)
                    )
                    await kalshi_ledger.initialize()

                    # Parse discover-specific flags
                    days_back = 90
                    max_markets = 500
                    exclude_cats = {"Crypto", "Sports", "Entertainment", "Mentions",
                                    "Climate and Weather", "Social"}
                    for j, arg in enumerate(args):
                        if arg == "--days-back" and j + 1 < len(args):
                            days_back = int(args[j + 1])
                        elif arg == "--max-markets" and j + 1 < len(args):
                            max_markets = int(args[j + 1])
                        elif arg == "--no-exclude":
                            exclude_cats = set()

                    print(f"Discovering settled Kalshi markets (last {days_back}d, "
                          f"max {max_markets}, excluding {len(exclude_cats)} categories)...")
                    settled = await discover_settled_markets(
                        kalshi_client, days_back=days_back,
                        max_markets=max_markets,
                        exclude_categories=exclude_cats,
                    )
                    print(f"Found {len(settled)} settled markets.")

                    if not settled:
                        print("No settled markets found.")
                        return

                    # Show category breakdown
                    from collections import Counter as _Counter
                    cats = _Counter(m.category for m in settled)
                    for cat, n in cats.most_common(15):
                        print(f"  {cat:30s} {n:5d}")

                    # Backfill price history for each settled market
                    print(f"\nBackfilling price history for {len(settled)} tickers...")
                    from datetime import timedelta as _td
                    synced = 0
                    for idx, sm in enumerate(settled):
                        sync_start = sm.settlement_date - _td(days=35)
                        sync_end = sm.settlement_date
                        try:
                            await sync_kalshi_tickers(
                                kalshi_ledger, kalshi_client,
                                tickers=[sm.ticker],
                                start=sync_start, end=sync_end,
                            )
                            synced += 1
                        except Exception:
                            pass  # silent — many tickers won't have candlestick data
                        if (idx + 1) % 50 == 0:
                            print(f"  Progress: {idx + 1}/{len(settled)} tickers synced ({synced} with data)")

                    # Build dataset — try snapshot cutoffs first, fall back to metadata
                    print("Building benchmark dataset...")
                    questions = await build_benchmark_dataset(
                        kalshi_ledger, settled
                    )
                    if not questions:
                        print("  No candlestick history available — using market metadata prices.")
                        questions = await build_benchmark_from_metadata(
                            kalshi_ledger, settled
                        )
                    save_benchmark_dataset(questions, dataset_path)
                    print(
                        f"Saved {len(questions)} benchmark questions to {dataset_path}"
                    )

                    # Show probability distribution
                    extreme = sum(1 for q in questions if q.market_prob_at_cutoff <= 0.05 or q.market_prob_at_cutoff >= 0.95)
                    mid = sum(1 for q in questions if 0.10 <= q.market_prob_at_cutoff <= 0.90)
                    print(f"  Extreme (≤0.05 or ≥0.95): {extreme}")
                    print(f"  Mid-range (0.10-0.90): {mid}")
                    print(f"  Other: {len(questions) - extreme - mid}")
                    return

                if bench_action == "--run":
                    if not dataset_path.exists():
                        raise SystemExit(
                            f"No benchmark dataset at {dataset_path}. "
                            "Run: python -m nexus forecast kalshi-benchmark --discover"
                        )

                    dataset = load_benchmark_dataset(dataset_path)
                    print(f"Loaded {len(dataset)} benchmark questions.")

                    # Parse run-specific flags
                    filter_mid = "--filter-mid" in args
                    independent = "--independent" in args
                    concurrency = 5
                    for j, arg in enumerate(args):
                        if arg == "--concurrency" and j + 1 < len(args):
                            concurrency = int(args[j + 1])

                    if filter_mid:
                        full_count = len(dataset)
                        dataset = [q for q in dataset
                                   if 0.05 < q.market_prob_at_cutoff < 0.95]
                        print(f"  Filtered to {len(dataset)} non-extreme questions "
                              f"(from {full_count}).")

                    # Determine which engines to run
                    engine_names = engines  # from --engines arg
                    if "all" in engine_names:
                        engine_names = [
                            "market", "naked", "actor", "graphrag", "perspective"
                        ]

                    from nexus.engine.projection.kalshi_benchmark import MarketBaselineEngine
                    from nexus.engine.projection.naked_engine import NakedBenchmarkEngine
                    from nexus.engine.projection.actor_engine import ActorForecastEngine
                    from nexus.engine.projection.graphrag_engine import GraphRAGBenchmarkEngine
                    from nexus.engine.projection.perspective_engine import PerspectiveBenchmarkEngine
                    from nexus.engine.projection.debate_engine import DebateBenchmarkEngine
                    from nexus.engine.projection.structural_engine import StructuralBenchmarkEngine

                    engine_map = {
                        "market": MarketBaselineEngine,
                        "naked": NakedBenchmarkEngine,
                        "actor": ActorForecastEngine,
                        "graphrag": GraphRAGBenchmarkEngine,
                        "perspective": PerspectiveBenchmarkEngine,
                        "debate": DebateBenchmarkEngine,
                        "structural": StructuralBenchmarkEngine,
                    }

                    bench_engines = {}
                    for name in engine_names:
                        if name in engine_map:
                            bench_engines[name] = engine_map[name]()
                        else:
                            print(f"  Warning: unknown engine '{name}', skipping")

                    if not bench_engines:
                        raise SystemExit("No valid engines to benchmark.")

                    # Use first non-market engine to decide LLM needs
                    llm_engine = next((n for n in engine_names if n != "market"), "naked")
                    llm = _optional_llm(config, llm_engine) if config else None
                    if llm and store:
                        await llm.set_store(store)

                    mode = "INDEPENDENT (no market anchor)" if independent else "anchored to market"
                    print(f"Running benchmark with engines: {list(bench_engines.keys())} "
                          f"(concurrency={concurrency}, mode={mode})")
                    report = await run_benchmark(
                        dataset, bench_engines, llm=llm, store=store,
                        concurrency=concurrency, independent=independent,
                    )
                    save_benchmark_report(report, report_path)
                    print(f"Saved report to {report_path}")

                    # Print summary
                    print(f"\n{'Engine':<20s} {'Mean Brier':>10s} {'Questions':>10s}")
                    print("-" * 42)
                    for ename, eresult in report.engine_results.items():
                        brier = eresult.get("mean_brier")
                        n = eresult.get("questions_answered", 0)
                        brier_str = f"{brier:.4f}" if brier is not None else "N/A"
                        print(f"{ename:<20s} {brier_str:>10s} {n:>10d}")
                    return

                if bench_action == "--report":
                    if not report_path.exists():
                        raise SystemExit(
                            f"No benchmark report at {report_path}. "
                            "Run: python -m nexus forecast kalshi-benchmark --run --engines all"
                        )
                    raw = json.loads(report_path.read_text())
                    print(json.dumps(raw, indent=2, default=str))
                    return

                raise SystemExit(
                    "Usage: python -m nexus forecast kalshi-benchmark <--discover|--run|--report>\n"
                    "  --discover           Find settled markets + backfill prices\n"
                    "  --run --engines all  Run all engines on benchmark dataset\n"
                    "  --report             Display saved benchmark results"
                )

            if subcommand == "hindcast":
                from nexus.engine.projection.hindcast import backtest_forecasts, serialize_report
                from datetime import timedelta as _td

                llm = _optional_llm(config, "structural")

                hindcast_start = start or (date.today() - _td(days=30))
                hindcast_end = end or date.today()
                engine_list = engines if engines != ["actor", "native"] else ["structural"]

                # Build topic list from config
                topic_pairs = [
                    (topic_slug_from_name(t.name), t.name)
                    for t in config.topics
                ]
                if topic_slug_filter:
                    topic_pairs = [(s, n) for s, n in topic_pairs if s == topic_slug_filter]

                if not topic_pairs:
                    raise SystemExit("No topics matched. Check --topic filter or config.")

                print(f"Hindcast benchmark: {hindcast_start} → {hindcast_end}")
                print(f"  Engines: {engine_list}")
                print(f"  Topics: {[s for s, _ in topic_pairs]}")
                print(f"  Horizon: {horizon_days}d, max cases/topic: {max_cases}, min significance: {min_significance}")

                report = await backtest_forecasts(
                    store, llm,
                    topics=topic_pairs,
                    start=hindcast_start,
                    end=hindcast_end,
                    engines=engine_list,
                    horizon_days=horizon_days,
                    min_significance=min_significance,
                    max_cases_per_topic=max_cases,
                    persist="--no-persist" not in sys.argv,
                )

                report_data = serialize_report(report)
                report_path = data_dir / "benchmarks" / f"hindcast-{hindcast_start.isoformat()}-{hindcast_end.isoformat()}.json"
                report_path.parent.mkdir(parents=True, exist_ok=True)
                report_path.write_text(json.dumps(report_data, indent=2, default=str))

                print(f"\nResults ({report.total_cases} cases: {report.positive_cases}+ / {report.negative_cases}-):")
                for eng, results in report.engine_results.items():
                    if results.get("mean_brier") is not None:
                        print(f"  {eng}: mean_brier={results['mean_brier']:.4f}, "
                              f"median={results['median_brier']:.4f}, n={results['n']}")
                    else:
                        print(f"  {eng}: no results")
                print(f"\nReport saved to {report_path}")
                return

            raise SystemExit(
                "Usage: python -m nexus forecast <generate|replay|resolve|benchmark|audit-leakage|audit-predictions|backfill|backfill-keys|backfill-syntheses|export-graph|readiness|kalshi-bootstrap|kalshi-auth-check|kalshi-sync|kalshi-compare|kalshi-scan|kalshi-loop|kalshi-benchmark|hindcast> [args...]"
            )
        finally:
            if store is not None:
                await store.close()
            if kalshi_ledger is not None:
                await kalshi_ledger.close()
            if temp_dir is not None:
                shutil.rmtree(temp_dir, ignore_errors=True)

    asyncio.run(_run())


def run_test():
    """Run E2E smoke test with real APIs."""
    load_dotenv()
    from nexus.testing.smoke import SmokeTestConfig, run_smoke_test

    topic = "AI/ML Research"
    test_telegram = False
    test_audio = False

    args = sys.argv[2:]
    i = 0
    while i < len(args):
        if args[i] == "--topic" and i + 1 < len(args):
            topic = args[i + 1]
            i += 2
        elif args[i] == "--telegram":
            test_telegram = True
            i += 1
        elif args[i] == "--audio":
            test_audio = True
            i += 1
        else:
            i += 1

    config = SmokeTestConfig(
        topic_name=topic,
        test_telegram=test_telegram,
        test_audio=test_audio,
    )

    print(f"Running smoke test: topic='{topic}'")
    print(f"  telegram={test_telegram}, audio={test_audio}")
    print()

    result = asyncio.run(run_smoke_test(config))

    # Print results
    print(f"{'=' * 50}")
    print(f"Smoke Test Results  ({result.timing_s}s)")
    print(f"{'=' * 50}")
    for check in result.checks:
        status = "PASS" if check.passed else "FAIL"
        print(f"  [{status}] {check.name}: {check.detail}")

    if result.errors:
        print(f"\nErrors:")
        for err in result.errors:
            print(f"  - {err}")

    print(f"\nSummary: events={result.events_found}, "
          f"briefing={result.briefing_chars} chars, "
          f"cost=${result.cost_usd:.4f}")
    print(f"Result: {'PASS' if result.success else 'FAIL'}")

    sys.exit(0 if result.success else 1)


def run_audit_sources():
    """Audit source quality: score each feed's articles against a topic."""
    load_dotenv()
    import yaml
    from nexus.config.loader import load_config
    from nexus.engine.sources.audit import audit_registry
    from nexus.llm.client import LLMClient

    data_dir = Path("data")
    config_path = data_dir / "config.yaml"

    if not config_path.exists():
        print(f"Config not found at {config_path}. Run: python -m nexus setup")
        sys.exit(1)

    if len(sys.argv) < 3:
        print("Usage: python -m nexus audit-sources <topic-slug>")
        sys.exit(1)

    slug = sys.argv[2]
    config = load_config(config_path)

    topic = None
    for t in config.topics:
        topic_slug = t.name.lower().replace(" ", "-").replace("/", "-")
        if topic_slug == slug:
            topic = t
            break
    if not topic:
        print(f"Topic '{slug}' not found. Available:")
        for t in config.topics:
            print(f"  {t.name.lower().replace(' ', '-').replace('/', '-')}")
        sys.exit(1)

    registry_path = data_dir / "sources" / slug / "registry.yaml"
    if not registry_path.exists():
        print(f"No registry at {registry_path}")
        sys.exit(1)

    reg = yaml.safe_load(registry_path.read_text()) or {}
    sources = reg.get("sources", [])
    if not sources:
        print(f"Empty registry at {registry_path}")
        sys.exit(1)

    api_key = os.getenv("GEMINI_API_KEY")
    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("deepseek")
    llm = LLMClient(
        config.models, api_key=api_key, deepseek_api_key=deepseek_api_key,
    )

    print(f"Auditing {len(sources)} sources for '{slug}'...\n")

    results = asyncio.run(audit_registry(llm, sources, topic))

    # Print results table
    print(f"\n{'Source':<30s} {'Score':>5s}  {'Articles':>8s}  {'Verdict'}")
    print("-" * 60)
    for r in sorted(results, key=lambda x: x["mean_score"], reverse=True):
        print(f"{r['source_id']:<30s} {r['mean_score']:>5.1f}  {r['n_articles']:>8d}  {r['verdict'].upper()}")

    # Summary
    keeps = sum(1 for r in results if r["verdict"] == "keep")
    reviews = sum(1 for r in results if r["verdict"] == "review")
    drops = sum(1 for r in results if r["verdict"] == "drop")
    deads = sum(1 for r in results if r["verdict"] == "dead")
    print(f"\nSummary: {keeps} keep, {reviews} review, {drops} drop, {deads} dead")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    if len(sys.argv) < 2:
        print("Usage: python -m nexus <command>\n\n"
              "Commands:\n"
              "  setup      Interactive setup wizard (start here)\n"
              "  run        Start all services (dashboard + scheduler + Telegram)\n"
              "  engine     Run the pipeline once\n"
              "  serve      Start dashboard only\n"
              "  sources    Manage feeds (check | list | build | discover)\n"
              "  evaluate   Judge synthesis quality\n"
              "  projection Projection generate/backfill/evaluation/compare utilities\n"
              "  forecast   Quantified forecast generate/replay/resolve/benchmark utilities\n"
              "  benchmark  Fast benchmark: Suite A on saved fixtures (~3-5 min)\n"
              "  experiment Controlled experiment suites for README claims\n"
              "  test       Run E2E smoke test with real APIs\n"
              "  audit-sources  Score each feed's relevance to a topic\n")
        sys.exit(1)

    command = sys.argv[1]

    if command == "engine":
        run_engine()
    elif command == "run":
        run_all_services()
    elif command == "sources":
        run_sources()
    elif command == "evaluate":
        run_evaluate()
    elif command == "projection":
        run_projection()
    elif command == "forecast":
        run_forecast()
    elif command == "serve":
        run_serve()
    elif command == "setup":
        from nexus.cli.setup import run_setup
        run_setup(Path("data"))
    elif command == "benchmark":
        run_benchmark()
    elif command == "experiment":
        run_experiment()
    elif command == "test":
        run_test()
    elif command == "audit-sources":
        run_audit_sources()
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
