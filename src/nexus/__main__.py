"""CLI entry point: python -m nexus <command>."""

import asyncio
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv


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
        print(f"Config not found at {config_path}. Copy data/config.example.yaml to get started.")
        sys.exit(1)

    api_key = os.getenv("GEMINI_API_KEY")
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("deepseek")

    if not api_key and not anthropic_api_key and not deepseek_api_key:
        print("Set GEMINI_API_KEY, ANTHROPIC_API_KEY, or DEEPSEEK_API_KEY in .env")
        sys.exit(1)

    config = load_config(config_path)

    # Apply model overrides from CLI
    overrides = _parse_models_override(sys.argv)
    if overrides:
        for key, value in overrides.items():
            if hasattr(config.models, key):
                setattr(config.models, key, value)
                print(f"  Model override: {key}={value}")
            else:
                print(f"  Warning: unknown model key '{key}'")

    llm = LLMClient(
        config.models, api_key=api_key,
        anthropic_api_key=anthropic_api_key,
        deepseek_api_key=deepseek_api_key,
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
        asyncio.run(run_backtest(config, llm, data_dir, label=label))
    else:
        briefing_path = asyncio.run(run_pipeline(config, llm, data_dir, capture=do_capture))
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

    elif subcommand == "list":
        for s in sources:
            affil = f"{s.affiliation}/{s.country}" if s.country else s.affiliation
            print(f"  [{s.language}] {s.tier} {affil:12s} {s.id:25s} {','.join(s.tags)}")

    else:
        print(f"Unknown subcommand: {subcommand}. Use: check, build, list")
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


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    if len(sys.argv) < 2:
        print("Usage: python -m nexus <engine|sources|evaluate|setup>")
        sys.exit(1)

    command = sys.argv[1]

    if command == "engine":
        run_engine()
    elif command == "sources":
        run_sources()
    elif command == "evaluate":
        run_evaluate()
    elif command == "setup":
        print("Setup wizard not yet implemented.")
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
