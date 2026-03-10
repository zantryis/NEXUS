"""CLI entry point: python -m nexus <command>."""

import asyncio
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv


def run_engine():
    """Run the daily engine pipeline."""
    load_dotenv()
    from nexus.config.loader import load_config
    from nexus.config.models import ModelsConfig
    from nexus.engine.pipeline import run_pipeline
    from nexus.llm.client import LLMClient

    data_dir = Path("data")
    config_path = data_dir / "config.yaml"

    if not config_path.exists():
        print(f"Config not found at {config_path}. Copy data/config.example.yaml to get started.")
        sys.exit(1)

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY not set. Add it to .env")
        sys.exit(1)

    config = load_config(config_path)
    llm = LLMClient(config.models, api_key=api_key)

    briefing_path = asyncio.run(run_pipeline(config, llm, data_dir))
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


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    if len(sys.argv) < 2:
        print("Usage: python -m nexus <engine|sources|setup>")
        sys.exit(1)

    command = sys.argv[1]

    if command == "engine":
        run_engine()
    elif command == "sources":
        run_sources()
    elif command == "setup":
        print("Setup wizard not yet implemented.")
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
