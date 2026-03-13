#!/usr/bin/env python3
"""Multi-config smoke test orchestrator.

Generates per-variant configs, optionally pre-discovers sources, and
launches Docker containers for side-by-side dashboard comparison.

Usage:
    python scripts/multi_smoke.py                  # Full run
    python scripts/multi_smoke.py --skip-discovery  # Skip source discovery
    python scripts/multi_smoke.py --no-telegram     # Skip Telegram setup
    python scripts/multi_smoke.py --down            # Tear down containers + data
"""

import argparse
import asyncio
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nexus.config.writer import write_config

# ── Test matrix ──────────────────────────────────────────────────────────────

VARIANTS = [
    {
        "id": "a",
        "topic": "Space Exploration",
        "subtopics": ["Mars missions", "commercial spaceflight", "satellite technology"],
        "preset": "balanced",
        "style": "analytical",
        "tts_backend": "gemini",
        "tts_model": "gemini-2.5-flash-preview-tts",
        "voice_a": "Aoede",
        "voice_b": "Charon",
        "port": 8081,
        "telegram": False,
    },
    {
        "id": "b",
        "topic": "Quantum Computing",
        "subtopics": ["quantum hardware", "quantum algorithms", "post-quantum cryptography"],
        "preset": "quality",
        "style": "conversational",
        "tts_backend": "elevenlabs",
        "tts_model": "eleven_multilingual_v2",
        "voice_a": "Sarah",
        "voice_b": "Charlie",
        "port": 8082,
        "telegram": False,
    },
    {
        "id": "c",
        "topic": "Climate Adaptation",
        "subtopics": ["extreme weather", "renewable policy", "carbon capture"],
        "preset": "cheap",
        "style": "analytical",
        "tts_backend": "gemini",
        "tts_model": "gemini-2.5-flash-preview-tts",
        "voice_a": "Kore",
        "voice_b": "Puck",
        "port": 8083,
        "telegram": False,
    },
    {
        "id": "d",
        "topic": "Cybersecurity Threats",
        "subtopics": ["ransomware", "state-sponsored hacking", "zero-day exploits"],
        "preset": "balanced",
        "style": "conversational",
        "tts_backend": "elevenlabs",
        "tts_model": "eleven_multilingual_v2",
        "voice_a": "Laura",
        "voice_b": "George",
        "port": 8084,
        "telegram": True,
    },
]


def build_variant_config(variant: dict, chat_id: int | None = None) -> dict:
    """Build a NexusConfig-compatible dict for one variant."""
    audio_cfg = {
        "enabled": True,
        "tts_backend": variant["tts_backend"],
        "tts_model": variant["tts_model"],
        "voice_host_a": variant["voice_a"],
        "voice_host_b": variant["voice_b"],
    }
    if variant["tts_backend"] == "elevenlabs":
        audio_cfg.update({
            "elevenlabs_stability": 0.7,
            "elevenlabs_similarity_boost": 0.8,
            "elevenlabs_style": 0.35,
            "elevenlabs_speaker_boost": True,
        })

    telegram_cfg = {"enabled": variant["telegram"]}
    if chat_id and variant["telegram"]:
        telegram_cfg["chat_id"] = chat_id

    return {
        "preset": variant["preset"],
        "user": {
            "name": f"SmokeTest-{variant['id'].upper()}",
            "timezone": "America/Denver",
            "output_language": "en",
        },
        "briefing": {
            "schedule": "06:00",
            "format": "two-host-dialogue",
            "style": variant["style"],
            "depth": "detailed",
        },
        "topics": [
            {
                "name": variant["topic"],
                "priority": "high",
                "subtopics": variant["subtopics"],
                "scope": "narrow",
                "max_events": 5,
                "source_languages": ["en"],
            },
        ],
        "audio": audio_cfg,
        "telegram": telegram_cfg,
        "breaking_news": {"enabled": False},
        "budget": {
            "daily_limit_usd": 0.50,
            "warning_threshold_usd": 0.25,
        },
        "sources": {
            "discover_new_sources": True,
        },
    }


def generate_configs(chat_id: int | None = None) -> None:
    """Write config.yaml and copy global_registry for each variant."""
    global_reg = PROJECT_ROOT / "data" / "sources" / "global_registry.yaml"

    for v in VARIANTS:
        data_dir = PROJECT_ROOT / f"data-test-{v['id']}"
        config_dict = build_variant_config(v, chat_id=chat_id)
        write_config(data_dir, config_dict)
        print(f"  [{v['id'].upper()}] config.yaml → {data_dir / 'config.yaml'}")

        # Copy global registry so discovery can match curated sources
        if global_reg.exists():
            dest = data_dir / "sources" / "global_registry.yaml"
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(global_reg, dest)


async def pre_discover_sources() -> None:
    """Run auto-discovery for each variant's topic."""
    from nexus.config.models import ModelsConfig, BudgetConfig
    from nexus.config.presets import apply_preset
    from nexus.engine.sources.discovery import discover_sources
    from nexus.llm.client import LLMClient

    import yaml

    api_key = os.getenv("GEMINI_API_KEY")
    deepseek_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("deepseek")

    for v in VARIANTS:
        data_dir = PROJECT_ROOT / f"data-test-{v['id']}"
        slug = v["topic"].lower().replace(" ", "-").replace("/", "-")
        reg_path = data_dir / "sources" / slug / "registry.yaml"

        if reg_path.exists():
            print(f"  [{v['id'].upper()}] {v['topic']}: registry already exists, skipping")
            continue

        print(f"  [{v['id'].upper()}] Discovering sources for '{v['topic']}'...")
        models = apply_preset(v["preset"])
        llm = LLMClient(
            models, api_key=api_key, deepseek_api_key=deepseek_key,
            budget_config=BudgetConfig(daily_limit_usd=0.50),
        )

        try:
            result = await discover_sources(
                llm, v["topic"], subtopics=v["subtopics"],
                max_feeds=8, data_dir=data_dir,
            )
            if result.feeds:
                reg_path.parent.mkdir(parents=True, exist_ok=True)
                reg_path.write_text(
                    yaml.dump({"sources": result.feeds}, default_flow_style=False)
                )
                print(f"  [{v['id'].upper()}] Found {len(result.feeds)} sources")
            else:
                print(f"  [{v['id'].upper()}] WARNING: No sources found")
        except Exception as e:
            print(f"  [{v['id'].upper()}] Discovery failed: {e}")


async def setup_telegram() -> int | None:
    """Validate test token and capture chat_id interactively."""
    from nexus.agent.telegram_utils import validate_token, poll_for_chat_id

    token = os.getenv("TELEGRAM_BOT_TOKEN_TEST")
    if not token:
        print("  TELEGRAM_BOT_TOKEN_TEST not set in .env — skipping Telegram")
        return None

    print("  Validating test bot token...")
    bot_info = await validate_token(token)
    if not bot_info:
        print("  ERROR: Test bot token is invalid")
        return None

    username = bot_info.get("username", "unknown")
    print(f"  Bot: @{username}")
    print(f"  Send /start to @{username} in Telegram now...")

    for attempt in range(6):
        chat_id = await poll_for_chat_id(token, timeout=10.0)
        if chat_id:
            print(f"  Captured chat_id: {chat_id}")
            return chat_id
        print(f"  Waiting for /start... ({(attempt + 1) * 10}s)")

    print("  Timed out waiting for /start — Telegram delivery will be skipped")
    return None


def start_containers() -> None:
    """Build and start Docker containers with staggered starts."""
    import time

    compose_file = PROJECT_ROOT / "docker-compose.test.yml"
    print(f"\n  Building Docker image...")
    subprocess.run(
        ["docker", "compose", "-f", str(compose_file), "build"],
        cwd=str(PROJECT_ROOT),
        check=True,
    )

    # Stagger starts by 15s to avoid Gemini rate limits
    for v in VARIANTS:
        service = f"smoke-{v['id']}"
        print(f"  Starting {service} (port {v['port']})...")
        subprocess.run(
            ["docker", "compose", "-f", str(compose_file), "up", "-d", service],
            cwd=str(PROJECT_ROOT),
            check=True,
        )
        if v != VARIANTS[-1]:
            print(f"  Waiting 15s before next container (rate limit mitigation)...")
            time.sleep(15)


def tear_down() -> None:
    """Stop containers and remove test data directories."""
    compose_file = PROJECT_ROOT / "docker-compose.test.yml"
    print("Tearing down smoke test containers...")
    subprocess.run(
        ["docker", "compose", "-f", str(compose_file), "down", "--remove-orphans"],
        cwd=str(PROJECT_ROOT),
    )
    for v in VARIANTS:
        data_dir = PROJECT_ROOT / f"data-test-{v['id']}"
        if data_dir.exists():
            shutil.rmtree(data_dir)
            print(f"  Removed {data_dir}")
    print("Done.")


def print_summary() -> None:
    """Print the variant summary table."""
    print("\n" + "=" * 72)
    print("  SMOKE TEST VARIANTS")
    print("=" * 72)
    fmt = "  {:<4} {:<24} {:<10} {:<14} {:<12} {}"
    print(fmt.format("ID", "Topic", "Preset", "Style", "TTS", "URL"))
    print("  " + "-" * 68)
    for v in VARIANTS:
        tg = " +TG" if v["telegram"] else ""
        print(fmt.format(
            v["id"].upper(),
            v["topic"],
            v["preset"],
            v["style"],
            v["tts_backend"] + tg,
            f"http://localhost:{v['port']}",
        ))
    print("=" * 72)
    print("\n  Monitor logs: docker compose -f docker-compose.test.yml logs -f")
    print("  Tear down:   python scripts/multi_smoke.py --down")


def main():
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")

    parser = argparse.ArgumentParser(description="Multi-config smoke test orchestrator")
    parser.add_argument("--skip-discovery", action="store_true",
                        help="Skip source auto-discovery (assume registries exist)")
    parser.add_argument("--no-telegram", action="store_true",
                        help="Skip Telegram setup for variant D")
    parser.add_argument("--down", action="store_true",
                        help="Tear down containers and clean test data")
    parser.add_argument("--configs-only", action="store_true",
                        help="Generate configs only, don't start containers")
    args = parser.parse_args()

    if args.down:
        tear_down()
        return

    # Step 1: Telegram setup (interactive)
    chat_id = None
    if not args.no_telegram:
        print("\n[1/4] Telegram setup")
        chat_id = asyncio.run(setup_telegram())
    else:
        print("\n[1/4] Telegram setup — skipped")

    # Step 2: Generate configs
    print("\n[2/4] Generating variant configs")
    generate_configs(chat_id=chat_id)

    # Step 3: Pre-discover sources
    if not args.skip_discovery:
        print("\n[3/4] Pre-discovering sources (this may take 1-2 min per topic)")
        asyncio.run(pre_discover_sources())
    else:
        print("\n[3/4] Source discovery — skipped")

    if args.configs_only:
        print("\n[4/4] Configs generated — skipping container start (--configs-only)")
        print_summary()
        return

    # Step 4: Start Docker containers
    print("\n[4/4] Starting Docker containers")
    start_containers()

    print_summary()


if __name__ == "__main__":
    main()
