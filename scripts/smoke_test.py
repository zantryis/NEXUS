"""Smoke test — verify each pipeline stage works before deployment."""

import asyncio
import os
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import yaml
from nexus.config.models import NexusConfig


DATA_DIR = Path(__file__).parent.parent / "data"


def step(name):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")


def ok(msg):
    print(f"  [OK] {msg}")


def fail(msg):
    print(f"  [FAIL] {msg}")


def warn(msg):
    print(f"  [WARN] {msg}")


def test_config():
    step("1. Config Loading")
    config_path = DATA_DIR / "config.yaml"
    raw = yaml.safe_load(config_path.read_text())
    config = NexusConfig(**raw)
    ok(f"Loaded config: {config.user.name}, timezone={config.user.timezone}")
    ok(f"Topics: {[t.name for t in config.topics]}")
    ok(f"Style: {config.briefing.style}")
    ok(f"Audio: backend={config.audio.tts_backend}, voices={config.audio.voice_host_a}/{config.audio.voice_host_b}")
    ok(f"Schedule: {config.briefing.schedule} {config.user.timezone}")
    return config


def test_source_registries(config):
    step("2. Source Registries")
    for topic in config.topics:
        slug = topic.name.lower().replace(" ", "-").replace("/", "-")
        reg_path = DATA_DIR / "sources" / slug / "registry.yaml"
        if not reg_path.exists():
            fail(f"{topic.name}: no registry at {reg_path}")
            continue
        raw = yaml.safe_load(reg_path.read_text())
        sources = raw.get("sources", [])
        ok(f"{topic.name}: {len(sources)} sources in registry")


def test_feed_polling(config):
    step("3. Feed Polling (sample 2 feeds per topic)")
    import feedparser
    for topic in config.topics:
        slug = topic.name.lower().replace(" ", "-").replace("/", "-")
        reg_path = DATA_DIR / "sources" / slug / "registry.yaml"
        if not reg_path.exists():
            continue
        raw = yaml.safe_load(reg_path.read_text())
        sources = raw.get("sources", [])[:2]  # Just test first 2
        for src in sources:
            t0 = time.monotonic()
            try:
                feed = feedparser.parse(src["url"])
                n = len(feed.entries)
                elapsed = time.monotonic() - t0
                if n > 0:
                    ok(f"{src['id']}: {n} entries ({elapsed:.1f}s)")
                else:
                    warn(f"{src['id']}: 0 entries ({elapsed:.1f}s) — feed may be empty or blocked")
            except Exception as e:
                fail(f"{src['id']}: {e}")


def test_llm_client(config):
    step("4. LLM Client")
    from nexus.llm.client import LLMClient
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        fail("GEMINI_API_KEY not set")
        return None
    ok(f"GEMINI_API_KEY present ({len(api_key)} chars)")

    llm = LLMClient(config.models, api_key=api_key)
    ok(f"LLMClient initialized")
    ok(f"Filtering model: {llm.resolve_model('filtering')}")
    ok(f"Synthesis model: {llm.resolve_model('synthesis')}")
    return llm


async def test_llm_call(llm):
    step("5. LLM Completion (real API call)")
    if not llm:
        warn("Skipping — no LLM client")
        return
    t0 = time.monotonic()
    try:
        response = await llm.complete(
            config_key="filtering",
            system_prompt="You are a test assistant. Respond with valid JSON only.",
            user_prompt='Say hello. Respond: {"message": "hello"}',
            json_response=True,
        )
        elapsed = time.monotonic() - t0
        ok(f"Got response in {elapsed:.1f}s: {response[:100]}")
    except Exception as e:
        fail(f"LLM call failed: {e}")


async def test_knowledge_store():
    step("6. Knowledge Store")
    from nexus.engine.knowledge.store import KnowledgeStore
    db_path = DATA_DIR / "knowledge.db"
    store = KnowledgeStore(db_path)
    await store.initialize()

    # Check schema version
    async with store._db.execute("PRAGMA user_version") as cursor:
        row = await cursor.fetchone()
        version = row[0]
    ok(f"Schema version: {version}")

    # Check table count
    async with store._db.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ) as cursor:
        tables = [row[0] async for row in cursor]
    ok(f"Tables ({len(tables)}): {', '.join(tables)}")

    await store.close()


async def test_tts_backend(config):
    step("7. TTS Backend Init")
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        warn("Skipping — no GEMINI_API_KEY")
        return
    try:
        from nexus.engine.audio.tts import get_tts_backend
        backend = get_tts_backend(config.audio, gemini_api_key=api_key)
        ok(f"TTS backend: {type(backend).__name__}")
        ok(f"Model: {config.audio.tts_model}")
        ok(f"Voices: {config.audio.voice_host_a} (female/Host A), {config.audio.voice_host_b} (male/Host B)")
    except Exception as e:
        fail(f"TTS init failed: {e}")


def test_telegram_token():
    step("8. Telegram Bot Token")
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if token:
        ok(f"TELEGRAM_BOT_TOKEN present ({len(token)} chars)")
    else:
        warn("TELEGRAM_BOT_TOKEN not set — Telegram delivery disabled")


async def main():
    print("\n" + "=" * 60)
    print("  NEXUS-CLAUDE SMOKE TEST")
    print("=" * 60)

    # Load env
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                os.environ.setdefault(key.strip(), val.strip())

    config = test_config()
    test_source_registries(config)
    test_feed_polling(config)
    llm = test_llm_client(config)
    await test_llm_call(llm)
    await test_knowledge_store()
    await test_tts_backend(config)
    test_telegram_token()

    print(f"\n{'='*60}")
    print("  SMOKE TEST COMPLETE")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    asyncio.run(main())
