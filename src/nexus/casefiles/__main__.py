"""Dev entrypoint for building casefiles from disk."""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

from nexus.casefiles.builder import build_casefile
from nexus.casefiles.storage import case_dir
from nexus.config.loader import load_config
from nexus.engine.knowledge.store import KnowledgeStore
from nexus.llm.client import LLMClient
from nexus.utils.runtime_env import load_runtime_env, runtime_env_path


def _build_llm(data_dir: Path):
    config = load_config(data_dir / "config.yaml")
    llm = LLMClient(
        config.models,
        api_key=os.getenv("GEMINI_API_KEY"),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
        deepseek_api_key=os.getenv("DEEPSEEK_API_KEY") or os.getenv("deepseek"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL"),
        budget_config=config.budget,
        litellm_base_url=os.getenv("LITELLM_BASE_URL") or os.getenv("LITELLM_PROXY_URL"),
        litellm_api_key=os.getenv("LITELLM_API_KEY") or os.getenv("LITELLM_PROXY_API_KEY"),
    )
    return config, llm


async def _run(slug: str, data_dir: Path) -> None:
    load_runtime_env(runtime_env_path(data_dir))
    _, llm = _build_llm(data_dir)
    store = KnowledgeStore(data_dir / "knowledge.db")
    await store.initialize()

    def _progress(message: str) -> None:
        print(f"[casefile:{slug}] {message}")

    try:
        bundle = await build_casefile(case_dir(data_dir, slug), llm=llm, store=store, progress=_progress)
        verdict = "presentable" if bundle.review.presentable else "draft"
        print(f"[casefile:{slug}] complete ({verdict})")
        for blocker in bundle.review.blocker_reasons:
            print(f"  - {blocker}")
    finally:
        await store.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a Nexus casefile bundle")
    parser.add_argument("slug", help="Case slug under data/casefiles/")
    parser.add_argument("--data-dir", default="data", help="Nexus data directory")
    args = parser.parse_args()
    asyncio.run(_run(args.slug, Path(args.data_dir)))


if __name__ == "__main__":
    main()
