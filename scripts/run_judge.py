"""Run dual LLM-as-judge evaluation: Gemini judges + DeepSeek judges, averaged.

Picks representative synthesis pairs from backtest data and scores them.
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nexus.config.models import ModelsConfig
from nexus.engine.evaluation.judge import judge_synthesis, JUDGE_SYSTEM_PROMPT, _format_synthesis_for_judge
from nexus.engine.synthesis.knowledge import TopicSynthesis
from nexus.llm.client import LLMClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Representative pairs: (date, topic_slug)
PAIRS = [
    ("2026-03-09", "iran-us-relations"),
    ("2026-03-09", "ai-ml-research"),
    ("2026-03-09", "global-energy-transition"),
    ("2026-03-08", "iran-us-relations"),
    ("2026-03-06", "ai-ml-research"),
    ("2026-03-05", "global-energy-transition"),
]

BACKTEST_DIR = Path("data/backtest")


def load_synthesis(provider: str, date: str, topic: str) -> TopicSynthesis | None:
    path = BACKTEST_DIR / provider / "artifacts" / "syntheses" / date / f"{topic}.yaml"
    if not path.exists():
        return None
    raw = yaml.safe_load(path.read_text())
    return TopicSynthesis(**raw)


async def judge_with_provider(
    llm: LLMClient, synthesis: TopicSynthesis, provider_label: str
) -> dict:
    """Judge a synthesis, returning scores + judge label."""
    scores = await judge_synthesis(llm, synthesis)
    scores["judged_by"] = provider_label
    return scores


async def main():
    load_dotenv()

    gemini_key = os.getenv("GEMINI_API_KEY")
    deepseek_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("deepseek")

    if not gemini_key:
        print("GEMINI_API_KEY required"); sys.exit(1)
    if not deepseek_key:
        print("DEEPSEEK_API_KEY required"); sys.exit(1)

    # Gemini judge client
    gemini_config = ModelsConfig()
    gemini_config.agent = "gemini-3.1-pro-preview"  # Use smart model for judging
    gemini_llm = LLMClient(gemini_config, api_key=gemini_key)

    # DeepSeek judge client
    ds_config = ModelsConfig()
    ds_config.agent = "deepseek-reasoner"  # Use reasoning model for judging
    ds_llm = LLMClient(ds_config, deepseek_api_key=deepseek_key)

    results = []

    for date, topic in PAIRS:
        gemini_syn = load_synthesis("gemini", date, topic)
        deepseek_syn = load_synthesis("deepseek", date, topic)

        if not gemini_syn or not deepseek_syn:
            logger.warning(f"Skipping {date}/{topic} — missing synthesis")
            continue

        logger.info(f"Judging {date}/{topic}...")

        # Each provider's synthesis judged by BOTH judges
        try:
            gemini_by_gemini, gemini_by_deepseek, deepseek_by_gemini, deepseek_by_deepseek = (
                await asyncio.gather(
                    judge_with_provider(gemini_llm, gemini_syn, "gemini"),
                    judge_with_provider(ds_llm, gemini_syn, "deepseek"),
                    judge_with_provider(gemini_llm, deepseek_syn, "gemini"),
                    judge_with_provider(ds_llm, deepseek_syn, "deepseek"),
                )
            )
        except Exception as e:
            logger.error(f"Failed {date}/{topic}: {e}")
            continue

        dims = ["completeness", "source_balance", "convergence_accuracy",
                "divergence_detection", "entity_coverage"]

        def avg_scores(scores_a: dict, scores_b: dict) -> dict:
            averaged = {}
            for d in dims:
                a = scores_a.get(d, 5)
                b = scores_b.get(d, 5)
                averaged[d] = round((a + b) / 2, 1)
            averaged["overall"] = round(sum(averaged[d] for d in dims) / len(dims), 1)
            return averaged

        gemini_avg = avg_scores(gemini_by_gemini, gemini_by_deepseek)
        deepseek_avg = avg_scores(deepseek_by_gemini, deepseek_by_deepseek)

        result = {
            "date": date,
            "topic": topic,
            "gemini_synthesis": {
                "averaged": gemini_avg,
                "by_gemini": {d: gemini_by_gemini.get(d, "?") for d in dims},
                "by_deepseek": {d: gemini_by_deepseek.get(d, "?") for d in dims},
            },
            "deepseek_synthesis": {
                "averaged": deepseek_avg,
                "by_gemini": {d: deepseek_by_gemini.get(d, "?") for d in dims},
                "by_deepseek": {d: deepseek_by_deepseek.get(d, "?") for d in dims},
            },
            "winner": "gemini" if gemini_avg["overall"] > deepseek_avg["overall"]
                      else "deepseek" if deepseek_avg["overall"] > gemini_avg["overall"]
                      else "tie",
            "strengths_gemini": gemini_by_gemini.get("strengths", []),
            "weaknesses_gemini": gemini_by_gemini.get("weaknesses", []),
            "strengths_deepseek": deepseek_by_gemini.get("strengths", []),
            "weaknesses_deepseek": deepseek_by_gemini.get("weaknesses", []),
        }
        results.append(result)

        # Print progress
        print(f"\n{'='*60}")
        print(f"{date} / {topic}")
        print(f"  Gemini synthesis:  {gemini_avg['overall']:.1f} overall")
        print(f"    (Gemini judge: {gemini_by_gemini.get('overall', '?')} | DeepSeek judge: {gemini_by_deepseek.get('overall', '?')})")
        print(f"  DeepSeek synthesis: {deepseek_avg['overall']:.1f} overall")
        print(f"    (Gemini judge: {deepseek_by_gemini.get('overall', '?')} | DeepSeek judge: {deepseek_by_deepseek.get('overall', '?')})")
        print(f"  Winner: {result['winner']}")

    # Summary
    if results:
        print(f"\n{'='*60}")
        print("OVERALL SUMMARY")
        print(f"{'='*60}")

        gemini_wins = sum(1 for r in results if r["winner"] == "gemini")
        deepseek_wins = sum(1 for r in results if r["winner"] == "deepseek")
        ties = sum(1 for r in results if r["winner"] == "tie")

        dims = ["completeness", "source_balance", "convergence_accuracy",
                "divergence_detection", "entity_coverage", "overall"]

        print(f"\nWins: Gemini {gemini_wins} | DeepSeek {deepseek_wins} | Ties {ties}")

        print(f"\nAverage scores across all pairs:")
        for d in dims:
            g = sum(r["gemini_synthesis"]["averaged"].get(d, 0) for r in results) / len(results)
            ds = sum(r["deepseek_synthesis"]["averaged"].get(d, 0) for r in results) / len(results)
            marker = "<" if g < ds else ">" if g > ds else "="
            print(f"  {d:25s}  Gemini {g:.1f} {marker} DeepSeek {ds:.1f}")

        # Save full results
        out_path = Path("data/evaluation/judge_results.yaml")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(yaml.dump(results, default_flow_style=False, sort_keys=False))
        print(f"\nFull results saved to {out_path}")

        # Cost accounting
        print(f"\n{'='*60}")
        print("COST ACCOUNTING")
        print(f"{'='*60}")
        for label, client in [("Gemini judge", gemini_llm), ("DeepSeek judge", ds_llm)]:
            usage = client.usage.summary()
            print(f"\n{label}:")
            print(f"  Calls: {usage['total_calls']}")
            print(f"  Input tokens: {usage['total_input_tokens']:,}")
            print(f"  Output tokens: {usage['total_output_tokens']:,}")
            print(f"  LLM time: {usage['total_elapsed_s']:.1f}s")


if __name__ == "__main__":
    asyncio.run(main())
