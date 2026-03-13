# nexus-claude

Agentic news intelligence compiler. Self-hosted, model-agnostic, always-on.

## Dev Preferences
- TDD: write tests first, then implement to pass them
- No bloat: minimal code, no over-engineering, no premature abstractions
- Use git: commit at meaningful milestones with descriptive messages
- Time accounting: before long-running tasks (live pipeline, bulk API calls), estimate
  wall-clock time and confirm with user. Break into smaller runs if >5 min.
- Lightweight tests: use mocks/fixtures for LLM and network calls. Integration tests that
  hit real APIs must be marked `@pytest.mark.integration` and skipped by default.

## LLM Usage
- Fast/cheap: `gemini-3-flash-preview` (filtering, scoring, knowledge summaries, breaking news)
- Smart/slow: `gemini-3.1-pro-preview` (synthesis, agent Q&A, dialogue script)
- DeepSeek: `deepseek-chat` (fast), `deepseek-reasoner` (strong) — OpenAI-compatible
- TTS: `gemini-2.5-flash-preview-tts` (Gemini native TTS)
- All LLM calls go through `src/nexus/llm/client.py`, never directly to SDK
- Keys in `.env`: `GEMINI_API_KEY`, `DEEPSEEK_API_KEY`, `TELEGRAM_BOT_TOKEN`

## Project Structure
```
src/nexus/
  config/           # Pydantic models, YAML loader
  engine/           # Pipeline: sources/ ingestion/ filtering/ knowledge/ synthesis/ audio/ evaluation/
  agent/            # Telegram bot, Q&A, breaking news, delivery, feedback
  scheduler/        # APScheduler job definitions
  web/              # FastAPI dashboard + podcast RSS
  llm/              # Multi-provider async LLM client
  testing/          # Fixture capture/replay for backtesting
  runner.py         # Unified runner (dashboard + scheduler + Telegram bot)
data/               # Runtime data (gitignored except source registries)
  sources/          # Per-topic source registries (committed)
  config.yaml       # User config (gitignored, see data/config.example.yaml)
tests/              # Mirrors src structure, 640 tests (unit + e2e)
```

## Key Concepts
- TopicSynthesis (X): intermediate knowledge object with NarrativeThread, convergence/divergence
- Artifacts render FROM X (briefings, audio scripts, etc.)
- CLI: `python -m nexus run` (all services) | `engine` (pipeline only) | `serve` (dashboard only)
- Pipeline parameters (thresholds, scoring rubrics, limits): `docs/pipeline-parameters.md`
- Cost accounting: `LLMClient` persists usage to SQLite via `set_store()`, budget guard syncs on startup
