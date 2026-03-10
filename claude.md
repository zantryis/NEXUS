# nexus-claude

Agentic news compiler. Self-hosted, model-agnostic, chat-first. Full spec: design.md.

## User Preferences
- TDD: write tests first, then implement to pass them
- No bloat: minimal code, no over-engineering, no premature abstractions
- Use git: commit at meaningful milestones with descriptive messages
- Session state: read `state.md` at start, update concisely at end
- Keep claude.md under 60 lines
- Time accounting: before any long-running task (live pipeline, bulk API calls), estimate
  wall-clock time and confirm with user. Break into smaller runs if >5 min.
- Lightweight tests: use mocks/fixtures for LLM and network calls. Integration tests that
  hit real APIs must be marked `@pytest.mark.integration` and skipped by default.

## LLM Usage (Development)
- Fast/cheap: `gemini-3-flash-preview` (filtering, scoring, knowledge summaries)
- Smart/slow: `gemini-3.1-pro-preview` (synthesis, agent, dialogue script)
- DeepSeek A/B: `deepseek-chat` (fast), `deepseek-reasoner` (strong) — V3.2, OpenAI-compatible
- All calls go through `src/nexus/llm/` abstraction layer, never directly to SDK
- Keys in `.env`: `GEMINI_API_KEY`, `deepseek` (lowercase)

## Tech Stack
- Python 3.12+, uv for deps, `pyproject.toml`
- Pydantic v2 for config + validation
- feedparser, trafilatura for ingestion
- google-genai SDK, openai SDK (DeepSeek), anthropic SDK
- pytest + pytest-asyncio for testing

## Project Structure
```
src/nexus/
  config/           # Pydantic models, YAML loader
  engine/           # Pipeline: sources/ ingestion/ filtering/ knowledge/ synthesis/ evaluation/
  llm/              # Multi-provider wrapper (Gemini, DeepSeek, Anthropic)
  testing/          # FixtureCapture/Replay, partition_by_date
data/               # Runtime data (gitignored except config)
  config.yaml       # User config with per-topic thresholds
  knowledge/        # Per-topic events + summaries
  sources/          # Per-topic source registries
  artifacts/        # Briefings, syntheses, metrics
tests/              # Mirrors src structure
```

## Key Concepts
- TopicSynthesis (X): intermediate knowledge object with NarrativeThread, convergence/divergence
- Artifacts render FROM X (briefings, audio scripts, etc.)
- CLI: `python -m nexus engine [--capture] [--backtest] [--label X] [--models-override k=v]`

## Current Phase: Phase 1 overhaul complete → backtest + evaluate → Phase 2
Phase 2 (audio pipeline) deferred. Don't touch Phase 2 yet.
