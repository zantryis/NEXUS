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
- Fast/cheap: `gemini-3-flash-preview` (filtering, scoring, relevance)
- Smart/slow: `gemini-3.1-pro-preview` (synthesis, knowledge summaries)
- All calls go through `src/nexus/llm/` abstraction layer, never directly to SDK
- Keys: `GEMINI_API_KEY` in `.env`

## Tech Stack
- Python 3.11+, uv for deps, `pyproject.toml`
- Pydantic v2 for config + validation
- feedparser, trafilatura for ingestion
- google-genai SDK (new unified API)
- pytest + pytest-asyncio for testing

## Project Structure
```
src/nexus/          # main package
  config/           # Pydantic models, YAML loader
  engine/           # Pipeline: sources/ ingestion/ filtering/ knowledge/ synthesis/
  llm/              # Provider-agnostic wrapper (Gemini now, extensible later)
  utils/            # Logging, date helpers
data/               # Runtime data (gitignored except examples)
  config.yaml       # User config
  knowledge/        # Per-topic events + summaries
  sources/          # Per-topic source registries
  artifacts/        # Generated briefings
tests/              # Mirrors src structure
```

## Current Phase: 1 complete → 2 next
Phase 1 done: config, LLM layer, polling, ingestion, batch filtering, knowledge (events), synthesis
Next: knowledge compression (weekly/monthly rollups), scheduling, source health
Deferred: audio, Telegram agent, PWA, breaking news, source discovery
