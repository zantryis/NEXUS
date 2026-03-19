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
- LiteLLM: `litellm/` prefix routes through OpenAI-compatible proxy (cloud VM models)
- TTS: `gemini-2.5-flash-preview-tts` (Gemini native TTS)
- All LLM calls go through `src/nexus/llm/client.py`, never directly to SDK
- Keys in `.env`: `GEMINI_API_KEY`, `DEEPSEEK_API_KEY`, `TELEGRAM_BOT_TOKEN`
- Cloud env vars: `LITELLM_BASE_URL`, `LITELLM_API_KEY`

## Project Structure
```
src/nexus/
  config/           # Pydantic models, YAML loader
  engine/           # Pipeline: sources/ ingestion/ filtering/ knowledge/ synthesis/ audio/ evaluation/ projection/
  utils/            # Feed health monitoring
  agent/            # Telegram bot, Q&A, breaking news, delivery, feedback
  scheduler/        # APScheduler job definitions
  web/              # FastAPI dashboard + podcast RSS
  llm/              # Multi-provider async LLM client
  testing/          # Fixture capture/replay for backtesting
  runner.py         # Unified runner (dashboard + scheduler + Telegram bot)
data/               # Runtime data (gitignored except source registries)
  sources/          # Per-topic source registries (committed)
  config.yaml       # User config (gitignored, see data/config.example.yaml)
  cli/              # Setup wizard, demo seeder
tests/              # Mirrors src structure, 1,520 tests (unit + 31 E2E)
```

## Key Concepts
- TopicSynthesis (X): intermediate knowledge object with NarrativeThread, convergence/divergence
- Artifacts render FROM X (briefings, audio scripts, etc.)
- CLI: `python -m nexus run` | `engine` | `serve` | `forecast` | `projection` | `benchmark`
       | `sources` | `evaluate` | `experiment` | `test` | `audit-sources` | `enrich-entities`
       | `demo seed [--from-scratch]` | `demo serve`
- Experiment CLI: `python -m nexus experiment --suite A,G --export-fixtures PATH --env cloud`
- Prediction pipeline: 6 engines (actor/structural/graphrag/debate/perspective/naked),
  calibrated with gamma=0.8, optional Kalshi market benchmarking
- Pipeline parameters (thresholds, scoring rubrics, limits): `docs/pipeline-parameters.md`
- Cost accounting: `LLMClient` persists usage to SQLite via `set_store()`, budget guard syncs on startup
- Cross-env comparison: fixture export/import, re-judge mode, `compare.py` quality reports

## Documentation Maintenance

When code changes affect any of the following, update the corresponding docs:

| What Changed | Update These Files |
|-------------|-------------------|
| New/removed CLI command | CLAUDE.md (CLI line), README.md (CLI Commands section) |
| Schema migration (new table/column) | ARCHITECTURE.md (schema version, table count, table DDL) |
| New web route | ARCHITECTURE.md (Web Dashboard Routes table) |
| New/changed pipeline stage | `docs/pipeline.html` (node data), `docs/pipeline-parameters.md` (if tunable) |
| Test count crosses a milestone | README.md, ARCHITECTURE.md, CLAUDE.md |
| New Python module | ARCHITECTURE.md (Module Map), CLAUDE.md (Project Structure) |
| Config model changes | ARCHITECTURE.md (Configuration section) |
| New LLM model preset | ARCHITECTURE.md (Model Presets table), SYSTEM_SPEC.md (AI Provider table) |

Verification commands:
```bash
grep -r "def test_" tests/ | wc -l                          # Test count
find src/nexus -name "*.py" | wc -l                         # Source file count
grep "CURRENT_VERSION" src/nexus/engine/knowledge/schema.py  # Schema version
```
