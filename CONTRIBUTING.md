# Contributing to Nexus

Thanks for your interest in contributing! Here's how to get started.

## Development Setup

```bash
git clone https://github.com/Tyan3001/NEXUS.git
cd NEXUS
python -m venv .venv
source .venv/bin/activate
pip install -e ".[all,dev]"
cp .env.example .env        # Add your API keys
cp data/config.example.yaml data/config.yaml
```

## Workflow

We follow **TDD** (test-driven development):

1. Write tests first in `tests/` (mirrors `src/nexus/` structure)
2. Implement the feature to pass them
3. Run the full suite before submitting

```bash
# Run all unit tests
.venv/bin/pytest

# Run a specific test file
.venv/bin/pytest tests/web/test_settings.py -v

# Skip integration tests (require live API keys)
.venv/bin/pytest -m "not integration"

# Run E2E smoke tests (requires API keys in .env)
.venv/bin/pytest -m e2e -v

# CLI smoke test (runs a minimal pipeline end-to-end)
python -m nexus test
```

## Test Markers

Tests that require external services use pytest markers so they're skipped by default:

| Marker | When to use | What it needs |
|--------|------------|---------------|
| `@pytest.mark.integration` | Hits a live LLM API | API keys in `.env` |
| `@pytest.mark.e2e` | Full pipeline smoke test | API keys + network |

Unit tests should mock all LLM and network calls using fixtures. See `tests/conftest.py` for shared fixtures and `src/nexus/testing/` for the fixture capture/replay framework.

## Fixture Capture & Replay

Nexus includes a fixture system for deterministic backtesting without live API calls:

```bash
# Capture fixtures from a live pipeline run
python -m nexus experiment --suite A,G --export-fixtures fixtures/

# Replay captured fixtures (no API keys needed)
python -m nexus experiment --suite A,G --import-fixtures fixtures/

# Compare results across environments
python -m nexus experiment --suite A,G --compare fixtures/cloud fixtures/local
```

Fixture files are JSON snapshots of LLM responses. When adding a new pipeline stage that calls an LLM, add fixture capture hooks in `src/nexus/testing/` so the stage can be replayed offline.

## Code Style

- Keep it minimal — no over-engineering, no premature abstractions
- All LLM calls go through `src/nexus/llm/client.py`, never directly to SDKs
- Use `logging` for observability, not `print()`
- Tests that hit real APIs must be marked `@pytest.mark.integration` or `@pytest.mark.e2e`

## Pull Requests

1. Create a feature branch from `main`
2. Write descriptive commit messages (focus on *why*, not *what*)
3. Ensure all tests pass
4. Keep PRs focused — one feature or fix per PR

## Project Structure

See [ARCHITECTURE.md](ARCHITECTURE.md) for a detailed system overview.

```
src/nexus/
  config/     # Pydantic models, YAML loader, presets
  engine/     # Pipeline: sources, filtering, knowledge, synthesis, audio
  agent/      # Telegram bot, Q&A, breaking news
  scheduler/  # APScheduler job definitions
  web/        # FastAPI dashboard + setup wizard
  llm/        # Multi-provider async LLM client
  testing/    # Fixture capture/replay for backtesting
tests/        # Mirrors src structure (unit + e2e)
```

## Questions?

Open an issue on GitHub — we're happy to help.
