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
# Run all tests
.venv/bin/pytest

# Run a specific test file
.venv/bin/pytest tests/web/test_settings.py -v

# Skip integration tests (require live API keys)
.venv/bin/pytest -m "not integration"
```

## Code Style

- Keep it minimal — no over-engineering, no premature abstractions
- All LLM calls go through `src/nexus/llm/client.py`, never directly to SDKs
- Use `logging` for observability, not `print()`
- Tests that hit real APIs must be marked `@pytest.mark.integration`

## Pull Requests

1. Create a feature branch from `main`
2. Write descriptive commit messages (focus on *why*, not *what*)
3. Ensure all tests pass
4. Keep PRs focused — one feature or fix per PR

## Project Structure

See [ARCHITECTURE.md](../ARCHITECTURE.md) for a detailed system overview.

```
src/nexus/
  config/     # Pydantic models, YAML loader, presets
  engine/     # Pipeline: sources, filtering, knowledge, synthesis, audio
  agent/      # Telegram bot, Q&A, breaking news
  scheduler/  # APScheduler job definitions
  web/        # FastAPI dashboard + setup wizard
  llm/        # Multi-provider async LLM client
tests/        # Mirrors src structure
```

## Questions?

Open an issue on GitHub — we're happy to help.
