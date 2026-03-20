# Contributing to Nexus

Nexus is a self-hosted intelligence project, not a framework. The best contributions make the core product sharper: better ingestion, clearer threads, more trustworthy briefings, cleaner setup, and less accidental complexity.

## Start Here

The stable release-facing docs are:

- [`README.md`](README.md) for setup and the user-facing product story
- [`docs/index.html`](docs/index.html) for the landing page
- [`docs/pipeline.html`](docs/pipeline.html) for the public system map
- [`docs/release-checklist.md`](docs/release-checklist.md) for release sign-off

The forecast benchmark, hindcast, and other research-heavy material are still in the repo, but they are lab surfaces, not the main contributor funnel.

## Development Setup

```bash
git clone https://github.com/zantryis/NEXUS.git
cd NEXUS
python -m venv .venv
source .venv/bin/activate
pip install -e ".[all,dev]"
cp .env.example .env
cp data/config.example.yaml data/config.yaml
```

If you want the easiest first run, use the browser setup flow instead:

```bash
python -m nexus run
```

Then open `http://localhost:8080` and let the setup wizard write `data/config.yaml` and `.env` for you.

## Default Workflow

Use the smallest loop that proves the change:

```bash
.venv/bin/ruff check src tests
.venv/bin/pytest -m "not e2e and not integration"
```

Useful focused commands:

```bash
.venv/bin/pytest tests/web/test_app.py -q
.venv/bin/pytest tests/engine/knowledge/test_store.py -q
python -m nexus test
```

Use `python -m nexus test` only when you actually need the real end-to-end smoke path.

## Test Markers

Tests that need live services are opt-in:

| Marker | Use it for | Needs |
|--------|------------|-------|
| `@pytest.mark.integration` | Live API calls | API keys in `.env` |
| `@pytest.mark.e2e` | Full product smoke path | API keys + network |

Everything else should stay deterministic and runnable offline.

## Stable vs Lab Surfaces

When in doubt, optimize for the stable path:

- Stable: setup, config loading, polling, filtering, knowledge store, threads, briefings, dashboard, Forward Look, optional Kalshi comparison
- Lab: benchmark suites, hindcast, multi-engine comparisons, experiment harnesses, research writeups

That distinction matters in code review. Stable-path changes need stronger simplicity and docs discipline. Lab changes can stay narrower and more experimental, but they should not leak into the default onboarding story by accident.

## Fixture Capture

The repo includes fixture capture and replay support for pipeline work:

```bash
python -m nexus experiment --suite A,G --export-fixtures fixtures/
python -m nexus experiment --suite A,G --import-fixtures fixtures/
python -m nexus experiment --suite A,G --compare fixtures/cloud fixtures/local
```

If you add a new LLM-backed stage that should be replayable offline, wire it into `src/nexus/testing/`.

## Code Expectations

- Keep the codebase smaller, not just newer.
- Prefer deleting stale paths over preserving pre-`0.1` compatibility baggage.
- Route all LLM calls through `src/nexus/llm/client.py`.
- Use `logging`, not `print()`, outside of CLI commands and tests.
- Add tests for risky behavior changes, especially setup flows, public routes, and store selectors.
- Keep docs aligned with shipped behavior in the same PR when you change the public surface.

## Pull Requests

- Keep PRs focused.
- Explain the user-facing impact and the maintenance tradeoff.
- Call out if a change touches the stable release path or only lab tooling.
- Before asking for review, run the relevant tests and note anything you could not run.

For release work, use [`docs/release-checklist.md`](docs/release-checklist.md) as the source of truth.
