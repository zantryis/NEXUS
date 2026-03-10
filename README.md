# Nexus

An agentic news intelligence compiler that polls 50+ RSS feeds across 8 languages, extracts structured events via LLM, resolves entities into a knowledge graph, identifies narrative threads with cross-source convergence/divergence analysis, generates daily briefings with podcast audio, and delivers via Telegram.

## Features

- **Multi-source intelligence** — 50+ RSS feeds across 8 languages with affiliation tracking (state/public/private media)
- **Two-pass LLM filtering** — Relevance scoring then significance + novelty assessment against known events
- **Knowledge graph** — SQLite-backed entity resolution, persistent narrative threads, convergence/divergence detection
- **Podcast audio** — Two-host dialogue script generation + Gemini TTS (380+ voices, 24+ languages)
- **Telegram delivery** — Daily briefings, breaking news alerts, knowledge-grounded Q&A
- **Web dashboard** — FastAPI + HTMX interface for exploring topics, threads, entities, events, and sources
- **Scheduled pipeline** — APScheduler runs the daily pipeline and breaking news checks automatically

## Quick Start

```bash
# Clone
git clone https://github.com/Tyan3001/NEXUS.git
cd NEXUS

# Install
pip install -e ".[all]"

# Configure
cp .env.example .env           # Add your API keys
cp data/config.example.yaml data/config.yaml  # Customize topics

# Run
python -m nexus run            # Start everything (dashboard + scheduler + Telegram bot)
python -m nexus engine         # Run pipeline once
python -m nexus serve          # Dashboard only
```

## Docker

```bash
cp .env.example .env           # Add your API keys
cp data/config.example.yaml data/config.yaml
docker compose up
```

Dashboard at `http://localhost:8080`. Podcast feed at `http://localhost:8080/feed.xml`.

## Configuration

### API Keys (`.env`)

| Key | Required | Purpose |
|-----|----------|---------|
| `GEMINI_API_KEY` | Yes | LLM completions + TTS |
| `TELEGRAM_BOT_TOKEN` | No | Telegram delivery (get from [@BotFather](https://t.me/BotFather)) |
| `DEEPSEEK_API_KEY` | No | Alternative LLM provider |
| `ANTHROPIC_API_KEY` | No | Alternative LLM provider |

### Topics (`data/config.yaml`)

Define your intelligence topics with subtopics, source languages, filter thresholds, and perspective diversity requirements. See `data/config.example.yaml` for the full schema.

Each topic gets its own source registry at `data/sources/<topic-slug>/registry.yaml` with RSS feeds, affiliation metadata, and tier rankings.

## Architecture

```
RSS Feeds (50+ sources, 8 languages)
    |
    v
POLL -> DEDUP -> INGEST -> FILTER -> EXTRACT -> DEDUP
                          (2-pass LLM)
    -> ENTITY RESOLVE -> SYNTHESIZE -> PERSIST THREADS
    -> REFRESH PAGES -> RENDER BRIEFING -> AUDIO PIPELINE
    |                    |                    |
    v                    v                    v
SQLite Knowledge      Telegram Bot         Dashboard
  Graph               (delivery +          (FastAPI +
                       Q&A + alerts)        Jinja2)
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full module map and schema details.

## Development

```bash
# Install with dev dependencies
pip install -e ".[all,dev]"

# Run tests
pytest

# Smoke test (validates config, feeds, LLM, store)
python scripts/smoke_test.py
```

## Project Structure

```
src/nexus/
  engine/         Pipeline: sources, ingestion, filtering, knowledge, synthesis, audio
  agent/          Telegram bot, Q&A, breaking news, delivery
  scheduler/      APScheduler job definitions
  web/            FastAPI dashboard + podcast RSS
  llm/            Multi-provider async LLM client (Gemini, Anthropic, DeepSeek)
  config/         Pydantic config models
  runner.py       Unified runner (all services on one event loop)
data/
  sources/        Per-topic RSS source registries
  config.yaml     Your personal topic configuration (gitignored)
tests/            316+ tests mirroring src structure
```
