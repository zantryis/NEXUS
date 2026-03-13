# Nexus

An agentic news intelligence compiler that polls 50+ RSS feeds across 8 languages, extracts structured events via LLM, resolves entities into a knowledge graph, identifies narrative threads with cross-source convergence/divergence analysis, generates daily briefings with podcast audio, and delivers via Telegram.

## Features

- **Multi-source intelligence** — 50+ RSS feeds across 8 languages with affiliation tracking (state/public/private media)
- **Two-pass LLM filtering** — Relevance scoring then significance + novelty assessment against known events
- **Knowledge graph** — SQLite-backed entity resolution, persistent narrative threads, convergence/divergence detection
- **Podcast audio** — Two-host dialogue script generation + TTS (Gemini, OpenAI, or ElevenLabs)
- **Telegram delivery** — Daily briefings, breaking news digests, knowledge-grounded Q&A with web search
- **Web dashboard** — FastAPI + HTMX interface for exploring topics, threads, entities, events, and sources
- **Web setup wizard** — Browser-based first-run configuration (no YAML editing required)
- **Web chat** — Built-in Q&A widget on the dashboard, rate-limited per IP
- **Demo mode** — Read-only dashboard for public showcasing via tunnel
- **Scheduled pipeline** — APScheduler runs the daily pipeline and breaking news checks automatically
- **Source auto-discovery** — LLM-powered RSS feed discovery for new topics
- **Multi-provider LLM** — Gemini, OpenAI, Anthropic, DeepSeek, and Ollama (local)
- **Budget controls** — Daily spend limits with automatic degradation strategies

## Quick Start

### Docker (recommended)

```bash
git clone https://github.com/Tyan3001/NEXUS.git
cd NEXUS
docker compose up
```

Open `http://localhost:8080` — the web setup wizard will guide you through configuration. No manual file editing needed.

After the wizard saves `data/config.yaml`, restart the container once so the scheduler, Telegram bot, and other always-on services start with your new config.

### Without Docker

Requires Python 3.11+ and [ffmpeg](https://ffmpeg.org/) (for podcast audio).

```bash
git clone https://github.com/Tyan3001/NEXUS.git
cd NEXUS
pip install -e ".[all]"
python -m nexus run
```

Open `http://localhost:8080` to complete setup via the web wizard, or use the CLI wizard:

```bash
python -m nexus setup    # Interactive CLI setup
python -m nexus run      # Start all services
```

If you use the web setup wizard from `python -m nexus run`, restart that process once after setup completes so scheduled jobs and Telegram delivery start with the saved config.

### Manual Setup

If you prefer to configure manually:

```bash
cp .env.example .env                          # Add your API keys
cp data/config.example.yaml data/config.yaml  # Customize topics and models
python -m nexus run
```

Dashboard at `http://localhost:8080`. Podcast feed at `http://localhost:8080/feed.xml`.

## What You Need

Only **one LLM API key** is required to get started. The setup wizard will help you choose:

| Preset | Provider | Cost/Day | Key Needed |
|--------|----------|----------|------------|
| `free` | Ollama (local) | $0 | None (requires running [Ollama](https://ollama.com) server) |
| `cheap` | DeepSeek V3.2 | ~$0.01 | `DEEPSEEK_API_KEY` |
| `balanced` | Gemini Flash + Pro | ~$0.05 | `GEMINI_API_KEY` |
| `quality` | Gemini Pro | ~$0.15 | `GEMINI_API_KEY` |
| `openai-cheap` | GPT-4.1 Nano + 5 Mini | ~$0.03 | `OPENAI_API_KEY` |
| `openai-balanced` | GPT-4.1 Mini + 5.4 | ~$0.10 | `OPENAI_API_KEY` |
| `openai-quality` | GPT-5 Mini + 5.4 | ~$0.25 | `OPENAI_API_KEY` |
| `anthropic` | Claude Haiku 4.5 / Sonnet 4.6 | ~$0.10 | `ANTHROPIC_API_KEY` |

**Optional extras** (can be added later via Settings):
- `TELEGRAM_BOT_TOKEN` — Enable Telegram delivery (get from [@BotFather](https://t.me/BotFather))
- `ELEVENLABS_API_KEY` — Alternative TTS provider for podcasts

## CLI Commands

```bash
python -m nexus run                     # Start all services (binds to localhost by default)
python -m nexus run --port 9090         # Start on a different port (default: 8080)
python -m nexus run --host 0.0.0.0      # Explicitly expose beyond localhost
python -m nexus engine                  # Run pipeline once
python -m nexus engine --topic <slug>   # Run pipeline for one topic
python -m nexus engine --capture        # Run pipeline and save fixtures for backtesting
python -m nexus serve                   # Dashboard only (also accepts --port/--host)
python -m nexus setup                   # Interactive CLI setup wizard
python -m nexus sources check           # Test RSS feed health
python -m nexus sources list            # List all global sources
python -m nexus sources discover <slug> # Auto-discover RSS feeds for a topic
python -m nexus test                            # E2E smoke test (runs minimal pipeline)
python -m nexus evaluate synthesis <path>       # Judge synthesis quality
python -m nexus evaluate compare <path> <path>  # Compare two syntheses
```

## Configuration

### API Keys (`.env`)

| Key | Required | Purpose |
|-----|----------|---------|
| `GEMINI_API_KEY` | For Gemini preset | LLM completions + TTS |
| `OPENAI_API_KEY` | For OpenAI preset | LLM completions + TTS |
| `ANTHROPIC_API_KEY` | For Anthropic preset | LLM completions |
| `DEEPSEEK_API_KEY` | For DeepSeek preset | LLM completions (legacy `deepseek` env name is still accepted) |
| `TELEGRAM_BOT_TOKEN` | No | Telegram delivery (get from [@BotFather](https://t.me/BotFather)) |
| `ELEVENLABS_API_KEY` | No | Alternative TTS provider |
| `OLLAMA_BASE_URL` | No | Ollama server URL (default: `http://localhost:11434`) |
| `NEXUS_ADMIN_TOKEN` | No | Required for remote access to `/setup` and `/settings` |

At least one LLM API key is required (unless using the free/Ollama preset).

> **Ollama + Docker:** The `free` preset expects Ollama at `localhost:11434`. In Docker, set `OLLAMA_BASE_URL=http://host.docker.internal:11434` in `.env` to reach a host-side Ollama server, or add an Ollama service to `docker-compose.yml`.

### Network Exposure

`python -m nexus run` binds to `127.0.0.1` by default. If you intentionally expose Nexus with `--host 0.0.0.0`, the dashboard becomes network-reachable.

- `/setup` and `/settings` are localhost-only unless you set `NEXUS_ADMIN_TOKEN`
- Remote admin access works by opening `http://your-host:8080/settings?admin_token=YOUR_TOKEN` once; Nexus then stores a short-lived admin cookie
- After first-run setup, the setup wizard is disabled unless you set `NEXUS_ALLOW_SETUP_RESET=1`
- If you publish Nexus behind a reverse proxy or tunnel, add external auth there too

### Topics (`data/config.yaml`)

Define your intelligence topics with subtopics, source languages, filter thresholds, and perspective diversity requirements. See `data/config.example.yaml` for the full schema.

Four topics ship with pre-built source registries: Iran-US Relations, AI/ML Research, Formula 1, and Global Energy Transition. For custom topics, run `python -m nexus sources discover <topic-slug>` to auto-discover RSS feeds.

### Demo Mode

To run a read-only demo instance (e.g., for sharing via Cloudflare tunnel):

```bash
NEXUS_DEMO_MODE=1 python -m nexus run
```

In demo mode, all settings are locked and a chat widget appears for visitor Q&A (rate-limited to 5 questions/day per IP).

### Smoke Mode

To cap pipeline ingestion for fast testing (e.g., verifying the setup flow end-to-end):

```bash
NEXUS_SMOKE_MODE=20 python -m nexus run   # Limits ingestion to 20 articles per topic
```

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
pip install -e ".[all,dev]"
pytest                          # Run unit tests
pytest -m e2e                   # Run E2E smoke tests (requires API keys)
```

## Project Structure

```
src/nexus/
  engine/         Pipeline: sources, ingestion, filtering, knowledge, synthesis, audio
  agent/          Telegram bot, Q&A, breaking news, delivery, web search
  scheduler/      APScheduler job definitions
  web/            FastAPI dashboard, setup wizard, settings, chat widget
  llm/            Multi-provider async LLM client (Gemini, OpenAI, Anthropic, DeepSeek, Ollama)
  config/         Pydantic config models, presets, config writer
  testing/        E2E smoke test runner
  runner.py       Unified runner (all services on one event loop)
data/
  sources/        Per-topic RSS source registries (pre-built for 4 topics)
  config.yaml     Your personal configuration (gitignored — created by setup wizard)
tests/            640 tests mirroring src structure (unit + e2e)
```

## License

[MIT](LICENSE)
