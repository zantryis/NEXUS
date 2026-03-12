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

### Without Docker

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
| `free` | Ollama (local) | $0 | None |
| `cheap` | DeepSeek V3.2 | ~$0.01 | `DEEPSEEK_API_KEY` |
| `balanced` | Gemini Flash + Pro | ~$0.05 | `GEMINI_API_KEY` |
| `quality` | Gemini Pro | ~$0.15 | `GEMINI_API_KEY` |
| `openai-cheap` | GPT-4.1 Nano + 5 Mini | ~$0.03 | `OPENAI_API_KEY` |
| `openai-balanced` | GPT-4.1 Mini + 5.4 | ~$0.10 | `OPENAI_API_KEY` |
| `openai-quality` | GPT-5 Mini + 5.4 | ~$0.25 | `OPENAI_API_KEY` |
| `anthropic` | Claude Haiku 4.5 / Sonnet 4.6 | ~$0.10 | `ANTHROPIC_API_KEY` |
| `custom` | Mix and match | Varies | Any provider key |

**Optional extras** (can be added later via Settings):
- `TELEGRAM_BOT_TOKEN` — Enable Telegram delivery (get from [@BotFather](https://t.me/BotFather))
- `ELEVENLABS_API_KEY` — Alternative TTS provider for podcasts

## CLI Commands

```bash
python -m nexus run                     # Start all services (dashboard + scheduler + Telegram bot)
python -m nexus engine                  # Run pipeline once
python -m nexus engine --topic <slug>   # Run pipeline for one topic
python -m nexus engine --capture        # Run pipeline and save fixtures for backtesting
python -m nexus serve                   # Dashboard only
python -m nexus setup                   # Interactive CLI setup wizard
python -m nexus sources check           # Test RSS feed health
python -m nexus sources list            # List all global sources
python -m nexus sources discover <slug> # Auto-discover RSS feeds for a topic
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
| `DEEPSEEK_API_KEY` | For DeepSeek preset | LLM completions |
| `TELEGRAM_BOT_TOKEN` | No | Telegram delivery (get from [@BotFather](https://t.me/BotFather)) |
| `ELEVENLABS_API_KEY` | No | Alternative TTS provider |

At least one LLM API key is required (unless using the free/Ollama preset).

### Topics (`data/config.yaml`)

Define your intelligence topics with subtopics, source languages, filter thresholds, and perspective diversity requirements. See `data/config.example.yaml` for the full schema.

Four topics ship with pre-built source registries: Iran-US Relations, AI/ML Research, Formula 1, and Global Energy Transition. For custom topics, run `python -m nexus sources discover <topic-slug>` to auto-discover RSS feeds.

### Demo Mode

To run a read-only demo instance (e.g., for sharing via Cloudflare tunnel):

```bash
NEXUS_DEMO_MODE=1 python -m nexus run
```

In demo mode, all settings are locked and a chat widget appears for visitor Q&A (rate-limited to 5 questions/day per IP).

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
pytest                          # Run 500+ tests
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
  runner.py       Unified runner (all services on one event loop)
data/
  sources/        Per-topic RSS source registries (pre-built for 4 topics)
  config.yaml     Your personal configuration (gitignored — created by setup wizard)
tests/            500+ tests mirroring src structure
```

## License

[MIT](LICENSE)
