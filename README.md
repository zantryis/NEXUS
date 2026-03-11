# Nexus

An agentic news intelligence compiler that polls 50+ RSS feeds across 8 languages, extracts structured events via LLM, resolves entities into a knowledge graph, identifies narrative threads with cross-source convergence/divergence analysis, generates daily briefings with podcast audio, and delivers via Telegram.

## Features

- **Multi-source intelligence** — 50+ RSS feeds across 8 languages with affiliation tracking (state/public/private media)
- **Two-pass LLM filtering** — Relevance scoring then significance + novelty assessment against known events
- **Knowledge graph** — SQLite-backed entity resolution, persistent narrative threads, convergence/divergence detection
- **Podcast audio** — Two-host dialogue script generation + TTS (Gemini, OpenAI, or ElevenLabs)
- **Telegram delivery** — Daily briefings, breaking news digests, knowledge-grounded Q&A with web search
- **Web dashboard** — FastAPI + HTMX interface for exploring topics, threads, entities, events, and sources
- **Scheduled pipeline** — APScheduler runs the daily pipeline and breaking news checks automatically
- **Source auto-discovery** — LLM-powered RSS feed discovery for new topics
- **Multi-provider LLM** — Gemini, OpenAI, Anthropic, DeepSeek, and Ollama (local)
- **Budget controls** — Daily spend limits with automatic degradation strategies

## Quick Start

```bash
# Clone and install
git clone https://github.com/Tyan3001/NEXUS.git
cd NEXUS
pip install -e ".[all]"

# Interactive setup (creates config + .env)
python -m nexus setup

# Run everything
python -m nexus run
```

The setup wizard lets you pick a model provider (free local, DeepSeek, Gemini, OpenAI, or Anthropic), select topics, and configure your API key.

### Manual Setup

If you prefer to configure manually:

```bash
cp .env.example .env                          # Add your API keys
cp data/config.example.yaml data/config.yaml  # Customize topics and models
python -m nexus run
```

## Docker

```bash
python -m nexus setup    # Or manually: cp .env.example .env && cp data/config.example.yaml data/config.yaml
docker compose up
```

Dashboard at `http://localhost:8080`. Podcast feed at `http://localhost:8080/feed.xml`.

## CLI Commands

```bash
python -m nexus run                     # Start all services (dashboard + scheduler + Telegram bot)
python -m nexus engine                  # Run pipeline once
python -m nexus engine --topic <slug>   # Run pipeline for one topic
python -m nexus engine --capture        # Run pipeline and save fixtures for backtesting
python -m nexus serve                   # Dashboard only
python -m nexus setup                   # Interactive setup wizard
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

### Model Presets

Choose a preset in `data/config.yaml` or via the setup wizard:

| Preset | Provider | Cost/Day | Key Needed |
|--------|----------|----------|------------|
| `free` | Ollama (local) | $0 | None |
| `cheap` | DeepSeek | ~$0.01 | `DEEPSEEK_API_KEY` |
| `balanced` | Gemini Flash + Pro | ~$0.05 | `GEMINI_API_KEY` |
| `quality` | Gemini Pro | ~$0.15 | `GEMINI_API_KEY` |
| `openai-cheap` | GPT-4.1 Nano/Mini | ~$0.03 | `OPENAI_API_KEY` |
| `openai-balanced` | GPT-4.1 Mini/Full | ~$0.10 | `OPENAI_API_KEY` |
| `anthropic` | Claude Haiku/Sonnet | ~$0.10 | `ANTHROPIC_API_KEY` |

### Topics (`data/config.yaml`)

Define your intelligence topics with subtopics, source languages, filter thresholds, and perspective diversity requirements. See `data/config.example.yaml` for the full schema.

Four topics ship with pre-built source registries: Iran-US Relations, AI/ML Research, Formula 1, and Global Energy Transition. For custom topics, run `python -m nexus sources discover <topic-slug>` to auto-discover RSS feeds.

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
pytest                          # Run 450+ tests
```

## Project Structure

```
src/nexus/
  engine/         Pipeline: sources, ingestion, filtering, knowledge, synthesis, audio
  agent/          Telegram bot, Q&A, breaking news, delivery, web search
  scheduler/      APScheduler job definitions
  web/            FastAPI dashboard + podcast RSS
  llm/            Multi-provider async LLM client (Gemini, OpenAI, Anthropic, DeepSeek, Ollama)
  config/         Pydantic config models + presets
  runner.py       Unified runner (all services on one event loop)
data/
  sources/        Per-topic RSS source registries (pre-built for 4 topics)
  config.yaml     Your personal configuration (gitignored — copy from config.example.yaml)
tests/            450+ tests mirroring src structure
```
