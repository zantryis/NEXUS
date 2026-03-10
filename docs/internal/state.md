# Session State

## Current Phase
Always-on service complete — ready for Docker deployment.

## What's Built

### Engine (Phase 0+1) — COMPLETE
- Config: Pydantic models + YAML loader, per-topic filter thresholds + perspective diversity
- LLM: Multi-provider async client (Gemini, Anthropic, DeepSeek) with config_key routing + usage tracking
- Source polling: feedparser RSS, ContentItem model with affiliation/language metadata
- Source registries: 52 global feeds across 8 languages with affiliation tracking
- Ingestion: async (semaphore-limited), trafilatura extraction, language detection, paywall detection
- Filtering: two-pass (batch relevance → significance+novelty with knowledge context), perspective diversity
- Knowledge layer: SQLite-backed knowledge graph (14 tables, WAL mode, v3 schema)
  - Entity resolution (LLM-based canonicalization into graph nodes)
  - Persistent narrative threads (entity overlap matching + LLM, lifecycle management)
  - Cached narrative pages (backstory, entity profiles, thread deep-dives with TTL)
  - Breaking alerts dedup table + user feedback table
- Synthesis: TopicSynthesis with NarrativeThread, convergence/divergence, thread persistence
- Renderers: text briefing from TopicSynthesis objects
- Evaluation: automated metrics (Shannon entropy, convergence ratio), LLM-as-judge
- Pipeline orchestrator: poll → dedup → ingest → filter → extract → dedup → entity resolve → synthesize → persist threads → refresh pages → render → audio
- Fixture capture/replay + backtest infrastructure

### Dashboard (Phase 3+4) — COMPLETE
- FastAPI + Jinja2 + HTMX + Pico CSS dark theme
- Routes: dashboard, topics, threads, events, entities, pages, filters, sources, podcast RSS
- 17 web tests

### Audio Pipeline (Phase A) — COMPLETE
- Dialogue script generation (two-host podcast from TopicSynthesis via LLM)
- Gemini TTS backend (native via google-genai, 380+ voices, 24+ languages)
- Audio concatenation (pydub, WAV→MP3, 300ms silence gaps)
- Pipeline integration into engine + podcast RSS feed at /feed.xml

### Telegram Agent (Phase B) — COMPLETE
- Bot with long-polling (/start, /briefing, /status, text Q&A)
- Q&A agent (knowledge context from store → LLM answer)
- Breaking news poller (wire RSS → LLM scoring → dedup alerts)
- Briefing delivery (text + audio) + inline feedback (thumbs up/down)
- Schema v3: breaking_alerts + feedback tables

### Scheduler + Runner (Phase C) — COMPLETE
- APScheduler: daily pipeline at config schedule, breaking news every N hours
- Unified runner: `python -m nexus run` starts dashboard + scheduler + Telegram bot
- All services share one asyncio event loop

### Docker (Phase D) — COMPLETE
- Dockerfile (python:3.11-slim + ffmpeg)
- docker-compose.yml (data/ volume, .env, port 8080)

## Test Counts
- 316 tests, all passing
- Config: 15 | Audio: 18 | Agent: 24 | Scheduler: 7 | Runner: 1
- Knowledge store: 41 | Schema: 9 | Entity: 15 | Threads: 19 | Pages: 12
- Events: 16 | Filtering: 17 | Synthesis: 9 | Pipeline: 6
- Ingestion: 10 | Sources: 16 | LLM: 6 | Dedup: 9 | Fixtures: 8
- Evaluation: 13 | Perspective: 5 | Web: 20

## CLI Commands
- `python -m nexus run` — Start everything (dashboard + scheduler + Telegram bot)
- `python -m nexus engine` — Run daily pipeline once
- `python -m nexus serve` — Dashboard only
- `python -m nexus sources check|build|list` — Source management
- `python -m nexus evaluate synthesis|compare` — Quality evaluation
- `docker compose up` — Run via Docker
