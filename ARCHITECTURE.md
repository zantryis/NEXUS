# Nexus-Claude: Architecture & Capability Overview

## What It Is

An agentic news intelligence compiler that polls 52+ RSS feeds across 8 languages, filters and extracts structured events via LLM, resolves entities into a knowledge graph, identifies narrative threads with cross-source convergence/divergence analysis, generates daily briefings with podcast audio, and delivers via Telegram — all backed by a SQLite knowledge store and packaged as a Docker service.

## System Stats

| Metric | Value |
|--------|-------|
| Source files | ~50 modules |
| Tests passing | 316 / 316 |
| Source feeds | 52 feeds, 8 languages |
| LLM providers | Gemini, Anthropic, DeepSeek |
| TTS backend | Gemini native (380+ voices, 24+ languages) |
| Knowledge store | SQLite, 14 tables, WAL mode, schema v3 |
| Delivery | Telegram bot + Web dashboard + Podcast RSS |

## Pipeline Architecture

```
RSS Feeds (52 sources, 8 languages)
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  POLL → DEDUP → INGEST → FILTER → EXTRACT → DEDUP      │
│  (feedparser) (URL)  (trafilatura) (2-pass LLM) (LLM)   │
│                                                          │
│  → ENTITY RESOLVE → SYNTHESIZE → PERSIST THREADS         │
│    (LLM canonical.)  (convergence/   (entity overlap     │
│                       divergence)     + LLM matching)     │
│                                                          │
│  → REFRESH PAGES → RENDER BRIEFING → AUDIO PIPELINE      │
│    (TTL-based)      (markdown)       (script → TTS → MP3)│
└─────────────────────────────────────────────────────────┘
    │                    │                    │
    ▼                    ▼                    ▼
SQLite Knowledge    Telegram Bot         Dashboard
  Graph               (delivery +         (FastAPI +
                       Q&A + alerts)       Jinja2)
```

## Knowledge Graph Schema (v3)

```
entities ──┐
           ├── event_entities (M2M)
events ────┘
  │
  ├── event_sources (1:N)
  │
  ├── thread_events (M2M) ──── threads
  │                              ├── convergence
  │                              ├── divergence
  │                              └── thread_topics (M2M)
  │
  ├── summaries (weekly/monthly compression)
  ├── syntheses (daily snapshots)
  ├── pages (cached LLM narratives)
  ├── filter_log (decision audit trail)
  ├── breaking_alerts (dedup for alerts)
  └── feedback (user briefing ratings)
```

**Entity types:** person | org | country | treaty | concept | unknown
**Thread statuses:** emerging → active → stale → resolved
**Page types:** backstory | entity_profile | thread_deepdive | weekly_recap

## Module Map

### Core Pipeline (`src/nexus/engine/`)

| Module | Purpose |
|--------|---------|
| `pipeline.py` | Orchestrator: `run_pipeline()`, `run_backtest()` |
| `filtering/filter.py` | Two-pass LLM filter: relevance → significance+novelty |
| `ingestion/ingest.py` | Async article fetch, trafilatura extraction, language detection |
| `sources/polling.py` | RSS polling via feedparser, ContentItem model |
| `sources/registry.py` | 52-feed global registry with affiliation/country metadata |

### Knowledge Layer (`src/nexus/engine/knowledge/`)

| Module | Purpose |
|--------|---------|
| `store.py` | `KnowledgeStore` — all CRUD against SQLite |
| `schema.py` | DDL for 14 tables, indexes, v3 migrations |
| `entities.py` | LLM entity resolution: raw strings → canonical graph nodes |
| `pages.py` | Cached narrative page generation with TTL staleness |
| `events.py` | Event model, LLM extraction, dedup/merge |
| `compression.py` | Weekly/monthly summary compression |

### Synthesis (`src/nexus/engine/synthesis/`)

| Module | Purpose |
|--------|---------|
| `knowledge.py` | `TopicSynthesis` builder — threads, convergence, divergence |
| `threads.py` | Persistent thread matching (entity overlap + LLM) |
| `renderers.py` | Text briefing renderer |

### Audio Pipeline (`src/nexus/engine/audio/`)

| Module | Purpose |
|--------|---------|
| `script.py` | Two-host dialogue script generation via LLM |
| `tts.py` | TTS backends (Gemini native, extensible) |
| `concat.py` | Audio segment concatenation (pydub → MP3) |
| `pipeline.py` | Audio pipeline orchestrator |

### Telegram Agent (`src/nexus/agent/`)

| Module | Purpose |
|--------|---------|
| `bot.py` | Telegram bot (long-polling, /start, /briefing, /status, Q&A) |
| `qa.py` | Knowledge-grounded question answering |
| `breaking.py` | Wire feed polling + LLM significance scoring |
| `delivery.py` | Briefing + audio delivery via Telegram |
| `feedback.py` | Inline keyboard feedback handling |

### Scheduler (`src/nexus/scheduler/`)

| Module | Purpose |
|--------|---------|
| `jobs.py` | APScheduler job definitions (daily pipeline, breaking news) |

### Web Dashboard (`src/nexus/web/`)

| Module | Purpose |
|--------|---------|
| `app.py` | FastAPI app factory with KnowledgeStore lifespan |
| `routes/` | 9 route modules (dashboard, topics, threads, events, entities, pages, filters, sources, podcast) |
| `templates/` | 13 Jinja2 templates (Pico CSS dark theme + HTMX) |

### Infrastructure

| Module | Purpose |
|--------|---------|
| `llm/client.py` | Multi-provider async LLM client with usage tracking |
| `runner.py` | Unified runner (dashboard + scheduler + Telegram bot) |
| `testing/fixtures.py` | Fixture capture/replay for deterministic backtesting |
| `config/models.py` | Pydantic config models |
| `evaluation/` | LLM-as-judge + automated metrics |

## Key Design Decisions

1. **SQLite as knowledge store** — Single file, no server, portable, relational queries for entity/thread joins. WAL mode for read concurrency.

2. **Entity resolution as pipeline stage** — After event extraction, LLM canonicalizes entity strings. Entities are first-class graph nodes, not strings.

3. **Two-stage thread matching** — Entity overlap (Jaccard ≥ 0.5 = auto-match) + LLM confirmation for ambiguous cases (0.3–0.5 overlap).

4. **Gemini TTS** — Uses existing GEMINI_API_KEY, no additional keys needed. 380+ voices, 24+ languages.

5. **Telegram long-polling** — Works behind NAT/firewall, no public URL needed. Single-user authorization via chat_id.

6. **Unified runner** — Single process runs dashboard, scheduler, and Telegram bot on one asyncio event loop.

7. **Source attribution throughout** — Every event tracks source outlet, affiliation (state/public/private), country, and language. Convergence requires 2+ independent sources.

## Running

```bash
# Development
python -m nexus run                          # Everything
python -m nexus engine                       # Pipeline only
python -m nexus serve                        # Dashboard only

# Docker
docker compose up                            # Everything in container
```
