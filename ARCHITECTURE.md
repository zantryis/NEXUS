# Nexus Architecture

An agentic news intelligence compiler that polls 52+ RSS feeds across 8 languages, extracts structured events via LLM, resolves entities into a knowledge graph, identifies narrative threads with cross-source convergence/divergence analysis, generates daily briefings with podcast audio, and delivers via Telegram — all backed by a SQLite knowledge store.

## System Stats

| Metric | Value |
|--------|-------|
| Source files | ~76 Python modules |
| Tests passing | 454 |
| Source feeds | 52 global + per-topic registries, 8 languages |
| LLM providers | Gemini, OpenAI, Anthropic, DeepSeek, Ollama |
| TTS backends | Gemini native, OpenAI, ElevenLabs |
| Knowledge store | SQLite, 16 tables, WAL mode, schema v4 |
| Delivery | Telegram bot + Web dashboard + Podcast RSS |

---

## Three Runtime Flows

The system has three independent code flows, all sharing the same SQLite knowledge store and LLM client:

1. **Daily Pipeline** — Scheduled batch job. Polls feeds → extracts events → builds knowledge graph → generates briefing + audio.
2. **Telegram Bot** — Long-polling bot. Delivers briefings, handles Q&A (knowledge-grounded + web search), collects feedback.
3. **Breaking News** — Interval job. Polls wire feeds → LLM scores significance → sends alerts for threshold-exceeding headlines.

All three run on a single asyncio event loop via `runner.py`.

---

## Flow 1: Daily Pipeline

**Entry point:** `run_pipeline()` — `src/nexus/engine/pipeline.py:254`

Runs for each topic in config, then renders cross-topic briefing and audio.

### Step 1: Source Polling
**`poll_all_feeds(sources)`** — `src/nexus/engine/sources/polling.py:70`

- Parses each RSS feed via `feedparser`
- Creates `ContentItem` per entry (title, URL, snippet, published date)
- Attaches source metadata: `source_id`, `language`, `affiliation` (state/public/private/nonprofit/academic), `country`, `tier` (A/B/C)

### Step 2: URL Deduplication
**`dedup_items(raw_items)`** — `src/nexus/engine/ingestion/dedup.py:26`

- Normalizes URLs (strips tracking params, fragments, trailing slashes)
- Keeps first occurrence per normalized URL

### Step 3: Full-Text Ingestion
**`async_ingest_items(unique_items)`** — `src/nexus/engine/ingestion/ingest.py:97`

- Fetches HTML via `httpx`, extracts text via `trafilatura`
- Rate-limited: global semaphore (10 concurrent), per-domain semaphore (2)
- Detects paywalls heuristically
- Sets `full_text` and `detected_language` on each item

### Step 4: Two-Pass LLM Filtering
**`filter_items(llm, ingested, topic, recent_events)`** — `src/nexus/engine/filtering/filter.py:263`

**Pass 1 — Relevance (batch, cheap model):**
- Scores each article 1-10 against topic name + subtopics
- Threshold: `topic.filter_threshold` (default 6.0)
- Batch size: 10 articles per LLM call

**Pass 2 — Significance + Novelty (individual, cheap model):**
- Receives recent 7-day events as context (what's already known)
- Scores significance 1-10, flags novelty (bool)
- Passes if `significance >= 4 OR is_novel == true`
- Skipped entirely if no recent events exist (first run)

**Perspective Diversity:**
- Ensures minimum representation per affiliation type (20% for high, 10% for medium)
- Returns top 30 by diversity-weighted composite score: 40% relevance + 60% significance + novelty bonus

Returns `FilterResult` with `accepted: list[ContentItem]` and `log_entries: list[dict]` (full audit trail).

### Step 5: Event Extraction
**`extract_event(llm, item, topic, existing_events, current_date)`** — `src/nexus/engine/knowledge/events.py:114`

- LLM extracts structured `Event` from each article
- Prompt includes: topic context, today's date, recent events (for relation_to_prior), source metadata, first 3000 chars of article
- Future dates hard-clamped to current_date
- Parallel extraction with semaphore(5)
- Event cap per topic: narrow=15, medium=20, broad=35 (overridable via `topic.max_events`)

### Step 6: Event Dedup & Merge
**`is_duplicate_event()`** — `src/nexus/engine/knowledge/events.py:52`
**`merge_events()`** — `src/nexus/engine/knowledge/events.py:78`

- Entity overlap (Jaccard similarity >= 0.6) + date proximity (±1 day) = duplicate
- Merge combines sources (by URL), entities (deduplicated), and takes max significance
- Checks against both same-run events and existing events from store

### Step 7: Entity Resolution
**`resolve_entities(llm, all_raw, known)`** — `src/nexus/engine/knowledge/entities.py:41`

- Collects unique raw entity strings from all new events
- LLM canonicalizes: "Donald Trump" / "Trump" / "President Trump" → canonical "Donald Trump", type "person"
- Known entities from store provided as context to avoid duplicates
- Returns `list[EntityResolution]` with canonical name, type, is_new flag
- Entity types: person | org | country | treaty | concept | unknown
- Upserts into store, links events ↔ entities via `event_entities` junction table

### Step 8: Knowledge Compression
**`maybe_compress(llm, store, topic_slug, topic_name, events)`** — `src/nexus/engine/pipeline.py:45`

- Compresses event weeks older than 7 days into weekly `Summary` objects
- LLM creates narrative summary per ISO week
- Summaries stored in `summaries` table, used as background context for synthesis

### Step 9: Synthesis
**`synthesize_topic(llm, topic, events, articles, weekly_summaries, monthly_summaries, store, topic_slug)`** — `src/nexus/engine/synthesis/knowledge.py:145`

Builds the central `TopicSynthesis` object:

1. Formats events + background summaries as context
2. Scope-aware system prompt (narrow/medium/broad affects thread granularity)
3. LLM generates `NarrativeThread` objects with:
   - `headline`: descriptive thread title
   - `events`: indices mapping to extracted events
   - `convergence`: facts confirmed by 2+ independent sources
   - `divergence`: conflicting framings between sources (source_a vs source_b)
   - `key_entities`: entities central to this thread
   - `significance`: 1-10 score
4. Fallback: if LLM fails, creates one thread per event

### Step 10: Thread Persistence
**`match_events_to_threads()`** — `src/nexus/engine/synthesis/threads.py:82`
**`_persist_threads()`** — `src/nexus/engine/synthesis/knowledge.py:271`

Two-stage matching:
1. **Entity overlap (no LLM):** Jaccard >= 0.5 → auto-match, 0.3-0.5 → ambiguous
2. **LLM confirmation:** for ambiguous/unmatched cases

Thread lifecycle: **emerging → active → stale → resolved**

Persists to `threads`, `thread_events`, `thread_topics`, `convergence`, `divergence` tables.

### Step 11: Page Refresh
**`refresh_stale_pages()`** — `src/nexus/engine/knowledge/pages.py`

LLM-generated cached narrative pages with TTL:
- `backstory` — topic background, TTL 7 days
- `entity_profile` — entity deep-dive, TTL 3 days
- `thread_deepdive` — thread analysis, TTL 1 day
- `weekly_recap` — weekly summary, TTL 365 days

### Step 12: Briefing Rendering
**`render_text_briefing(llm, config, syntheses)`** — `src/nexus/engine/synthesis/renderers.py:93`

- Takes all `TopicSynthesis` objects across topics
- LLM renders markdown briefing (<800 words)
- 2-3 sentence executive summary, `##` per topic, key claims attributed to sources, source tally
- Saved to `data/artifacts/briefings/{today}.md`

### Step 13: Audio Pipeline
**`run_audio_pipeline()`** — `src/nexus/engine/audio/pipeline.py:17`

Three stages:
1. **Script generation** (`audio/script.py:62`): LLM generates two-host dialogue (Nova + Atlas) from synthesis context
2. **TTS synthesis** (`audio/tts.py`): Per-turn audio via configured backend (Gemini/OpenAI/ElevenLabs)
3. **Concatenation** (`audio/concat.py:37`): Joins segments with 300ms silence gaps, exports MP3

Saved to `data/artifacts/audio/{today}.mp3`

---

## Flow 2: Telegram Bot

**Entry point:** `NexusBot.start()` — `src/nexus/agent/bot.py:41`

Long-polling bot with 5 handlers:

### Commands
- **/start** — Authorization, records `chat_id` for delivery
- **/briefing** — Reads today's briefing + audio from artifacts, delivers via `deliver_briefing()`, sends feedback keyboard
- **/status** — Shows topic stats (event/thread counts), briefing availability

### Q&A (text messages)
**`answer_question()`** — `src/nexus/agent/qa.py:159`

Three-stage pipeline:
1. **Analyze question** (`qa.py:56`): LLM extracts entities and intent
2. **Gather context** (`qa.py:71`):
   - Recent events, active threads, entity-targeted lookups
   - Convergence/divergence data for relevant threads
   - Background pages and weekly summaries
   - **Web search fallback** if knowledge context is thin (via `web_search()`)
3. **Generate answer** (`qa.py:189`): LLM answers with full context, markdown output

### Feedback
- Inline keyboard (thumbs up/down) after briefing delivery
- Stored in `feedback` table with briefing date

---

## Flow 3: Breaking News

**`check_breaking_news()`** — `src/nexus/agent/breaking.py:75`

- Polls wire feeds (configurable RSS sources or defaults)
- LLM scores headline significance (1-10)
- Deduplicates via `headline_hash` in `breaking_alerts` table
- Alerts above `config.breaking_news.threshold` (default 7) delivered via `deliver_breaking_digest()`

Runs every `poll_interval_hours` (default 3) via APScheduler.

---

## LLM Client

**`LLMClient`** — `src/nexus/llm/client.py:88`

### Provider Resolution (`client.py:17`)
| Prefix | Provider |
|--------|----------|
| `ollama/...` | Ollama (local) |
| `gemini...` | Gemini |
| `claude...` | Anthropic |
| `deepseek...` | DeepSeek |
| `gpt-*`, `o1`, `o3`, `o4` | OpenAI |

### Complete Flow (`client.py:160`)
1. Resolve model name from `config_key` (e.g., "filtering" → "gemini-3-flash-preview")
2. Resolve provider from model name
3. Budget check → may raise `BudgetExceededError` or `BudgetDegradedError`
4. Route to provider-specific implementation
5. Record usage (tokens, cost, elapsed time) in `UsageTracker`
6. Record cost in `BudgetGuard` for daily limit enforcement

### Config Keys
Each pipeline stage has a `config_key` mapping to a model:
`discovery`, `filtering`, `synthesis`, `dialogue_script`, `knowledge_summary`, `breaking_news`, `agent`

### Model Presets (`src/nexus/config/presets.py`)
| Preset | Fast model | Smart model | Cost/day |
|--------|-----------|-------------|----------|
| `free` | ollama/qwen2 | ollama/qwen2 | $0 |
| `cheap` | deepseek-chat | deepseek-chat | ~$0.01 |
| `balanced` | gemini-3-flash | gemini-3.1-pro | ~$0.05 |
| `quality` | gemini-3-flash | gemini-3.1-pro | ~$0.15 |
| `openai-cheap` | gpt-4.1-nano | gpt-4.1-mini | ~$0.03 |
| `openai-balanced` | gpt-4.1-mini | gpt-4.1 | ~$0.10 |
| `anthropic` | claude-haiku | claude-sonnet | ~$0.10 |

---

## Configuration

**`NexusConfig`** — `src/nexus/config/models.py:77`

```
NexusConfig
├── user: UserConfig {name, timezone, output_language}
├── briefing: BriefingConfig {schedule, format, style, depth, additional_languages}
├── topics: list[TopicConfig]
│   └── TopicConfig {name, priority, subtopics, source_languages,
│                    perspective_diversity, filter_threshold, scope, max_events}
├── models: ModelsConfig {discovery, filtering, synthesis, dialogue_script,
│                         knowledge_summary, breaking_news, agent}
├── audio: AudioConfig {enabled, tts_backend, tts_model, voice_host_a, voice_host_b}
├── breaking_news: BreakingNewsConfig {enabled, poll_interval_hours, threshold, wire_feeds}
├── telegram: TelegramConfig {enabled, chat_id}
├── sources: SourcesConfig {global_feeds, blocked_sources, discover_new_sources}
├── budget: BudgetConfig {daily_limit_usd, warning_threshold_usd, degradation_strategy}
└── preset: Optional[str]
```

---

## SQLite Schema (v4)

16 tables with WAL mode and foreign keys enabled.

### Core Tables

**events** — Primary knowledge unit
```sql
id, date, summary, significance (1-10), relation_to_prior, topic_slug, created_at
```

**entities** — Graph nodes (canonical names)
```sql
id, canonical_name (UNIQUE), entity_type (person|org|country|treaty|concept|unknown),
aliases (JSON array), first_seen, last_seen
```

**event_entities** — Many-to-many with role
```sql
event_id, entity_id, role (subject|object|mentioned)
```

**event_sources** — Normalized source attribution
```sql
id, event_id, url, outlet, affiliation, country, language
```

### Thread Tables

**threads** — Persistent narrative threads
```sql
id, slug (UNIQUE), headline, status (emerging|active|stale|resolved),
significance, created_at, updated_at
```

**thread_events** — Many-to-many
```sql
thread_id, event_id, added_date
```

**thread_topics** — Cross-topic threads
```sql
thread_id, topic_slug
```

**convergence** — Cross-source fact confirmation
```sql
id, thread_id, fact_text, confirmed_by (JSON array of sources)
```

**divergence** — Conflicting editorial framing
```sql
id, thread_id, shared_event, source_a, framing_a, source_b, framing_b
```

### Support Tables

**summaries** — Compressed period summaries (weekly/monthly)
```sql
id, topic_slug, period_type, period_start, period_end, text, event_count
```

**pages** — Cached LLM-generated narrative pages with TTL
```sql
id, slug (UNIQUE), title, page_type, topic_slug, content_md, generated_at,
stale_after, prompt_hash
```

**syntheses** — Daily synthesis snapshots
```sql
id, topic_slug, date, data_json (serialized TopicSynthesis)
```

**filter_log** — Full filtering decision audit trail
```sql
id, run_date, topic_slug, url, title, source_id, source_affiliation,
relevance_score, relevance_reason, passed_pass1, significance_score,
is_novel, significance_reason, passed_pass2, final_score,
outcome (accepted|rejected_relevance|rejected_significance|rejected_diversity)
```

**breaking_alerts** — Dedup for breaking news
```sql
id, headline_hash (UNIQUE), headline, source_url, significance_score, alerted_at
```

**feedback** — User briefing ratings
```sql
id, briefing_date, rating (up|down), comment
```

**usage_log** — LLM cost tracking
```sql
id, date, provider, model, config_key, input_tokens, output_tokens, cost_usd
```

**schema_version** — Migration tracking
```sql
version, applied_at
```

---

## Data Models

### ContentItem (`src/nexus/engine/sources/polling.py:14`)
RSS entry enriched through the pipeline:
```
title, url, source_id, snippet, published, full_text, language,
relevance_score, source_language, source_affiliation, source_country,
source_tier, detected_language, extraction_status, extraction_error
```

### Event (`src/nexus/engine/knowledge/events.py:19`)
Structured fact extracted from an article:
```
date, summary, sources: list[dict], entities: list[str],
relation_to_prior, significance (1-10)
```

### EntityResolution (`src/nexus/engine/knowledge/entities.py:12`)
```
raw (original string), canonical (resolved name),
entity_type (person|org|country|treaty|concept|unknown), is_new
```

### NarrativeThread (`src/nexus/engine/synthesis/knowledge.py:37`)
```
headline, events: list[Event],
convergence: list[str|dict], divergence: list[dict],
key_entities: list[str], significance (1-10),
thread_id, slug, status (emerging|active|stale|resolved)
```

### TopicSynthesis (`src/nexus/engine/synthesis/knowledge.py:51`)
The central intermediate object — all output artifacts render FROM this:
```
topic_name, threads: list[NarrativeThread],
background: list[Summary], source_balance: dict,
languages_represented: list[str], metadata: dict
```

### DialogueTurn / DialogueScript (`src/nexus/engine/audio/script.py:17`)
```
DialogueTurn: {speaker ("A"|"B"), text}
DialogueScript: {turns: list[DialogueTurn]}
```

### Summary (`src/nexus/engine/knowledge/compression.py:17`)
```
period_start, period_end, text, event_count
```

### GlobalSource (`src/nexus/engine/sources/registry.py:15`)
```
id, name, url, language, tier (A|B|C),
tags: list[str], affiliation (state|public|private|nonprofit|academic), country
```

---

## Source Registry

**Global registry:** `data/sources/global_registry.yaml` — 52 feeds across 8 languages

**Per-topic registries:** `data/sources/{topic-slug}/registry.yaml`
- `iran-us-relations`: 31 sources
- `ai-ml-research`: 9 sources
- `formula-1`: 9 sources
- `global-energy-transition`: 7 sources

Each source carries: `id`, `url`, `language`, `affiliation` (state/public/private/nonprofit/academic), `country`, `tier` (A/B/C).

Source auto-discovery: `python -m nexus sources discover <slug>` uses LLM to find new RSS feeds for a topic.

---

## Web Dashboard

**App factory:** `src/nexus/web/app.py:23` — FastAPI + Jinja2 + HTMX, Pico CSS dark theme

### Routes (11 routers)

| Route | File | Purpose |
|-------|------|---------|
| `GET /` | dashboard.py | Landing page: topic stats, active threads, recent events, source balance, cost |
| `GET /topics/{slug}` | topics.py | Topic detail: threads, events, filter stats, backstory page |
| `GET /threads/` | threads.py | Thread list with status/topic filters |
| `GET /threads/{slug}` | threads.py | Thread detail: events, convergence, divergence, deep-dive page |
| `GET /events/` | events.py | Event list with topic filter |
| `GET /events/{id}` | events.py | Event detail with sources and entities |
| `GET /entities/` | entities.py | Entity list with search and topic filter |
| `GET /entities/{id}` | entities.py | Entity profile: events, threads, related entities, profile page |
| `GET /pages/{slug}` | pages.py | Cached narrative page (markdown → HTML) |
| `GET /filters/{topic}/{date}` | filters.py | Filter audit log for a topic+date |
| `GET /sources/` | sources.py | Source balance by affiliation |
| `GET /feed.xml` | podcast.py | Podcast RSS feed for audio episodes |
| `GET /cost` | cost.py | Cost tracking page (30-day chart) |
| `GET /api/cost` | cost.py | JSON cost endpoint |
| `GET /api/cost-badge` | cost.py | HTMX badge fragment |
| `GET /settings` | settings.py | Settings page (API key status, preset) |

---

## Scheduler

**`schedule_jobs()`** — `src/nexus/scheduler/jobs.py:98`

| Job | Trigger | Function |
|-----|---------|----------|
| `daily_pipeline` | Cron at `config.briefing.schedule` (e.g., 06:00) in user timezone | Runs full pipeline → delivers briefing via Telegram |
| `breaking_news` | Interval every `poll_interval_hours` (default 3h) | Polls wire feeds → delivers alerts above threshold |

---

## Runner

**`run_all()`** — `src/nexus/runner.py:21`

Single process, single event loop:
1. Initialize `LLMClient` + `KnowledgeStore`
2. Start Telegram bot (long-polling) if configured
3. Start APScheduler with daily pipeline + breaking news jobs
4. Start FastAPI/uvicorn dashboard (blocks until shutdown)
5. Graceful shutdown: scheduler → bot → store

---

## Key Design Decisions

1. **TopicSynthesis as intermediate object** — All output artifacts (briefing, audio script, dashboard views) render FROM `TopicSynthesis`. The pipeline produces X; renderers consume X. This decouples extraction from presentation.

2. **Two-pass filtering** — Pass 1 (cheap, batch) eliminates obvious noise. Pass 2 (per-article, with event context) catches significance and novelty. This reduces LLM cost by ~70% vs single-pass.

3. **Convergence/divergence detection** — Convergence: facts confirmed by 2+ sources with different affiliations. Divergence: same event, different editorial framing. Stored per-thread, surfaced in briefings and dashboard.

4. **Entity resolution as pipeline stage** — Raw entity strings from different articles/languages are canonicalized into graph nodes. "Trump", "Donald Trump", "President Trump" → one entity. Enables relationship queries across the knowledge graph.

5. **Persistent narrative threads** — Events are grouped into threads that persist across days. Two-stage matching: entity overlap (Jaccard, no LLM) for clear matches, LLM confirmation for ambiguous cases. Lifecycle tracking: emerging → active → stale → resolved.

6. **Source affiliation tracking** — Every source carries metadata: state media, public broadcaster, private outlet, nonprofit, academic. The system enforces perspective diversity minimums and attributes claims to source affiliations in briefings.

7. **SQLite with WAL mode** — Single file, no server, portable. WAL enables concurrent reads from dashboard while pipeline writes. Foreign keys and indexes for efficient joins across the knowledge graph.

8. **Budget enforcement** — Daily spend limits with two strategies: `skip_expensive` (degrade to cheaper models) or `stop_all` (halt LLM calls). Warning threshold triggers Telegram alerts.

9. **Fixture capture/replay** — `--capture` flag saves all LLM inputs/outputs during a pipeline run. Replay mode uses captured fixtures for deterministic backtesting without API calls.

---

## Module Map

```
src/nexus/
├── __main__.py              CLI entry point
├── runner.py                Unified runner (dashboard + scheduler + bot)
├── config/
│   ├── models.py            Pydantic config models (9 sub-models)
│   └── presets.py           Model preset definitions (7 presets)
├── llm/
│   ├── client.py            Multi-provider async LLM client
│   └── cost.py              Per-model cost calculations
├── engine/
│   ├── pipeline.py          Pipeline orchestrator
│   ├── sources/
│   │   ├── polling.py       RSS polling, ContentItem model
│   │   ├── registry.py      Source registry, GlobalSource model
│   │   └── discover.py      LLM-powered source auto-discovery
│   ├── ingestion/
│   │   ├── ingest.py        Full-text fetch via trafilatura
│   │   └── dedup.py         URL normalization + dedup
│   ├── filtering/
│   │   └── filter.py        Two-pass LLM filter + perspective diversity
│   ├── knowledge/
│   │   ├── store.py         KnowledgeStore (all SQLite CRUD)
│   │   ├── schema.py        DDL for 16 tables + migrations
│   │   ├── events.py        Event model, extraction, dedup/merge
│   │   ├── entities.py      Entity resolution (LLM canonicalization)
│   │   ├── pages.py         Cached narrative pages with TTL
│   │   └── compression.py   Weekly/monthly summary compression
│   ├── synthesis/
│   │   ├── knowledge.py     TopicSynthesis builder, thread persistence
│   │   ├── threads.py       Thread matching (entity overlap + LLM)
│   │   └── renderers.py     Text briefing renderer
│   ├── audio/
│   │   ├── pipeline.py      Audio pipeline orchestrator
│   │   ├── script.py        Dialogue script generation
│   │   ├── tts.py           TTS backends (Gemini, OpenAI, ElevenLabs)
│   │   └── concat.py        Audio concatenation (pydub → MP3)
│   └── evaluation/
│       └── metrics.py       Pipeline run metrics
├── agent/
│   ├── bot.py               Telegram bot (commands + Q&A)
│   ├── qa.py                Knowledge-grounded Q&A + web search
│   ├── breaking.py          Breaking news detection
│   ├── delivery.py          Message formatting + delivery
│   ├── feedback.py          Inline keyboard feedback
│   └── web_search.py        Web search fallback for Q&A
├── scheduler/
│   └── jobs.py              APScheduler job definitions
├── web/
│   ├── app.py               FastAPI app factory
│   ├── routes/              11 route modules
│   ├── templates/           Jinja2 templates (Pico CSS + HTMX)
│   └── static/              CSS + JS assets
├── testing/
│   └── fixtures.py          Fixture capture/replay for backtesting
└── cli/
    └── setup.py             Interactive setup wizard
```
