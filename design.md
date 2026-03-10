# Agentic News Compiler — System Design (v0.2)

*Working document — March 2026*
*Intended audience: Claude Code (for build planning) + human contributors*

---

## Core Thesis

Recommendation algorithms optimize for engagement; this system optimizes for **epistemic value**. It is a personal news compiler: the user defines a spec, and the system fulfills it. Open-source, self-hosted, model-agnostic, and chat-first.

---

## Design Principles

1. **Chat is the OS.** The messaging app (Telegram, WhatsApp, etc.) is the primary interface. Dashboard and podcast are outputs, not the interface.
2. **Config as code.** All user preferences are expressed as structured YAML/JSON, human-readable, forkable, shareable.
3. **Model agnostic.** Every pipeline stage can use a different model. Users bring their own API keys and choose their cost/quality tradeoff.
4. **No centralized storage.** User data and generated artifacts live on the user's own VPS instance. The project ships code, not a platform.
5. **OpenClaw-inspired.** Local-first, skill-based extensibility, cron/heartbeat for proactive behavior, conversational configuration.
6. **User language first.** All user-facing outputs (briefings, podcast scripts, chat responses, dashboard UI, notifications, event log summaries) are generated in the user's configured output language. Internal processing (source ingestion, relevance scoring) operates in whatever language the source material is in. The system never forces English on the user; a Mandarin-speaking user gets Mandarin briefings synthesized from English, Farsi, and Chinese sources alike.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    USER DEVICES                          │
│  Telegram / WhatsApp / Signal    Podcast App (RSS)      │
│  PWA (dashboard, chat, timeline) Audio in-chat           │
└──────────────┬──────────────────────┬───────────────────┘
               │                      │
               ▼                      ▼
┌─────────────────────────────────────────────────────────┐
│                     VPS INSTANCE                         │
│                                                          │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────┐  │
│  │    SHELL    │  │    ENGINE    │  │     AGENT      │  │
│  │  (Frontend) │  │   (Batch)    │  │ (Interactive)  │  │
│  │             │  │              │  │                │  │
│  │ • PWA       │  │ • Cron jobs  │  │ • Chat handler │  │
│  │ • Podcast   │  │ • Discovery  │  │ • Q&A          │  │
│  │   RSS feed  │  │ • Synthesis  │  │ • Config edits │  │
│  │ • Dashboard │  │ • TTS/Audio  │  │ • Live lookups │  │
│  │ • Timeline  │  │ • Knowledge  │  │                │  │
│  │             │  │   layer ops  │  │                │  │
│  └─────────────┘  └──────────────┘  └────────────────┘  │
│                          │                  │            │
│                          ▼                  ▼            │
│               ┌──────────────────────────────┐           │
│               │       SHARED STATE           │           │
│               │                              │           │
│               │ • User config (YAML/JSON)    │           │
│               │ • Knowledge layer (per-topic) │           │
│               │ • Source registry             │           │
│               │ • Generated artifacts        │           │
│               │   (briefings, audio, etc.)   │           │
│               └──────────────────────────────┘           │
└─────────────────────────────────────────────────────────┘
```

---

## Three Components

### 1. Engine (Batch / Cron)

The heavy-lifting backend. Runs on schedule (default: overnight, configurable).

**Daily pipeline stages:**

| Stage | Description | Models Used |
|-------|-------------|-------------|
| **Source Polling** | Fetch new content from all registered sources (RSS, web) per topic | None (HTTP fetches) |
| **Active Discovery** | Periodic search for new high-quality sources per topic, including cross-language queries | Cheap/fast model for quality scoring |
| **Content Ingestion** | Fetch full articles, extract text, preserve original language | None (scraping/parsing) |
| **Relevance Filtering** | Score content against user's topic definitions, discard noise | Cheap classifier model (GPT-4o-mini, Gemini Flash) |
| **Knowledge Layer Update** | Append new events to per-topic logs (in user's output language), run periodic summarization/compression | Mid-tier model for summarization |
| **Briefing Synthesis** | Generate the daily briefing from filtered content + knowledge context, in user's output language. Foreign-language sources are read natively by the model — no separate translation step. | Strong model (Claude Sonnet, GPT-4o) |
| **Script Generation** | Convert briefing into two-host dialogue script (if podcast format selected), in user's output language | Strong model with dialogue prompt |
| **Audio Generation** | Voice the script with two TTS voices, concatenate into episode | TTS engine (Kokoro local / OpenAI / ElevenLabs) |
| **Artifact Publishing** | Save briefing markdown, update RSS feed, push notification to user via chat | — |

**Key design decision — no standalone translation pipeline.** Foreign-language sources (Farsi, Arabic, Mandarin, etc.) are ingested in their original language and passed directly to the synthesis model. The synthesis model reads them natively and incorporates their perspectives into the user-language briefing with proper attribution. This avoids lossy intermediate translation and preserves nuance. Event log summaries are written in the user's output language for searchability.

**Estimated cost per daily run:** $0.50–$3.00 depending on topic count, model choices, and TTS provider.

### 2. Shell (Frontend / PWA)

A lightweight web UI served from the VPS instance. Accessed via browser on phone or desktop. **All UI text, labels, and generated content rendered in the user's configured output language.**

**Views:**

- **Daily Brief** — Today's briefing as readable text in user's output language. Each section expandable with source links. Play button for audio at top.
- **Topic Timelines** — Per-topic chronological thread of events from the knowledge layer. Tap any node to expand. Phone-friendly timeline format (not a graph).
- **Podcast Player** — In-app audio player for users who don't use a podcast app.
- **Chat** — Side panel (desktop) or separate tab (mobile) for the interactive agent.

**External distribution:**

- **Private RSS feed** — Served by the instance. User subscribes in Apple Podcasts, Overcast, Pocket Casts, or Podcast Addict. Episodes appear automatically.
- **In-chat audio** — Agent sends audio file directly via Telegram/WhatsApp for zero-setup listening.

**Note:** Spotify does NOT support adding private RSS feeds as a listener. Not a viable distribution channel.

### 3. Agent (Interactive)

The conversational interface. Lives in the user's messaging app. Shares the engine's knowledge layer but responds in real-time. **Communicates in the user's configured output language.**

**Capabilities:**

- **Contextual Q&A** — When user asks about a briefing topic, the agent already has the full source material, event log, and historical background loaded.
- **Configuration** — "Dial back oil pricing coverage, add more renewable energy" → updates the YAML config.
- **Live lookups** — "What's the current oil price?" → web search or API call. "Is this restaurant open?" → external query. Not limited to the knowledge layer.
- **Temporary subscriptions** — "I'm traveling to Japan next week, add Japan coverage for 10 days" → scoped topic addition with auto-expiry.
- **Background/history** — "Give me the history of Iran-US tensions" → blends LLM training data (deep history) with accumulated event log (recent developments).

**Design principle: the agent is NOT a naive RAG chatbot.** It has distinct interaction modes based on context:
- Reading a briefing item → agent has that item's full context pre-loaded
- Cold open question → agent infers intent and decides whether to use knowledge layer, web search, or both
- Config request → agent modifies structured config and confirms

---

## Source Discovery (LOCKED)

### Three-tier source architecture

**Tier A — Official / Institutional (ship with v1)**
Government statements, state media, established wire services. These are ground truth for "what actually happened" and ground truth for "what each side claims happened." Includes:
- Wire services: Reuters, AP, AFP
- State/institutional media from relevant perspectives: Xinhua, IRNA, TASS, Al Jazeera, BBC, NHK
- Government press offices, ministry statements, UN/IAEA releases

The system does NOT editorialize about source bias at the discovery stage. It surfaces diversity. The synthesis step handles framing: "according to Iranian state media IRNA..." vs "Reuters reports..."

**Tier B — Regional / Specialist (ship with v1, grows over time)**
Domain-specific outlets, regional newspapers, think tanks, specialist blogs. Examples for Iran-US topic: Al-Monitor, War on the Rocks, Shargh (Farsi), Hamshahri (Farsi).

Discovery mechanism:
- **Seed lists** shipped with community-contributed topic templates
- **Active discovery** via periodic search queries (in English AND relevant foreign languages per topic config)
- **LLM quality scoring** of discovered sources: Is this a primary source? Domain expert? Institutional outlet? Or content farm rehash?
- **User feedback loop**: "this source was great, find more like it" / "this was garbage" adjusts source quality scores

**Tier C — Social / OSINT (deferred to v2+)**
Reddit, X/Twitter, Telegram channels. Design the source registry data model to accommodate these (a source is a source whether it's RSS, an X account, or a Telegram channel), but don't build ingestion for them in v1.

### Source Registry data model

```yaml
# sources/iran-us-tensions/registry.yaml
sources:
  - id: "reuters-world"
    type: rss
    url: "https://feeds.reuters.com/reuters/worldNews"
    tier: A
    quality_score: 0.95  # updated by user feedback + engagement
    language: en
    added: "2026-03-10"
    added_by: seed  # or "discovery" or "user"

  - id: "shargh-daily"
    type: rss
    url: "https://www.sharghdaily.com/rss"
    tier: B
    quality_score: 0.82
    language: fa
    added: "2026-03-12"
    added_by: discovery
    notes: "Tehran-based reformist newspaper, good for domestic political perspective"

  # Future v2 example:
  # - id: "arms-control-wonk-x"
  #   type: x_account
  #   handle: "@ArmsControlWonk"
  #   tier: B
  #   quality_score: 0.88
  #   language: en
```

---

## Translation Strategy (LOCKED)

**No standalone translation pipeline.** This is a deliberate architectural decision.

**Rationale:** The system's output is a synthesized briefing, not translated articles. Modern LLMs (Claude Sonnet ranked #1 in 9/11 language pairs at WMT24) can read foreign-language source material natively and incorporate its perspective into a user-language briefing. A separate translation step introduces lossy intermediate representations and can distort nuance in political/cultural content — exactly the content this system is designed to surface.

**How it works in practice:**
1. Farsi editorial from Shargh Daily is ingested as raw Farsi text
2. At synthesis, the model receives it alongside English-language sources
3. The briefing output (in user's language) says: "Shargh Daily, a Tehran-based reformist newspaper, argues that the sanctions will primarily hurt ordinary Iranians while strengthening hardliner positions..."
4. The knowledge layer event log stores a summary in the user's output language

**On-demand translation:** If a user asks the agent "can you translate that Shargh article for me?", the agent performs a full contextual translation on the fly. This is an interactive feature, not a batch pipeline step.

**Model choice for cross-language synthesis:** Use the strongest available model (Claude Sonnet / GPT-4o tier). This is not the place to save money with a cheap model — nuance preservation across languages is the core differentiator.

---

## Breaking News Detection (LOCKED)

**Lightweight polling loop, separate from the daily batch engine.**

**Design:**
- Separate cron job running every 2–4 hours (user-configurable interval)
- Checks a small, curated set of high-velocity sources per topic:
  - Wire service RSS feeds (Reuters, AP, AFP — update within minutes of major events)
  - Optionally: user-specified X/Twitter accounts (future, when Tier C sources are supported)
- For each polling cycle:
  1. Fetch latest headlines from monitored feeds (HTTP only, cheap)
  2. One small LLM call: "Do any of these headlines represent a significant development in [user's tracked topics]? Score each 1–10."
  3. If any item scores above threshold (user-configurable, default 7+): push alert via chat

**Alert format (in user's output language):**
> ⚡ Breaking: [2-sentence summary with source attribution]. Full coverage in tomorrow's briefing.

**What it does NOT do:**
- No full synthesis on breaking items
- No audio generation
- No knowledge layer update (that happens in the next daily run with fuller context)
- No real-time streaming APIs (cost and complexity not justified for v1)

**Cost:** < $0.10/day (RSS fetches are free; one small LLM call per cycle)

---

## Attribution Policy (LOCKED)

**Attribution is a hard requirement in the synthesis prompt, not optional.**

- Every factual claim in briefing output must trace to a named source
- Text briefings include hyperlinks to original articles
- Podcast scripts use verbal attribution: "Reuters is reporting that..." / "An editorial in Shargh Daily argues..."
- Knowledge layer event logs always store source URLs
- The two-host dialogue format naturally accommodates attribution: "So I was reading this piece in Al-Monitor, and they're making a really interesting argument about..."
- Sources in non-English languages are attributed with context: "Shargh Daily, a Tehran-based reformist newspaper" — giving the user enough to understand the source's perspective even if they can't read the original

---

## Evaluation Strategy (LOCKED for v1)

**Developer-as-first-user testing, minimal feedback loop.**

- After each briefing, rate via the agent: thumbs up/down + optional free-text feedback
- System logs per briefing: topics covered, sources used, source languages, user rating, follow-up questions asked (engagement proxy), "I already knew this" signals (redundancy), "you missed X" signals (coverage gaps)
- Agent periodically asks: "Is there anything you feel like I'm missing?" (configurable frequency)
- After 2+ weeks of daily data: enough signal to evaluate source discovery quality, synthesis quality, and knowledge layer continuity
- No automated quality metrics in v1; human judgment drives iteration

---

## Knowledge Layer

The critical differentiator. Without this, the system is just a daily RSS-to-LLM pipe.

### Three Tiers

**Tier 1 — LLM Training Data (free, static)**
Deep history, established facts. China-Taiwan since 1945, the Oslo Accords, Cold War dynamics. No storage needed; the model knows this.

**Tier 2 — Accumulated Observation Log (core, must-build)**
Per-topic, append-only structured log of observed events. **All summaries stored in user's output language.**

```yaml
# knowledge/iran-us-tensions/events.yaml
- date: 2026-02-28
  summary: "US imposes new sanctions on Iranian petroleum exports"
  sources:
    - url: "https://reuters.com/..."
      language: en
      outlet: "Reuters"
    - url: "https://irna.ir/..."
      language: fa
      outlet: "IRNA"
  entities: ["US Treasury", "Iran NIOC"]
  relation_to_prior: "Escalation from Jan diplomatic freeze"
  significance: 8  # 1-10, assigned by synthesis model

- date: 2026-03-01
  summary: "Iran announces enrichment increase to 60%"
  sources:
    - url: "https://irna.ir/..."
      language: fa
      outlet: "IRNA"
  entities: ["AEOI", "IAEA"]
  relation_to_prior: "Direct retaliation to Feb 28 sanctions"
  significance: 9
```

**Compression schedule:**
- Days 1–14: Full granularity (individual events)
- Weeks 3–8: Weekly rollup summaries
- Months 3+: Monthly narrative summaries

When generating a briefing, context is assembled as: `[monthly summaries] + [weekly summaries] + [recent events]` → fits in a single context window.

**Tier 3 — Cross-topic Relationships (deferred to v2+)**
Let the LLM infer cross-topic connections at synthesis time from combined topic logs. No explicit graph needed for v1.

### Cold Start / Backfill

When a user adds a new topic, the engine runs a one-time backfill pass:
1. Web search for the topic over the past 2–4 weeks
2. Construct initial event log from search results
3. Generate a "state of play" summary in user's output language

Result: the first briefing already has narrative continuity.

---

## User Configuration

### Structure

```yaml
# config.yaml
user:
  name: "Tristan"
  timezone: "America/Denver"
  output_language: "en"  # ALL user-facing content in this language
  # Examples: "en", "zh-CN", "fa", "ar", "de", "ja"

briefing:
  schedule: "06:00"  # local time
  duration_target_minutes: 30
  format: "two-host-dialogue"  # or "solo-narrator", "executive-summary"
  style: "analytical"  # inferred during onboarding probing
  depth: "detailed"    # vs "overview"

breaking_news:
  enabled: true
  poll_interval_hours: 3
  threshold: 7  # 1-10 significance score
  channel: "telegram"

topics:
  - name: "Iran-US Relations"
    priority: high
    subtopics: ["sanctions", "nuclear program", "regional proxies"]
    source_languages: ["en", "fa"]  # languages to FIND sources in
    perspective_diversity: high
    # source_languages != output_language
    # Sources are discovered in these languages; output is always user.output_language

  - name: "Renewable Energy Policy"
    priority: medium
    subtopics: ["US IRA implementation", "EU green deal", "grid storage"]
    source_languages: ["en", "de"]
    perspective_diversity: medium

  - name: "AI/ML Research"
    priority: low
    subtopics: ["agents", "benchmarks", "open source models"]
    source_languages: ["en", "zh"]
    perspective_diversity: low

models:
  # Each pipeline stage independently configurable
  discovery: "gpt-4o-mini"
  filtering: "gpt-4o-mini"
  synthesis: "claude-sonnet-4-20250514"
  dialogue_script: "claude-sonnet-4-20250514"
  knowledge_summary: "gpt-4o-mini"
  breaking_news: "gpt-4o-mini"
  agent: "claude-sonnet-4-20250514"
  tts_engine: "kokoro-local"  # or "openai-tts", "elevenlabs"
  tts_voice_host_a: "af_heart"
  tts_voice_host_b: "am_adam"

sources:
  global_feeds:
    - "https://feeds.reuters.com/reuters/topNews"
    - "https://www.aljazeera.com/xml/rss/all.xml"
  blocked_sources: []
  discover_new_sources: true
  discovery_interval_days: 7

notifications:
  morning_briefing: true
  breaking_news: true
  midday_digest: false
  channel: "telegram"

feedback:
  ask_for_rating: true
  periodic_check_in: "weekly"
```

### Community Sharing

Configs are just files. Users can export, import, and fork. Topic templates (shipped with the project and community-contributed) include: topic definition, subtopics, recommended source languages, Tier A and Tier B seed source lists.

---

## Onboarding Flow

Chat-based, conversational, in user's preferred language. **Probing, not surveying.**

The onboarding agent:
1. Detects or asks the user's preferred language
2. Explores professional information needs through conversation
3. Explores personal curiosity and "what are you missing?" gaps
4. Probes for cross-language interest
5. Calibrates style by generating two sample briefing paragraphs and asking which is more useful
6. Asks for podcast format preference (two-host dialogue vs. solo narrator)
7. Writes the full config.yaml and confirms with the user
8. Kicks off the first backfill run

All onboarding conversation happens in the user's chosen output language.

---

## Audio / Podcast Pipeline

### Two-Host Dialogue Generation

```
[Briefing content + knowledge context]
            │
            ▼
   ┌─────────────────────┐
   │  Script LLM          │  Generates structured dialogue in user's
   │  (strong model)      │  output language: Host A lines, Host B lines,
   │                      │  stage directions for tone
   └──────────┬──────────┘
              │
              ▼
   ┌─────────────────────┐
   │  Podcastfy /         │  Voices each line with distinct TTS voice,
   │  Custom TTS Pipeline │  concatenates, adds transitions
   └──────────┬──────────┘
              │
              ▼
   ┌─────────────────────┐
   │  MP3 artifact        │ → RSS feed, in-chat, PWA player
   └─────────────────────┘
```

### Tooling

- **Primary:** Podcastfy (open-source, Apache 2.0, multi-TTS, multi-language, CLI + Python API)
- **Upgrade path:** Google Podcast API (standalone API, GA with allowlist)
- **Fallback:** Custom pipeline with structured LLM output + raw TTS API calls
- **User option:** Solo narrator mode (skip dialogue generation, straight TTS on briefing text)

### TTS language note

TTS voice selection must match user's output language. Kokoro supports 8 languages; ElevenLabs supports 32; OpenAI supports ~13. Flag mismatches during onboarding.

### Distribution

| Channel | How | Setup Effort |
|---------|-----|-------------|
| In-chat audio | Agent sends MP3 via Telegram/WhatsApp | Zero (default) |
| PWA player | Audio player in dashboard | Zero (built-in) |
| Podcast app (iOS) | Private RSS → Apple Podcasts / Overcast | User adds URL once |
| Podcast app (Android) | Private RSS → Podcast Addict / AntennaPod | User adds URL once |
| Spotify | ❌ Not supported (no private RSS import) | — |

---

## Deployment

### Target: One-click cloud deploy

- **Railway / Fly.io / Coolify** — Click deploy button on GitHub, paste API keys, running in 5 minutes.
- **Docker Compose** — For self-hosters: `docker compose up` on any VPS.
- **Nix flake** — For the nix-pilled.

### Minimum VPS requirements (estimated)

- 2 vCPU, 4GB RAM (for Kokoro TTS local inference + pipeline)
- 20GB storage (knowledge layer + audio artifacts)
- $5–10/month on most cloud providers

---

## Cost Model (per user, per month)

| Tier | LLM Costs | TTS Costs | VPS | Total |
|------|-----------|-----------|-----|-------|
| **Budget** (small models, Kokoro, 3 topics) | ~$10–15 | ~$0 (local) | $5 | **~$15–20** |
| **Standard** (mid-tier models, OpenAI TTS, 5 topics) | ~$25–40 | ~$15 | $5–10 | **~$45–65** |
| **Premium** (frontier models, ElevenLabs, 8+ topics) | ~$50–80 | ~$30–50 | $10 | **~$90–140** |

---

## Tech Stack (Tentative)

| Layer | Technology | Rationale |
|-------|-----------|-----------|
| Runtime | Python 3.11+ | Podcastfy is Python, LLM SDKs are Python-first |
| Chat integration | python-telegram-bot | Telegram is lowest-friction bot platform |
| Web framework | FastAPI | Serves PWA, RSS feed, agent API |
| Frontend | Lightweight PWA (Preact or vanilla) | Minimal JS, works offline for cached briefings |
| TTS | Kokoro (local default) / OpenAI / ElevenLabs | Modular, user-configurable |
| Dialogue generation | Podcastfy | Open-source NotebookLM-style two-host pipeline |
| Knowledge store | Flat files (YAML/JSON per topic) | Simple, portable, git-friendly |
| Scheduling | APScheduler or system cron | Engine pipeline + breaking news poller |
| Containerization | Docker Compose | One-command deploy |
| Config management | Pydantic models + YAML | Type-safe config with human-readable storage |

---

## Open Questions (deferred to v2+)

1. Cross-topic relationship inference (Tier 3 knowledge)
2. Social media / OSINT source ingestion (Tier C — Reddit, X, Telegram)
3. Multi-user shared instances (family/team briefings)
4. Collaborative source curation / community source registry
5. Historical deep-dive generation (on-demand long-form content)
6. Voice cloning (user's preferred narrator voice)
7. Config marketplace / sharing platform

---

## Build Plan — Guidance for Claude Code

This section describes the recommended build order. Each phase produces a usable, testable artifact. The first user/tester is the developer (Tristan). The system should be functional and personally useful at the end of Phase 1.

---

### Phase 0: Project Scaffolding

**Goal:** Repo structure, dependency management, config system, basic CLI.

**Deliverables:**
- Python project with pyproject.toml (or Poetry/uv)
- Pydantic models for the full config schema (mirror the config.yaml structure above exactly)
- Config loader: reads YAML, validates with Pydantic, provides typed access everywhere
- Modular project structure:

```
src/
  config/          # Pydantic models, YAML loader
  engine/          # Daily pipeline stages
    sources/       # Source polling, discovery, registry management
    ingestion/     # Content fetching, text extraction
    filtering/     # Relevance scoring
    knowledge/     # Event log CRUD, summarization, compression
    synthesis/     # Briefing generation (the big prompt)
    audio/         # Script generation, TTS integration, Podcastfy wrapper
    publishing/    # RSS feed generation, artifact storage
  agent/           # Interactive chat handler, Telegram integration
  breaking/        # Breaking news poller
  shell/           # FastAPI app, PWA static files, RSS server
  llm/             # Model abstraction layer (provider-agnostic wrapper)
  utils/           # Shared utilities (logging, date handling, etc.)
data/
  config.yaml      # User configuration
  knowledge/       # Per-topic directories with events.yaml + summaries
  sources/         # Per-topic source registries
  artifacts/       # Generated briefings (markdown), audio files (mp3)
  feedback/        # Logged user ratings and feedback
```

- **LLM abstraction layer (build this EARLY, it's load-bearing):** A thin wrapper that takes a provider name + model name from config and routes to the right SDK (Anthropic, OpenAI, Google). Every pipeline stage calls through this layer, never directly to a provider SDK. Interface should be something like:
  ```python
  response = await llm.complete(
      config_key="synthesis",  # looks up provider+model from config.models.synthesis
      system_prompt="...",
      user_prompt="...",
      response_format=None,  # or a Pydantic model for structured output
  )
  ```
- Basic CLI entry points: `./run.py engine`, `./run.py agent`, `./run.py breaking`, `./run.py setup`
- Docker Compose skeleton with a single service
- .env.example with placeholders for API keys

---

### Phase 1: Engine Core (Daily Pipeline — Text Only)

**Goal:** Given a config with topics, produce a daily markdown briefing. No audio, no chat, no frontend. Just: cron runs, fetches sources, filters, updates knowledge layer, generates briefing, saves to file. **This is the core value proposition and the critical path.**

**Build in this order:**

**Step 1 — Source polling.** Given a source registry YAML with RSS feeds, fetch latest items. Use feedparser for RSS. Parse titles, URLs, publication dates, content snippets. Store as a list of raw `ContentItem` objects (Pydantic model). Handle gracefully: feeds that are down, feeds with bad XML, feeds that haven't updated.

**Step 2 — Content ingestion.** For each `ContentItem`, fetch the full article text from the URL. Use trafilatura (preferred) or newspaper3k for article extraction. Preserve original language (do NOT translate at this stage). Store extracted text on the `ContentItem`. Handle: paywalls (skip gracefully), PDFs (extract text with pymupdf or similar), non-HTML (skip), rate limiting (respect robots.txt, add delays).

**Step 3 — Relevance filtering.** For each ingested article, one LLM call via the abstraction layer using the cheap model (`config.models.filtering`). Prompt: "Is this article relevant to the topic '[topic name]' with subtopics [subtopics]? Score 1-10 and explain briefly." Filter by threshold (default: 5+). This is the first real LLM integration point — confirm the abstraction layer works end-to-end here.

**Step 4 — Knowledge layer event logging.** For articles that pass filtering, extract structured event data via LLM call. The prompt receives: the article text (in original language), the topic definition, and the last ~10 events from the existing log (for continuity). Output (structured/JSON): date, summary in user's output language, source info, entities mentioned, relation to prior events, significance score 1-10. Append to the topic's `events.yaml`.

**Step 5 — Knowledge layer compression.** Implement rollup logic as a maintenance task:
- Scan events older than 14 days → group by week → one LLM call per week to produce a weekly summary paragraph (in user's output language). Store in `weekly_summaries.yaml`.
- Scan weekly summaries older than 2 months → group by month → one LLM call to produce monthly narrative. Store in `monthly_summaries.yaml`.
- Old granular events can be archived (moved to `events_archive/`) but not deleted.
- This runs weekly, not daily. Schedule separately.

**Step 6 — Briefing synthesis.** The main event. Assemble the context window:
- Monthly summaries for each topic (background)
- Weekly summaries for each topic (recent context)
- Today's granular events for each topic (what's new)
- Full text of the highest-significance articles from today (in original languages — this is where cross-language synthesis happens)

One large LLM call to the strong synthesis model (`config.models.synthesis`). The prompt MUST:
- Specify the user's output language explicitly
- Require source attribution for every factual claim
- Organize sections by topic, ordered by priority from config
- Match the user's configured style ("analytical", "overview", etc.) and depth
- Integrate cross-language sources seamlessly with contextual attribution
- Reference the knowledge layer context to provide narrative continuity ("This marks the third round of sanctions since...")

Output: a structured markdown document saved to `data/artifacts/briefings/YYYY-MM-DD.md`.

**Step 7 — Topic backfill.** When a new topic appears in config that has no existing knowledge layer:
- Run web search queries for the topic (in all configured source_languages)
- Extract events from the top results
- Build an initial event log spanning the past 2-4 weeks
- Generate an initial "state of play" summary
- This makes the first briefing immediately useful

**Test criterion for Phase 1:** Run the full pipeline manually for 3 topics over 7 consecutive days. Read every briefing. Are they informative? Do they have narrative continuity from day to day? Are cross-language sources incorporated meaningfully? Is attribution consistent? Iterate on the synthesis prompt until quality is satisfactory.

---

### Phase 2: Audio Pipeline

**Goal:** Convert the text briefing into a two-host podcast episode.

**Step 1 — Script generation.** LLM call (`config.models.dialogue_script`) that takes the briefing markdown and produces a structured dialogue script. Output format (JSON array):
```json
[
  {"speaker": "A", "text": "So there's been a really interesting development overnight with OPEC...", "tone": "curious"},
  {"speaker": "B", "text": "Right, and this is actually connected to what we talked about yesterday...", "tone": "analytical"},
  ...
]
```
The prompt needs heavy iteration. Key qualities: natural hooks and transitions between topics, Host B asks clarifying questions the listener would think of, occasional disagreement or pushback, verbal source attribution woven naturally into dialogue, matching user's style preference, entirely in user's output language.

**Step 2 — TTS integration.** Wire up Podcastfy (or build a thin custom wrapper) to voice the script. Map Host A → voice 1, Host B → voice 2. Ensure TTS voices match user's output language. Concatenate segments with brief pauses between speaker turns. Output: MP3 file saved to `data/artifacts/audio/YYYY-MM-DD.mp3`.

**Step 3 — Solo narrator fallback.** Simpler path: skip script generation, run briefing markdown directly through single-voice TTS. Useful for users who prefer speed/efficiency over conversational format.

**Step 4 — RSS feed.** FastAPI endpoint serving valid podcast RSS XML. Each daily episode is an `<item>` with `<enclosure>` pointing to the MP3. User subscribes to `https://their-instance/feed.xml` in their podcast app. Include proper podcast metadata (title, description, artwork).

**Test criterion:** Generate episodes for 5 consecutive days. Listen to each in full. Does the dialogue feel natural? Are transitions smooth? Is attribution clear in spoken form? Iterate on the script generation prompt.

---

### Phase 3: Agent (Interactive Chat)

**Goal:** A Telegram bot for configuration, Q&A, briefing delivery, and feedback.

**Step 1 — Telegram bot scaffolding.** Basic bot using python-telegram-bot. Message handler routes incoming text to agent logic. Supports sending text, audio files, and inline buttons (for thumbs up/down).

**Step 2 — Agent core.** LLM-powered conversational agent (`config.models.agent`). System prompt loaded dynamically with: user config summary, today's briefing content, recent knowledge layer events for all tracked topics. The agent answers questions about briefings, provides historical context, explains connections between events.

**Step 3 — Config modification via chat.** Agent parses natural language config requests ("add a topic about renewable energy", "change briefing time to 7 AM", "less coverage of oil pricing"). Updates config.yaml programmatically. Always confirms changes with user before writing.

**Step 4 — Live lookups.** When the agent detects a question requiring current data (prices, weather, store hours, etc.), it triggers a web search (integrate Tavily, SerpAPI, or similar). Returns results inline.

**Step 5 — Briefing delivery.** When daily engine run completes, agent pushes to Telegram: short text summary + audio file attachment. Followed by thumbs up/down prompt.

**Step 6 — Feedback collection.** Inline keyboard buttons for rating. Free-text feedback captured. All logged to `data/feedback/`.

**Step 7 — Onboarding flow.** The `./run.py setup` command initiates the conversational onboarding sequence through Telegram. Agent probes the user, builds config, kicks off first backfill.

---

### Phase 4: Breaking News Poller

**Goal:** Push alerts for significant developments between daily briefings.

**Step 1 — Polling loop.** APScheduler job running every N hours (from config). Fetches headlines from wire service RSS for each tracked topic.

**Step 2 — Relevance scoring.** One cheap LLM call with batch of headlines: "Score each 1-10 for significance relative to [topics]."

**Step 3 — Alert dispatch.** Items above threshold get a 2-sentence summary, pushed via Telegram agent.

**Step 4 — Deduplication.** Track alerted item hashes to avoid repeats across polling cycles.

---

### Phase 5: Shell (PWA Dashboard)

**Goal:** Web UI for deep-dive sessions on desktop or phone.

**Step 1 — FastAPI backend.** Endpoints: GET current briefing, GET topic timeline (paginated events), GET past briefings, WebSocket for agent chat.

**Step 2 — PWA frontend.** Three views: Daily Brief (rendered markdown + embedded audio player), Topic Timelines (chronological event list, expandable nodes), Chat (agent interface). All UI text in user's output language. Minimal framework — Preact or vanilla JS with Tailwind.

**Step 3 — Service worker.** Cache latest briefing + recent timelines for offline reading.

---

### Phase 6: Deployment & Distribution

**Goal:** Anyone can deploy with minimal effort.

**Step 1 — Docker Compose.** Full multi-service file: engine + agent + shell + Kokoro (optional GPU). Env vars for API keys.

**Step 2 — One-click deploy.** Railway/Fly.io/Render deploy button on GitHub README.

**Step 3 — Setup wizard.** `./run.py setup` validates API keys, runs onboarding via Telegram, writes config, runs first backfill, confirms everything works.

**Step 4 — Documentation.** README: what this is, 5-min quickstart, config reference, model recommendations by budget tier, contributing guide.

---

### Phase 7: Source Discovery Automation

**Goal:** System gets smarter about finding sources over time.

**Step 1 — Active discovery.** Periodic search queries per topic in all configured source_languages. LLM scores discovered sources. High-scoring sources added to registry with `added_by: discovery`.

**Step 2 — Quality tracking.** Per-source metrics: relevance filter pass rate, citation frequency in positively-rated briefings. Decay score for consistently irrelevant sources.

**Step 3 — User source management.** Via agent: "add this feed", "block this source", "find more like [X]".

---

### Phase Dependencies

```
Phase 0 (Scaffolding)
   │
   ▼
Phase 1 (Engine Core) ◄── CRITICAL PATH, build first
   │
   ├──► Phase 2 (Audio)     ─── can build in parallel
   ├──► Phase 3 (Agent)     ─── can build in parallel
   ├──► Phase 5 (Shell)     ─── can build in parallel
   │
   │    Phase 3 (Agent)
   │       │
   │       ▼
   │    Phase 4 (Breaking News) ─── needs agent for push delivery
   │
   ▼
Phase 6 (Deployment)        ─── after Phases 1-5 are stable
   │
   ▼
Phase 7 (Discovery)         ─── enhancement, build anytime after Phase 1
```

**Phases 2, 3, and 5 can be built in parallel after Phase 1.** Phase 1 is the critical path. If Phase 1 produces good briefings, everything else is distribution and interface.

---

## Project Name

TBD. Considerations:
- Should convey "personal," "open," and "signal/intelligence"
- Avoid anything that sounds like a news aggregator (trust baggage)
- A good name makes the concept immediately legible

---

*This document is a living design spec. Decisions marked LOCKED reflect deliberate choices made during design conversations and should not be changed without discussion. Everything else is open for revision during implementation.*
