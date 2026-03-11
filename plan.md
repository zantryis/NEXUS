# NEXUS Expansion Plan — Three Phases

## Phase 1: First-Run Experience
*Goal: One key, one command, see something useful in 5 minutes.*

### 1.1 Model Presets (`src/nexus/config/models.py` + `config/presets.py`)
Add a `preset` field to `NexusConfig` that auto-populates `ModelsConfig`:

| Preset | Filtering | Synthesis | Dialogue | Agent | Required Key | Est. Cost/day |
|--------|-----------|-----------|----------|-------|-------------|---------------|
| `free` | ollama/qwen2 | ollama/qwen2 | ollama/qwen2 | ollama/qwen2 | None (local) | $0 |
| `cheap` | deepseek-chat | deepseek-chat | deepseek-chat | deepseek-chat | DEEPSEEK_API_KEY | ~$0.01 |
| `balanced` | gemini-3-flash-preview | gemini-3-flash-preview | gemini-3.1-pro-preview | gemini-3-flash-preview | GEMINI_API_KEY | ~$0.05 |
| `quality` | gemini-3-flash-preview | gemini-3.1-pro-preview | gemini-3.1-pro-preview | gemini-3.1-pro-preview | GEMINI_API_KEY | ~$0.15 |

**Changes:**
- New file: `src/nexus/config/presets.py` — dict of preset name → `ModelsConfig` values
- Edit `models.py`: Add `preset: Optional[str] = None` to `NexusConfig`; if set, fill `ModelsConfig` from presets (user overrides still win)
- Edit `config.example.yaml`: Add `preset: balanced` with commented-out `models:` block

### 1.2 Ollama Provider (`src/nexus/llm/client.py`)
Add Ollama as a fourth provider for zero-cost local inference:
- Provider resolution: model names starting with `ollama/` → Ollama
- Use `httpx` (already a dev dep) to hit `http://localhost:11434/api/chat` (OpenAI-compatible endpoint)
- No API key needed; fail gracefully with clear error if Ollama isn't running
- Token tracking: Ollama returns usage in response; record with `$0` cost

### 1.3 Cost Tracking (`src/nexus/llm/cost.py`)
The `UsageTracker` already tracks tokens by provider/model/config_key. Add:
- New file: `src/nexus/llm/cost.py` — cost-per-token table by model
- New method on `UsageTracker`: `cost_summary() -> dict` returning `{provider: $X.XX, total: $X.XX, by_topic: {...}}`
- Pricing table (hardcoded, easy to update):
  ```python
  PRICING = {
      "gemini-3-flash-preview": {"input": 0.10, "output": 0.40},  # per 1M tokens
      "gemini-3.1-pro-preview": {"input": 1.25, "output": 5.00},
      "deepseek-chat": {"input": 0.14, "output": 0.28},
      "deepseek-reasoner": {"input": 0.55, "output": 2.19},
      "ollama/*": {"input": 0, "output": 0},
  }
  ```
- Persist daily cost to `knowledge.db` (new `usage_log` table: date, provider, model, config_key, input_tokens, output_tokens, cost_usd)

### 1.4 Setup Wizard (`src/nexus/cli/setup.py`)
Interactive CLI wizard triggered by `python -m nexus setup` or auto-triggered on first run if no `data/config.yaml` exists:

1. **Pick a preset**: free / cheap / balanced / quality (with cost estimates)
2. **Paste API key** (if needed): validate with a test call
3. **Pick starting topics**: show available registries (Iran-US, AI, F1, Energy) + "custom"
4. **Generate config.yaml** and `.env`
5. Print: "Run `python -m nexus run` to start. Dashboard at http://localhost:8000"

**Changes:**
- New file: `src/nexus/cli/setup.py`
- Edit `src/nexus/__main__.py`: Add `setup` subcommand

### 1.5 Minimal Web Dashboard (`src/nexus/web/`)
The routes already exist. Build a functional but minimal frontend:

**Dashboard home (`/`):**
- Today's briefing (rendered markdown)
- Topic cards with event counts and latest thread headlines
- Cost meter: "Today: $0.03 | This week: $0.18"
- System status: last run time, next scheduled run, active topics

**Tech stack:** Keep it simple — Jinja2 templates + HTMX + Pico CSS (classless, ~10kb). No build step, no npm.

**New/updated templates:**
- `templates/dashboard.html` — Main page with briefing + topic cards + cost
- `templates/components/cost_badge.html` — Reusable cost display
- `templates/base.html` — Layout with nav (Topics, Threads, Sources, Cost)
- Update existing topic/thread/event templates with consistent styling

**New route:**
- `GET /api/cost` — JSON endpoint returning `UsageTracker.cost_summary()` (for HTMX polling)

### 1.5 Tests for Phase 1
- `tests/config/test_presets.py` — Preset loading, override behavior
- `tests/llm/test_cost.py` — Cost calculation accuracy
- `tests/llm/test_ollama.py` — Ollama provider (mocked httpx)
- `tests/cli/test_setup.py` — Wizard flow with mocked input
- `tests/web/test_cost_route.py` — Cost API endpoint

---

## Phase 2: Source Expansion
*Goal: More signal, no extra API keys required, progressive disclosure.*

### 2.1 Source Type Abstraction (`src/nexus/engine/sources/`)
Currently `polling.py` only handles RSS. Refactor to support pluggable source types:

- New file: `src/nexus/engine/sources/base.py`
  ```python
  class SourceAdapter(ABC):
      source_type: str
      @abstractmethod
      async def poll(self, source_config: dict) -> list[ContentItem]: ...
  ```
- Rename/refactor: `polling.py` → `rss.py` (class `RSSAdapter(SourceAdapter)`)
- New file: `src/nexus/engine/sources/router.py`
  ```python
  async def poll_source(source_config: dict) -> list[ContentItem]:
      adapter = ADAPTERS[source_config["type"]]  # "rss", "telegram", "reddit", "twitter"
      return await adapter.poll(source_config)
  ```
- Update `poll_all_feeds()` to use router instead of direct feedparser calls

### 2.2 Telegram Channel Source (`src/nexus/engine/sources/telegram_channel.py`)
Monitor public Telegram channels for news (free, no API key):

- Use `telethon` or scrape via `t.me/s/{channel}` (public preview, no auth)
- Adapter: `TelegramChannelAdapter(SourceAdapter)`
- Registry format:
  ```yaml
  - id: reuters-telegram
    type: telegram_channel
    channel: "@reuters"
    language: en
    affiliation: private
  ```
- Polling: Fetch latest N messages, convert to `ContentItem`
- Rate limiting: respect Telegram's public page limits
- **No bot token needed** — this uses public channel web preview, not the Bot API

### 2.3 Reddit Source (`src/nexus/engine/sources/reddit.py`)
Reddit exposes RSS feeds natively — minimal new code:

- `https://www.reddit.com/r/{subreddit}/top/.rss?t=day`
- Adapter: `RedditAdapter(SourceAdapter)` — thin wrapper around RSS with Reddit-specific metadata
- Registry format:
  ```yaml
  - id: reddit-worldnews
    type: reddit
    subreddit: worldnews
    sort: top
    language: en
    affiliation: social
  ```
- Add `"social"` as a new affiliation type for user-generated sources
- Include in convergence/divergence: social sources as "public sentiment" signal, not authoritative

### 2.4 Twitter/X via Nitter Bridge (`src/nexus/engine/sources/twitter.py`)
Nitter instances expose RSS feeds for public Twitter accounts:

- `https://{nitter_instance}/{username}/rss`
- Adapter: `TwitterAdapter(SourceAdapter)`
- Config: `nitter_instance` field (user-configurable, defaults to a list of public instances with fallback)
- Registry format:
  ```yaml
  - id: twitter-reuters
    type: twitter
    username: Reuters
    language: en
    affiliation: private
  ```
- **Fragility handling**: Nitter instances go down. Adapter tries multiple instances, logs failures, never blocks pipeline
- Mark as optional/experimental in docs

### 2.5 Configurable Breaking News Feeds
Currently hardcoded to Reuters + NYT. Make it configurable:

- Add `wire_feeds: list[dict]` to `BreakingNewsConfig`
- Default to current Reuters + NYT, but users can add Telegram channels, Twitter accounts, or any source
- Wire feeds use the same source adapter router — any source type works as a wire feed
- Update `check_breaking_news()` to use config instead of hardcoded URLs

### 2.6 Source Registry Updates
Add new source entries to topic registries using new source types:
- `data/sources/global_registry.yaml`: Add Reddit worldnews, Twitter wire accounts, key Telegram channels
- Topic registries: Add relevant subreddits, Twitter lists, Telegram channels per topic

### 2.7 Tests for Phase 2
- `tests/engine/sources/test_base.py` — Adapter interface contract
- `tests/engine/sources/test_router.py` — Routing by source type
- `tests/engine/sources/test_telegram_channel.py` — Telegram scraping (mocked HTTP)
- `tests/engine/sources/test_reddit.py` — Reddit RSS (mocked feedparser)
- `tests/engine/sources/test_twitter.py` — Nitter fallback logic (mocked HTTP)
- `tests/agent/test_breaking_configurable.py` — Custom wire feeds

---

## Phase 3: Polish & Cost Control
*Goal: Users understand and control their spending. Dashboard is useful.*

### 3.1 API Key Manager (Dashboard)
New dashboard page for managing provider keys:

- `GET /settings` — Settings page with:
  - Current preset and model assignments
  - API key status per provider (set/not set, last validated, valid/invalid)
  - "Test connection" button per provider
- `POST /settings/keys` — Update keys (writes to `.env`, reloads LLMClient)
- Keys are masked in UI (show last 4 chars only)
- Validation: test call on save, show success/failure

### 3.2 Cost Dashboard
New dashboard page and widgets:

- `GET /cost` — Dedicated cost page:
  - Daily cost chart (last 30 days, simple bar chart via Chart.js or inline SVG)
  - Per-topic breakdown: "Iran-US: $0.02/day avg, AI: $0.01/day avg"
  - Per-stage breakdown: "Filtering: 60%, Synthesis: 25%, Audio: 15%"
  - Per-provider breakdown
- Cost badge on main dashboard (current day spend)
- Data from `usage_log` table (added in Phase 1)

### 3.3 Budget Cap (`src/nexus/config/models.py` + `src/nexus/llm/client.py`)
Let users set spending limits:

- New config:
  ```yaml
  budget:
    daily_limit_usd: 0.50        # Hard stop
    warning_threshold_usd: 0.30  # Telegram alert
    degradation_strategy: "skip_expensive"  # or "stop_all"
  ```
- New `BudgetConfig` Pydantic model
- `LLMClient.complete()` checks budget before each call:
  - Under warning: normal operation
  - Over warning: send Telegram alert (once per day)
  - Over limit with `skip_expensive`: skip synthesis/dialogue, keep filtering (user still gets event list, no polished briefing)
  - Over limit with `stop_all`: skip all LLM calls, log warning
- Dashboard shows budget gauge (green → yellow → red)

### 3.4 Graceful Degradation Modes
When budget is hit or a provider is down:

| Mode | Filtering | Events | Synthesis | Audio | Briefing |
|------|-----------|--------|-----------|-------|----------|
| Full | LLM | LLM | LLM | TTS | Rich |
| Degraded | LLM | LLM | Skip | Skip | Event list only |
| Minimal | Keyword | Heuristic | Skip | Skip | Headlines only |
| Off | Skip | Skip | Skip | Skip | None |

- Pipeline stages check `LLMClient.budget_mode` and adapt
- "Minimal" mode uses keyword matching instead of LLM filtering — zero cost fallback

### 3.5 Source Health Dashboard
Enhance existing `/sources` page:

- Per-source: last poll time, success rate, avg articles/day
- Source type badges (RSS, Telegram, Reddit, Twitter)
- Health indicators: green (healthy), yellow (slow/intermittent), red (down)
- "Add source" form: paste URL, auto-detect type, validate, add to registry

### 3.6 Enhanced Dashboard Pages
Polish existing routes with real frontend:

- **Thread view** (`/threads/{slug}`): Timeline visualization of events, convergence/divergence display, entity links
- **Entity view** (`/entities/{entity_id}`): Entity profile, related events, cross-topic appearances
- **Briefing archive** (`/`): Past briefings browsable by date
- **Audio player**: Inline player for podcast episodes on dashboard

### 3.7 Tests for Phase 3
- `tests/config/test_budget.py` — Budget config validation
- `tests/llm/test_budget_enforcement.py` — Budget checking in LLMClient (mocked)
- `tests/llm/test_degradation.py` — Graceful degradation modes
- `tests/web/test_settings.py` — API key management routes
- `tests/web/test_cost_dashboard.py` — Cost page data
- `tests/web/test_source_health.py` — Source health display

---

## Implementation Order & Dependencies

```
Phase 1 (foundation):
  1.1 Presets ──────────┐
  1.2 Ollama provider ──┤
  1.3 Cost tracking ────┼── 1.4 Setup wizard ── 1.5 Dashboard
                        │
Phase 2 (sources):      │
  2.1 Source abstraction ┼── 2.2 Telegram ── 2.3 Reddit ── 2.4 Twitter
                        │                                      │
  2.5 Configurable wire ┘                                      │
  2.6 Registry updates ───────────────────────────────────────-─┘

Phase 3 (polish):
  3.1 Key manager ── 3.2 Cost dashboard ── 3.3 Budget cap ── 3.4 Degradation
  3.5 Source health ── 3.6 Enhanced dashboard
```

Within each phase, items can be built and shipped incrementally. Phase 1 is prerequisite for Phase 3 (cost tracking feeds cost dashboard and budget). Phase 2 is independent of Phase 1 except that the setup wizard should know about new source types.

## New Dependencies
- `htmx` (CDN, no install) — Dashboard interactivity
- `picocss` (CDN, no install) — Classless CSS framework
- Phase 2 only: `httpx` (already dev dep) for Telegram channel scraping + Nitter
- Phase 2 optional: `telethon` if we want proper Telegram channel API access

## File Count Estimate
- Phase 1: ~8 new files, ~5 edited files, ~5 test files
- Phase 2: ~6 new files, ~4 edited files, ~6 test files
- Phase 3: ~4 new files, ~8 edited files, ~6 test files
