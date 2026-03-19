# Pipeline Parameters Reference

Every tunable threshold, limit, and scoring rubric in the Nexus engine pipeline.

Parameters marked **configurable** can be set in `data/config.yaml`.
All others are code constants — modify in source or wait for future config support.

---

## 1. Source Polling

| Parameter | Value | Location | Notes |
|-----------|-------|----------|-------|
| Recency filter | 48 hours | `polling.py` `filter_recent()` | Drops RSS items older than 48h before processing |

## 2. Ingestion

| Parameter | Value | Location | Notes |
|-----------|-------|----------|-------|
| `MAX_INGEST` | 250 articles (default) | `pipeline.py:131` | Cap per topic — overridable via `max_ingest` param. Smoke tests use 20 |
| Global concurrency | 10 | `ingest.py` semaphore | Max concurrent HTTP fetches |
| Per-domain concurrency | 2 | `ingest.py` semaphore | Rate-limits per domain |

## 3. Two-Pass Filtering

### Pass 1: Relevance Scoring

| Parameter | Value | Location | Notes |
|-----------|-------|----------|-------|
| `BATCH_SIZE` | 10 | `filter.py:44` | Articles per LLM call |
| `filter_threshold` | 4.0 (default) | `models.py:29` | **Configurable** per-topic. Scale: 1–10, lower = more pass. Benchmark: 4.0 optimal (6.4/10), 5.0 slightly lower (6.1), ≥7.0 collapses quality |
| Snippet cap | 1000 chars | `filter.py:90` | Max article text per batch item |

**Rubric** (system prompt at `filter.py:23-29`):
> Score relevance 1–10. Include source metadata (affiliation, country) in assessment.
> State media, public broadcasters, and private outlets all provide valid perspectives.

### Pass 2: Significance + Novelty

| Parameter | Value | Location | Notes |
|-----------|-------|----------|-------|
| `PASS2_BATCH_SIZE` | 5 | `filter.py:56` | Smaller batches — more text per article |
| Text cap | 2000 chars | `filter.py:147` | Max article text per batch item |
| Pass threshold | significance ≥ 4 **OR** `is_novel` | `filter.py:453` | Either condition passes |

**Rubric** (system prompt at `filter.py:42-48`):
> Score significance 1–10, assess novelty (boolean) against known events from last 7 days.

### Composite Score

```
novelty_bonus = 1.0 if is_novel else 0.7
composite = (relevance × 0.4 + significance × 0.6) × novelty_bonus
```

Constants: `RELEVANCE_WEIGHT=0.4`, `SIGNIFICANCE_WEIGHT=0.6`, `NOVEL_BONUS=1.0`, `NON_NOVEL_BONUS=0.7` (`filter.py:59-62`)

### Perspective Diversity

| Parameter | Value | Location | Notes |
|-----------|-------|----------|-------|
| `max_items` | 30 | `filter.py:198` | Max articles after diversity selection |
| `perspective_diversity` | `"high"` (default) | `models.py` | **Configurable** per-topic: `low`, `medium`, `high`. Benchmark: high diversity improves source balance by 43% |
| High diversity | ≥ 20% per affiliation | `filter.py:209` | Min representation from each source type |
| Medium diversity | ≥ 10% per affiliation | `filter.py:209` | |
| Low diversity | Pure score ranking | `filter.py:206-207` | No diversity constraint |

## 4. Event Extraction

| Parameter | Value | Location | Notes |
|-----------|-------|----------|-------|
| Event cap (narrow) | 15 | `pipeline.py:146` | Max events per topic per run |
| Event cap (medium) | 20 | `pipeline.py:146` | |
| Event cap (broad) | 35 | `pipeline.py:146` | |
| `max_events` | (override) | `models.py:31` | **Configurable** per-topic |
| Extraction concurrency | 5 | `pipeline.py:181` | Semaphore for parallel extraction |
| Entity overlap threshold | 0.6 | `events.py:52` | 60% entity overlap → duplicate event |
| Event context window | 14 days | `pipeline.py:153` | How far back to load existing events |
| Recent events for novelty | 7 days, max 30 | `pipeline.py:156-157` | Context window for novelty assessment |
| Text cap | 3000 chars | `events.py` | Max article text in extraction prompt |

**Rubric** (system prompt at `events.py:99-111`):
> Extract: date (MUST be ≤ today), summary (1–2 sentences), entities, relation to prior events, significance 1–10.
> Critical date rules: event date is when it happened, not future speculation.

## 5. Entity Resolution

| Parameter | Value | Location | Notes |
|-----------|-------|----------|-------|
| Known entity cap | 200 | `entities.py:62` | Max known entities in LLM prompt to avoid overflow |

**Rubric** (system prompt at `entities.py:21-38`):
> Map raw names to canonical forms. Types: person, org, country, treaty, concept.
> Be conservative — only merge entities you are CERTAIN are the same.

## 6. Thread Matching

| Parameter | Value | Location | Notes |
|-----------|-------|----------|-------|
| `HIGH_OVERLAP` | 0.5 | `threads.py:19` | Auto-match, no LLM confirmation needed |
| `LOW_OVERLAP` | 0.3 | `threads.py:20` | Below this = no match candidate |
| Thread lifecycle | emerging → active → stale → resolved | `threads.py` | active after 2+ distinct days |

**Rubric** (system prompt at `threads.py:70-79`):
> Match events to existing threads by slug, or group unmatched into new threads.

## 7. Knowledge Synthesis

| Parameter | Value | Location | Notes |
|-----------|-------|----------|-------|
| Synthesis fallback | last 3 days, 10 events | `pipeline.py:274` | Used when no new events extracted |
| Weekly summary context | last 3 weeks | `knowledge.py:193` | Background summaries for synthesis |
| Monthly summary context | last 1 month | `knowledge.py:196` | |
| Max threads on error | 10 | `knowledge.py:256` | Fallback cap for thread generation |

**Rubric** (system prompt at `knowledge.py:60-146`):
> Group events into story arcs. Identify convergence (2+ independent sources) and divergence (conflicting framing).
> Source affiliation assessment: state (gov-controlled), public (editorially independent), private (varies), nonprofit/academic.
> Thread consolidation: threads sharing 3+ entities in same causal chain MUST be merged.

## 8. Page Cache (Narrative Pages)

| Page Type | TTL | Location |
|-----------|-----|----------|
| `backstory` | 7 days | `pages.py:19` |
| `entity_profile` | 3 days | `pages.py:20` |
| `thread_deepdive` | 1 day | `pages.py:21` |
| `weekly_recap` | 365 days | `pages.py` (immutable) |

## 9. Breaking News

| Parameter | Value | Location | Notes |
|-----------|-------|----------|-------|
| `poll_interval_hours` | 3 | `models.py:54` | **Configurable** |
| `threshold` | 7 | `models.py:55` | **Configurable**. Significance ≥ 7 → alert |

## 10. Budget & Cost

| Parameter | Value | Location | Notes |
|-----------|-------|----------|-------|
| `daily_limit_usd` | 1.00 | `models.py:73` | **Configurable** |
| `warning_threshold_usd` | 0.50 | `models.py:74` | **Configurable** |
| `degradation_strategy` | `"skip_expensive"` | `models.py:75` | **Configurable**: `skip_expensive` or `stop_all` |
| Expensive operations | `synthesis`, `dialogue_script`, `agent` | `budget.py:22` | Blocked first when over limit |

### Cost Estimation

Pricing table in `cost.py:6-29`. Formula:

```
cost = (input_tokens / 1M) × input_price + (output_tokens / 1M) × output_price
```

## 11. Source Discovery

### Source Tier Taxonomy

Sources are classified into tiers that indicate editorial quality and coverage authority:

| Tier | Description | Examples |
|------|-------------|----------|
| **A** | Major outlets with original reporting, primary news agencies, official government/institutional sources | BBC, NYT, Reuters, Semiconductor Engineering, NASA, CISA |
| **B** | Established secondary outlets, regional publications, trade press with regular coverage | Tom's Hardware, The Register, Nikkei Asia, EE News Europe |
| **C** | Social media aggregators, personal blogs, unverified sources | Reddit, Twitter/X, individual blogs |

Tiers are informational metadata — they do not currently affect filtering, scoring, or synthesis weighting. Discovery prioritizes finding A-tier and B-tier sources; C-tier sources are not actively discovered.

### Discovery Flow (Agentic)

Discovery uses a budget-aware agentic loop that evaluates feed quality and refines search queries when initial results are too generic:

| Step | Description | Location |
|------|-------------|----------|
| 1. Global registry matching | LLM scores curated feeds against topic relevance | `discovery.py` |
| 2. Google News RSS | Free, always-valid RSS feeds via Google News search | `discovery.py` |
| 3. Web search discovery | DuckDuckGo search for RSS feed URLs | `discovery.py` |
| 3b. Feed evaluation | Score sample article titles from each feed against topic (1-10) | `discovery.py` |
| 3c. Query refinement | If < 3 good feeds found, generate specialized queries and repeat 3→3b | `discovery.py` |
| 4. Metadata classification | LLM classifies affiliation, country, tier for unknowns | `discovery.py` |
| 5. Diversity scoring | Shannon entropy across geographic, affiliation, language axes | `diversity.py` |

**Agentic refinement**: After initial web search, each discovered feed's sample article titles are scored against the topic. Feeds scoring below 5/10 are dropped as too generic. If fewer than 3 good feeds remain and the LLM budget allows, the system generates more targeted queries (focusing on trade journals, industry associations, government agencies) and runs another discovery+evaluation round.

### Discovery Parameters

| Parameter | Value | Location | Notes |
|-----------|-------|----------|-------|
| Max registry matches | 20 | `discovery.py` | Top curated feeds scored ≥ 5/10 |
| Registry relevance threshold | 5 | `discovery.py` | Score 1–10, below 5 excluded |
| Google News subtopic limit | 3 | `discovery.py` | Max subtopic queries for Google News RSS |
| Web search results per query | 8 | `websearch.py` | DuckDuckGo results per query |
| Feed validation concurrency | 5 | `discovery.py` | Semaphore for parallel URL validation |
| Max web URLs to validate | 30 | `discovery.py` | Cap on web-discovered URLs to check |
| Default max feeds | 25 | `discovery.py` | Total feeds returned per topic |
| Feed evaluation threshold | 5 | `discovery.py` | Feeds scoring < 5 are dropped as irrelevant |
| Sample titles per feed | 5 | `discovery.py` | Article titles sampled for evaluation |
| `max_rounds` | 2 | `discovery.py` | Discovery rounds (1=initial, 2+=with refinement) |
| `max_llm_calls` | 8 | `discovery.py` | Budget cap for total LLM calls during discovery |
| Refinement trigger | < 3 good web feeds | `discovery.py` | Triggers another round with refined queries |
| Smoke test max feeds | 8 | `smoke.py` | Reduced for speed |

### Curated Source Registries

For best results with niche topics, curate a dedicated source registry:

- **Global registry**: `data/sources/global_registry.yaml` — ~80 sources across 10+ verticals (world, politics, tech-AI, energy-climate, space, cyber, health, defense, finance, semiconductors, science)
- **Per-topic registries**: `data/sources/<topic-slug>/registry.yaml` — hand-curated sources for specific topics (e.g., `semiconductor-supply-chain/registry.yaml`)

Discovery checks the global registry first (LLM-scored relevance), then falls back to web search. Per-topic registries are loaded directly by the pipeline ingestion step — they bypass discovery entirely.

### Diversity Scoring

| Metric | Formula | Thresholds |
|--------|---------|------------|
| Geographic | Shannon entropy of country distribution | `< 1.0` → warning |
| Affiliation | Shannon entropy of affiliation distribution | `< 0.8` → warning |
| Language | Shannon entropy of language distribution | `< 0.5` → warning |
| Overall | Mean of geographic, affiliation, language scores | — |

Warnings triggered by:
- Geographic concentration: `< 1.0` entropy (< 3 countries effectively)
- Single perspective: only 1 affiliation type
- Unknown affiliations: > 50% feeds with `affiliation="unknown"`

### LLM Prompts

| Prompt | Location | Purpose |
|--------|----------|---------|
| `QUERY_SYSTEM_PROMPT` | `discovery.py` | Generate search queries for finding RSS feeds (prioritizes trade journals) |
| `REGISTRY_MATCH_PROMPT` | `discovery.py` | Score curated sources against topic (conservative for niche topics) |
| `CLASSIFY_PROMPT` | `discovery.py` | Classify feed metadata (defaults to tier B) |
| `EVALUATE_PROMPT` | `discovery.py` | Score sample article titles against topic relevance (1-10) |
| `REFINE_QUERIES_PROMPT` | `discovery.py` | Generate specialized queries after generic results |

## 12. Quality Evaluation (Judge)

**Rubric** (system prompt at `judge.py:12-34`):
> Score on 5 dimensions (1–10 each): completeness, source balance, convergence accuracy, divergence detection, entity coverage.

## 13. Future Projection Pipeline

### Projection Configuration

| Parameter | Value | Location | Notes |
|-----------|-------|----------|-------|
| `future_projection.enabled` | `false` | `models.py:108` | **Configurable**. Master switch for projection generation |
| `future_projection.engine` | `"actor"` | `models.py:110` | **Configurable**. Engine choice: `actor` \| `native` \| `graphrag` \| `perspective` \| `debate` \| `naked` \| `structural` |
| `future_projection.min_history_days` | 7 | `models.py:111` | **Configurable**. Minimum days of event data needed for eligibility |
| `future_projection.min_thread_snapshots` | 2 | `models.py:112` | **Configurable**. Minimum thread snapshots needed for eligibility |
| `future_projection.horizons` | `[3, 7, 14]` | `models.py:113` | **Configurable**. Forecast horizons in days (reserved for future use) |
| `future_projection.max_items_per_topic` | 3 | `models.py:114` | **Configurable**. Max forecast questions generated per topic (1-5) |
| `future_projection.critic_pass` | `true` | `models.py:115` | **Configurable**. Whether to apply confidence filtering pass |
| `future_projection.daily_engine` | `"structural"` | `models.py:118` | **Configurable**. Engine used for scheduled daily runs |
| `future_projection.prediction_schedule_offset_minutes` | 30 | `models.py:117` | **Configurable**. Delay after pipeline before prediction run |
| `future_projection.kg_native_enabled` | `true` | `models.py:119` | **Configurable**. Enable KG-native predictions (no external market) |
| `future_projection.max_kg_questions_per_topic` | 5 | `models.py:120` | **Configurable**. Max KG-native questions per topic (1-10) |
| `topic.projection_eligible` | `true` | `models.py:32` | **Configurable** per-topic. Set `false` to skip projection for a topic |

### Calibration Constants

| Parameter | Value | Location | Notes |
|-----------|-------|----------|-------|
| `gamma` (extremization) | 0.8 | `swarm.py:51` | gamma < 1 compresses overconfident probabilities toward 0.5 |
| `swarm_weight` (anchor blend) | 0.4 | `swarm.py:104` | LLM moves probability by 40% of gap between anchor and LLM estimate |
| Probability clipping | [0.02, 0.98] | `swarm.py:35,48,61` | Hard bounds on all probabilities |
| `MIN_RESOLVED_FOR_TUNING` | 20 | `swarm.py:120` | Minimum resolved forecasts needed before auto-calibration |
| Baseline `swarm_weight` (tuning) | 0.45 | `swarm.py:160` | Default weight used during grid search baseline |

### Verdict Derivation Thresholds

| Probability Range | Verdict | Confidence |
|-------------------|---------|------------|
| >= 0.80 | yes | high |
| >= 0.65 | yes | medium |
| >= 0.55 | yes | low |
| 0.45 - 0.55 | uncertain | low |
| <= 0.45 | no | low |
| <= 0.35 | no | medium |
| <= 0.20 | no | high |

### Eligibility Thresholds (service.py)

| Parameter | Value | Location | Notes |
|-----------|-------|----------|-------|
| Min history days | `config.min_history_days` (default 7) | `service.py:169` | Days between first and last event for topic |
| Min thread snapshots | `config.min_thread_snapshots` (default 2) | `service.py:169` | Max snapshot count across all threads |
| Stale synthesis cutoff | 3 days | `service.py:690` | KG-native skips topics with synthesis older than 3 days |
| Recent events window | 14 days, max 40 | `service.py:53` | Events loaded for forecast engine context |
| Cross-topic signal limit | 5 | `service.py:391` | Max signals attached per topic |

### Trajectory Classification Thresholds (analytics.py)

| Constant | Value | Label Triggered | Notes |
|----------|-------|-----------------|-------|
| `BREAK_MIN_EVENTS` | 3 | `about_to_break` | Minimum event count |
| `BREAK_MIN_SIGNIFICANCE` | 7 | `about_to_break` | Minimum significance score |
| `BREAK_MIN_VELOCITY` | 2.0 | `about_to_break` | Minimum 7-day velocity |
| `ACCEL_VELOCITY_THRESHOLD` | 1.0 | `accelerating` | Velocity > 1.0 triggers |
| `ACCEL_ACCEL_THRESHOLD` | 0.5 | `accelerating` | Acceleration > 0.5 triggers |
| `ACCEL_SIG_TREND_THRESHOLD` | 1.0 | `accelerating` | Significance trend > 1.0 triggers |
| `DECEL_VELOCITY_THRESHOLD` | -0.5 | `decelerating` | Velocity < -0.5 triggers |
| `DECEL_ACCEL_THRESHOLD` | -0.5 | `decelerating` | Acceleration < -0.5 triggers |
| `DECEL_SIG_TREND_THRESHOLD` | -1.0 | `decelerating` | Significance trend < -1.0 triggers |

`about_to_break` requires ALL conditions met (events >= 3, significance >= 7, velocity >= 2.0, acceleration > 0). `accelerating` and `decelerating` trigger on ANY single condition. Default label is `steady`.

### Momentum Formula (analytics.py)

```
momentum = (velocity_7d * 1.5) + acceleration_7d + significance_trend_7d
```

### Kalshi Benchmark Configuration

| Parameter | Value | Location | Notes |
|-----------|-------|----------|-------|
| `kalshi.enabled` | `false` | `models.py:93` | **Configurable** |
| `kalshi.auto_scan` | `false` | `models.py:102` | **Configurable**. Auto-scan markets after projection |
| `kalshi.auto_match_min_score` | 2 | `models.py:103` | **Configurable**. Minimum match score (1-based) |
| `kalshi.max_markets_per_topic` | 5 | `models.py:104` | **Configurable**. Max markets per topic (1-20) |
| `kalshi.comparison_tolerance_minutes` | 30 | `models.py:100` | **Configurable**. Tolerance for timestamp comparison |

---

## LLM Model Assignments

All LLM calls go through `src/nexus/llm/client.py`. Model selection is by `config_key`:

| Config Key | Default Model | Used For |
|------------|--------------|----------|
| `filtering` | gemini-3-flash-preview | Pass 1 + Pass 2 scoring |
| `knowledge_summary` | gemini-3-flash-preview | Event extraction, entity resolution, synthesis |
| `synthesis` | gemini-3.1-pro-preview | (currently unused — synthesis uses knowledge_summary) |
| `dialogue_script` | gemini-3.1-pro-preview | Podcast script generation |
| `breaking_news` | gemini-3-flash-preview | Breaking news assessment |
| `agent` | gemini-3.1-pro-preview | Telegram Q&A |
| `discovery` | gemini-3-flash-preview | Source discovery queries |

See `config/models.py` and `data/config.example.yaml` for preset configurations.
