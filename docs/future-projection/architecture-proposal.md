# Architecture Proposal: Projection Capabilities for Nexus

Three tiers from minimal (analytics on existing data) to ambitious (multi-agent scenario generation). Each tier builds on the previous.

---

## Tier 1: Analytical Projections

**Effort**: 1-2 weeks | **New LLM calls**: 0-1 per topic per run | **Dependencies**: None (SQLite only)

This tier extracts forward-looking signals from data Nexus already captures. It's the highest-value, lowest-risk starting point.

### 1a. Thread Trajectory Analysis

**Problem**: Nexus tracks thread lifecycle (emerging → active → stale → resolved) but has no concept of momentum. A thread gaining 5 events in one day looks identical to one gaining 1 event over 5 days. The `updated_at` field is overwritten, destroying temporal history.

**Solution**: Snapshot thread state daily, compute trajectory metrics.

**New table: `thread_snapshots`**
```sql
CREATE TABLE thread_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id INTEGER NOT NULL REFERENCES threads(id),
    snapshot_date TEXT NOT NULL,  -- YYYY-MM-DD
    significance INTEGER NOT NULL,
    event_count INTEGER NOT NULL,
    entity_count INTEGER NOT NULL,
    source_count INTEGER NOT NULL,
    UNIQUE(thread_id, snapshot_date)
);
CREATE INDEX idx_thread_snapshots_thread ON thread_snapshots(thread_id);
CREATE INDEX idx_thread_snapshots_date ON thread_snapshots(snapshot_date);
```

**New module: `src/nexus/engine/synthesis/trajectory.py`**

Core computations (all pure SQL/Python, no LLM):
- **Velocity**: `events_added / days_since_first_event` — how fast is this thread accumulating events?
- **Acceleration**: Change in velocity over last 7 days — is the thread speeding up or slowing down?
- **Significance trend**: Linear regression over daily significance snapshots — is importance rising or falling?
- **Predicted days to stale**: Based on current velocity, when will the thread go 14 days without events?
- **Trajectory classification**: `accelerating` (velocity increasing), `steady` (velocity stable), `decelerating` (velocity decreasing), `about-to-break` (high acceleration + significance > 7)

**Integration into NarrativeThread** (new optional fields):
```python
trajectory: Optional[str]  # accelerating | steady | decelerating | about-to-break
momentum_score: Optional[float]  # 0.0-1.0, composite of velocity + acceleration + significance trend
predicted_days_to_stale: Optional[int]  # estimated days until no new events
velocity: Optional[float]  # events per day (7-day window)
```

**Pipeline integration point**: After thread persistence in `pipeline.py` (after `persist_threads()`), capture daily snapshot and compute trajectory. Trajectory data is attached to `NarrativeThread` before returning the `TopicSynthesis`.

**Cost**: Zero LLM calls. Pure analytics on data already being stored.

---

### 1b. Causal Relation Extraction

**Problem**: Events have a `relation_to_prior` text field (e.g., "This escalates the sanctions dispute from last week") but causal relationships are not structured. You can't query "what events were caused by event X?" or "what are the upstream causes of this thread?"

**Solution**: Extend the event extraction prompt to output structured causal links alongside the existing `relation_to_prior` text.

**New table: `causal_links`**
```sql
CREATE TABLE causal_links (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cause_event_id INTEGER NOT NULL REFERENCES events(id),
    effect_event_id INTEGER NOT NULL REFERENCES events(id),
    relation_type TEXT NOT NULL,  -- causes | enables | prevents | escalates | resolves
    confidence REAL NOT NULL DEFAULT 0.5,  -- 0.0-1.0
    extracted_from TEXT,  -- which LLM call produced this link
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX idx_causal_cause ON causal_links(cause_event_id);
CREATE INDEX idx_causal_effect ON causal_links(effect_event_id);
CREATE INDEX idx_causal_type ON causal_links(relation_type);
```

**Prompt extension in `events.py`**: The current extraction prompt asks for `relation_to_prior` as free text. Extend to also request:
```json
{
  "causal_links": [
    {
      "prior_event_index": 3,
      "relation": "escalates",
      "confidence": 0.8
    }
  ]
}
```

The LLM already receives recent events as context (last 14 days). It already reasons about causal connections for `relation_to_prior`. Structuring this output adds minimal prompt overhead.

**New module: `src/nexus/engine/knowledge/causality.py`**

Core queries:
- `get_causal_chain(event_id) → list[Event]`: Walk upstream causes recursively
- `get_downstream_effects(event_id) → list[Event]`: Walk downstream effects
- `get_thread_causal_graph(thread_id) → dict`: Full causal DAG for a thread
- `get_cross_thread_causal_links(topic_slug) → list`: Causal bridges between threads

**Pipeline integration point**: During event extraction (`extract_events()` in `events.py`), parse causal link output alongside existing fields. Persist to `causal_links` table after event insertion (need event IDs first).

**Cost**: ~0 additional LLM cost (extending an existing prompt, not adding a new call).

---

### 1c. Cross-Topic Entity Bridging

**Problem**: The pipeline processes topics sequentially and independently. An entity appearing in multiple topics (e.g., "Nvidia" in both AI/ML and Energy) may signal a cross-topic correlation, but this is never detected.

**Solution**: Add a post-pipeline cross-topic analysis pass.

**New store query: `get_cross_topic_entities()`**
```sql
SELECT e.id, e.canonical_name, e.entity_type,
       GROUP_CONCAT(DISTINCT ev.topic_slug) as topics,
       COUNT(DISTINCT ev.topic_slug) as topic_count,
       COUNT(DISTINCT ev.id) as event_count
FROM entities e
JOIN event_entities ee ON e.id = ee.entity_id
JOIN events ev ON ee.event_id = ev.id
WHERE ev.date >= date('now', '-14 days')
GROUP BY e.id
HAVING COUNT(DISTINCT ev.topic_slug) >= 2
ORDER BY topic_count DESC, event_count DESC;
```

**New store query: `get_temporal_coincidences(entity_id, window_hours=48)`**
Find events about the same entity in different topics within a time window.

**New model: `CrossTopicCorrelation`**
```python
@dataclass
class CrossTopicCorrelation:
    entity: str
    topics: list[str]
    recent_events_by_topic: dict[str, list[Event]]  # topic_slug → events
    temporal_coincidence: bool  # events in different topics within 48h
    potential_causal_bridge: Optional[str]  # LLM-generated assessment (if requested)
```

**Pipeline integration point**: After all topics complete in `run_pipeline()` (after the per-topic loop), run `get_cross_topic_entities()`. For entities appearing in 2+ topics with events in the last 48h, flag as a cross-topic correlation. Optionally, make one LLM call to assess whether the correlation is causal or coincidental.

**Output**: Cross-topic correlations are included in the pipeline result and can be surfaced in briefings, dashboard, or alerts.

**Cost**: One SQL query (fast) + optionally 1 LLM call for the batch of cross-topic entities found.

---

### 1d. Projection Artifact

**Problem**: Nexus produces briefings that describe what happened. Users want forward-looking content: "based on what we're tracking, here's what to watch for."

**Solution**: New `projection` page type that synthesizes thread trajectories, causal chains, and cross-topic correlations into a "Forward Look."

**New page type in `pages.py`**:
```python
PAGE_CONFIGS = {
    ...
    "projection": PageConfig(ttl_days=1, max_words=800),
}
```

**Prompt design** (LLM-generated, grounded in data):
```
You are an intelligence analyst producing a forward-looking assessment.

## Thread Trajectories
{for each active thread with trajectory data:}
- "{headline}" — {trajectory} (velocity: {velocity} events/day, significance trend: {trend})
  Events: {recent_event_summaries}

## Causal Chains
{for each thread with causal links:}
- Chain: {event_A} --[{relation}]--> {event_B} --[{relation}]--> {event_C}

## Cross-Topic Signals
{for each cross-topic correlation:}
- Entity "{name}" active in topics: {topic_list}
  Recent events: {event_summaries_by_topic}

## Instructions
Generate 2-4 forward-looking assessments. For each:
1. State what you expect may happen next (1-2 sentences)
2. Cite the specific signals: which thread trajectory, causal chain, or cross-topic signal
3. Rate confidence: LOW (single signal), MEDIUM (2+ corroborating signals), HIGH (strong trajectory + causal chain + cross-topic confirmation)
4. Suggest what to watch for (a "signpost" — a specific event that would confirm or disconfirm this assessment)

CRITICAL: Every assessment must be grounded in the data above. Do not speculate beyond what the signals support.
```

**Pipeline integration point**: After synthesis and trajectory computation, before briefing rendering. The projection artifact becomes a section in the daily briefing and a standalone page in the dashboard.

**Cost**: 1 LLM call per topic per run (using the `knowledge_summary` model, typically a flash model).

---

## Tier 2: Enhanced Knowledge Graph

**Effort**: 2-4 weeks | **Dependencies**: None (ports concepts into SQLite) | **Prerequisite**: Tier 1

Inspired by Graphiti's design, but implemented within Nexus's existing SQLite store.

### 2a. Temporal Validity Windows

**Problem**: Convergence records facts that sources agree on, but facts can become outdated or be superseded. "Russia and Ukraine are in ceasefire negotiations" may have been true last week but false today. No mechanism to track this.

**Solution**: Add temporal validity to convergence records, inspired by Graphiti's bi-temporal model.

**Schema extension on `convergence`**:
```sql
ALTER TABLE convergence ADD COLUMN valid_from TEXT;  -- YYYY-MM-DD, when fact first confirmed
ALTER TABLE convergence ADD COLUMN valid_until TEXT;  -- YYYY-MM-DD, when fact superseded (NULL = still valid)
ALTER TABLE convergence ADD COLUMN superseded_by INTEGER REFERENCES convergence(id);
```

**Behavior**:
- When a new convergence fact is added, set `valid_from = today`, `valid_until = NULL`
- During synthesis, if a new fact contradicts an existing one (detected via entity + topic overlap + LLM confirmation), set `valid_until = today` on the old fact and `superseded_by = new_fact_id`
- Query: "What was believed true about topic X on date Y?" → `WHERE valid_from <= Y AND (valid_until IS NULL OR valid_until > Y)`

**Value for projection**: Enables tracking how the "consensus view" of a topic evolves. A fact that has been superseded 3 times in 2 weeks signals an unstable situation — useful input for projection confidence.

### 2b. Source Reliability Scoring

**Problem**: All outlets are treated equally in convergence/divergence analysis. A source that has been consistently accurate should carry more weight in projections.

**Solution**: Track per-outlet accuracy over time.

**New table: `source_accuracy`**
```sql
CREATE TABLE source_accuracy (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    outlet TEXT NOT NULL,
    topic_slug TEXT NOT NULL,
    period_start TEXT NOT NULL,  -- YYYY-MM-DD
    period_end TEXT NOT NULL,
    claims_made INTEGER NOT NULL DEFAULT 0,
    claims_confirmed INTEGER NOT NULL DEFAULT 0,  -- later confirmed by 2+ independent sources
    claims_contradicted INTEGER NOT NULL DEFAULT 0,  -- later contradicted by 2+ sources
    accuracy_score REAL,  -- confirmed / (confirmed + contradicted), NULL if insufficient data
    UNIQUE(outlet, topic_slug, period_start)
);
```

**Population**: Runs weekly as a background job. For each outlet's convergence claims from 2+ weeks ago, check whether subsequent convergence records from other outlets confirmed or contradicted them.

**Value for projection**: Claims from outlets with high `accuracy_score` on a topic get higher weight when generating forward-looking assessments.

**Note**: Requires weeks of accumulated data before scores become meaningful. Should be gated behind a minimum data threshold (e.g., 20+ claims per outlet per topic).

---

## Tier 3: Multi-Agent Projection

**Effort**: Variable | **Dependencies**: Tier 1 complete, 30+ days of data | **Risk**: Experimental

### 3a. Structured Scenario Generation

**Problem**: Thread trajectory and causal chains show where things are heading, but don't explore alternative outcomes. Traders and founders need to prepare for multiple scenarios.

**Solution**: A 3-agent structured reasoning workflow using the existing `LLMClient`. Not a MiroFish-style simulation — no agent personalities or social media simulation. Just structured analytical reasoning.

**Workflow**:
```
                    Thread trajectory data
                    Causal chain graph
                    Cross-topic signals
                           │
                    ┌──────▼──────┐
                    │   ANALYST   │  Reads context, identifies
                    │   Agent     │  2-3 key uncertainties
                    └──────┬──────┘
                           │ uncertainties + context
                    ┌──────▼──────┐
                    │  SCENARIO   │  For each uncertainty, generates
                    │   Agent     │  2-3 plausible scenarios with
                    └──────┬──────┘  triggers and consequences
                           │ scenarios
                    ┌──────▼──────┐
                    │   CRITIC    │  Rates plausibility against
                    │   Agent     │  historical patterns, flags
                    └──────┬──────┘  hallucination risks
                           │
                    Calibrated scenarios
```

**Implementation**: Sequential `LLMClient.complete()` calls (not a framework like LangGraph). Three calls per active thread batch.

**Budget impact**: With 10 active threads across 4 topics, processed in batches of 5: ~6-8 LLM calls per pipeline run. At flash-model pricing (~$0.001/call), adds ~$0.01/run. At pro-model pricing (~$0.01/call), adds ~$0.08/run. Feasible within `balanced` budget preset but should be configurable.

**Gating**: Only runs if (a) topic has 30+ days of event history, (b) thread has trajectory data (3+ snapshots), (c) user has opted into projection features.

### 3b. Calibration Loop

**Problem**: Without feedback on prediction accuracy, projections are just structured speculation. The system needs to learn from its mistakes.

**Solution**: Record projections, compare to outcomes, feed back into prompts.

**New tables**:
```sql
CREATE TABLE projections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    topic_slug TEXT NOT NULL,
    thread_id INTEGER REFERENCES threads(id),
    projection_text TEXT NOT NULL,
    confidence TEXT NOT NULL,  -- LOW | MEDIUM | HIGH
    signpost TEXT,  -- what to watch for
    signals_cited TEXT,  -- JSON: which data points supported this
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    review_after TEXT  -- YYYY-MM-DD, when to check outcome
);

CREATE TABLE projection_outcomes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    projection_id INTEGER NOT NULL REFERENCES projections(id),
    outcome TEXT NOT NULL,  -- confirmed | partially_confirmed | disconfirmed | inconclusive
    evidence TEXT,  -- which events confirmed or contradicted
    reviewed_at TEXT NOT NULL DEFAULT (datetime('now'))
);
```

**Population**: Weekly review job compares projections whose `review_after` date has passed against actual events. Initially manual (user confirms via Telegram bot or dashboard), later semi-automated (LLM compares projection text against subsequent events).

**Feedback into prompts**: After accumulating 20+ reviewed projections, include accuracy stats in projection prompts: "Your recent LOW-confidence projections were correct 30% of the time. Your HIGH-confidence projections were correct 75% of the time. Calibrate accordingly."

---

## Pipeline Integration Map

Where each new capability fits in the existing pipeline flow:

```
EXISTING PIPELINE                          NEW STEPS
─────────────────                          ─────────

[1] Poll & Filter Recent
[2] Dedup
[3] Ingest
[4] Two-Pass Filter
[5] Event Extraction  ──────────────────►  [5b] Causal Link Extraction (Tier 1b)
                                                Parse structured causal output
                                                from extended extraction prompt
[6] Entity Resolution
[7] Persist Events
[8] Compress (background)
[9] Load Context
[10] Synthesis
[11] Persist Threads  ──────────────────►  [11b] Thread Snapshot (Tier 1a)
                                                 Capture daily snapshot to
                                                 thread_snapshots table
                                           [11c] Trajectory Computation (Tier 1a)
                                                 Compute velocity, acceleration,
                                                 trend, classification

─── Per-topic loop ends ───

                      ──────────────────►  [12] Cross-Topic Correlation (Tier 1c)
                                                get_cross_topic_entities()
                                                Temporal coincidence analysis

[13] Render Briefing  ──────────────────►  [13b] Projection Artifact (Tier 1d)
                                                 LLM-generated "Forward Look"
                                                 from trajectory + causal + cross-topic

                      ──────────────────►  [14] Scenario Generation (Tier 3a)
                                                 3-agent workflow (optional)

[15] Audio Pipeline (optional)
[16] Metrics & Logging
```

---

## Schema Migration Plan

All new tables follow the established migration pattern in `schema.py`:

```python
MIGRATION_V10 = """
-- Thread trajectory snapshots
CREATE TABLE IF NOT EXISTS thread_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id INTEGER NOT NULL REFERENCES threads(id),
    snapshot_date TEXT NOT NULL,
    significance INTEGER NOT NULL,
    event_count INTEGER NOT NULL,
    entity_count INTEGER NOT NULL,
    source_count INTEGER NOT NULL,
    UNIQUE(thread_id, snapshot_date)
);
CREATE INDEX IF NOT EXISTS idx_thread_snapshots_thread ON thread_snapshots(thread_id);
CREATE INDEX IF NOT EXISTS idx_thread_snapshots_date ON thread_snapshots(snapshot_date);

-- Causal links between events
CREATE TABLE IF NOT EXISTS causal_links (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cause_event_id INTEGER NOT NULL REFERENCES events(id),
    effect_event_id INTEGER NOT NULL REFERENCES events(id),
    relation_type TEXT NOT NULL,
    confidence REAL NOT NULL DEFAULT 0.5,
    extracted_from TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_causal_cause ON causal_links(cause_event_id);
CREATE INDEX IF NOT EXISTS idx_causal_effect ON causal_links(effect_event_id);
"""

MIGRATION_V11 = """
-- Temporal validity for convergence (Tier 2a)
ALTER TABLE convergence ADD COLUMN valid_from TEXT;
ALTER TABLE convergence ADD COLUMN valid_until TEXT;
ALTER TABLE convergence ADD COLUMN superseded_by INTEGER REFERENCES convergence(id);
"""

MIGRATION_V12 = """
-- Source reliability tracking (Tier 2b)
CREATE TABLE IF NOT EXISTS source_accuracy ( ... );

-- Projection tracking (Tier 3b)
CREATE TABLE IF NOT EXISTS projections ( ... );
CREATE TABLE IF NOT EXISTS projection_outcomes ( ... );
"""
```

---

## Files to Modify (by tier)

### Tier 1
| File | Change |
|---|---|
| `src/nexus/engine/knowledge/schema.py` | Add MIGRATION_V10 (thread_snapshots, causal_links) |
| `src/nexus/engine/knowledge/store.py` | Add snapshot/trajectory/cross-topic query methods |
| `src/nexus/engine/synthesis/trajectory.py` | **New file**: velocity, acceleration, trend, classification |
| `src/nexus/engine/knowledge/causality.py` | **New file**: causal chain queries |
| `src/nexus/engine/knowledge/events.py` | Extend extraction prompt for causal link output |
| `src/nexus/engine/synthesis/knowledge.py` | Add trajectory fields to NarrativeThread, CrossTopicCorrelation model |
| `src/nexus/engine/pipeline.py` | Wire in snapshot capture, trajectory, cross-topic pass, projection artifact |
| `src/nexus/engine/knowledge/pages.py` | Add `projection` page type |

### Tier 2
| File | Change |
|---|---|
| `src/nexus/engine/knowledge/schema.py` | Add MIGRATION_V11 (convergence temporal validity) |
| `src/nexus/engine/knowledge/store.py` | Add temporal validity queries, supersession logic |
| `src/nexus/engine/synthesis/knowledge.py` | Extend convergence handling to check for contradictions |

### Tier 3
| File | Change |
|---|---|
| `src/nexus/engine/knowledge/schema.py` | Add MIGRATION_V12 (projections, projection_outcomes) |
| `src/nexus/engine/knowledge/store.py` | Add projection persistence and retrieval |
| `src/nexus/engine/synthesis/scenarios.py` | **New file**: 3-agent scenario workflow |
| `src/nexus/engine/evaluation/calibration.py` | **New file**: projection outcome comparison |
