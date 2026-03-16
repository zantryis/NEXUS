# Nexus Gap Analysis: Current State vs. Projection Requirements

A precise mapping between what Nexus's knowledge layer provides today and what each projection capability requires. References point to actual code paths.

---

## 1. Entity Relationship Graph

### What Nexus Has

- **`event_entities` join table** (`schema.py:40-47`): Links events to canonicalized entities with role tagging (subject / object / mentioned)
- **`get_related_entities()`** (`store.py`): Computes co-occurrence counts between entities
- **`get_graph_data()`** (`store.py`): Builds force-directed graph with nodes (entities) and links (co-occurrence frequency) for dashboard visualization
- **`cluster_threads()`** (`synthesis/clustering.py`): Groups threads by Jaccard entity overlap
- **Entity resolution** (`entities.py`): LLM-based canonicalization with conservative merging — "be CERTAIN they are the same"
- **Entity types**: person, org, country, treaty, concept, unknown

### What's Missing for Projection

- **No edge types beyond co-occurrence**: The graph connects entities that appear in the same events, but doesn't capture the nature of the relationship (competes-with, supplies-to, regulates, influences, allies-with)
- **No directional relationships**: A → influences → B is not distinguished from B → influences → A
- **No edge weights beyond frequency**: Co-occurrence count doesn't capture relationship strength, polarity (positive/negative), or temporal evolution
- **No cross-topic entity correlation**: While the `entities` table is technically global (no topic_slug column), queries in `store.py` are typically scoped by topic through the events table. No automatic detection of entities bridging multiple topics
- **No entity importance scoring**: Beyond appearance frequency, there's no measure of an entity's centrality or influence in the knowledge graph

### Gap Severity: **Medium**

The entity co-occurrence graph is a solid foundation. Enriching edge types and adding directionality would transform it from a visualization aid into a reasoning substrate.

---

## 2. Thread Lifecycle & Trajectory

### What Nexus Has

- **Four-state lifecycle** (`threads.py`): emerging → active → stale → resolved
- **Lifecycle rules**: emerging = 0-1 days of events, active = 2+ distinct days, stale = 14+ days without new events, resolved = archived
- **Staleness detection** (`store.py`): Query for threads where last event > 14 days ago
- **Thread significance**: 1-10 integer score assigned during synthesis
- **Thread matching**: Two-stage (entity overlap Jaccard + optional LLM confirmation)
- **Synthesis diffing** (`diff.py`): Computes new/updated/resolved threads, new convergence/divergence, entity emergence between synthesis snapshots

### What's Missing for Projection

- **No significance history**: `threads.updated_at` is overwritten on each update, destroying the temporal record. There is no way to ask "was this thread's significance rising or falling over the past week?"
- **No velocity metric**: Events per day for a thread is not tracked. A thread gaining 5 events in one day vs. 1 event over 5 days looks identical
- **No acceleration detection**: Can't identify threads that are gaining momentum faster than baseline
- **No predicted status transitions**: Can't estimate when an emerging thread will become active, or when an active thread will go stale
- **No trajectory classification**: No concept of "accelerating", "steady", "decelerating", or "about to break"
- **Daily synthesis snapshots** (`syntheses` table) store full JSON blobs but are not queryable for trend analysis — they're opaque serialized objects

### Gap Severity: **High**

This is the highest-value gap to close. Thread trajectory is the most natural projection Nexus can make ("this story is accelerating") and requires no new LLM calls — pure SQL analytics on data Nexus already captures, if we add a `thread_snapshots` table.

---

## 3. Temporal Data & Time-Series

### What Nexus Has

- **Event dates**: Daily granularity (`events.date` as YYYY-MM-DD)
- **Weekly summaries**: LLM-compressed narratives for weeks >7 days old (`compression.py`)
- **Monthly summaries**: Schema supports `period_type='monthly'` in `summaries` table
- **14-day event context window**: Pipeline loads last 14 days of events for novelty checking
- **7-day novelty window**: Filter pass 2 checks significance against last 7 days
- **Daily synthesis snapshots**: Serialized `TopicSynthesis` objects in `syntheses` table

### What's Missing for Projection

- **No time-series analytics**: No computation of event frequency trends, significance distribution over time, or entity appearance rates
- **Summaries are text-only**: Weekly/monthly summaries preserve narrative but discard structured metrics (event count, average significance, entity distribution). The `Summary` model has `event_count` but no significance stats
- **No concept of "trend"**: The data model has no field or table representing a directional change over time
- **Synthesis snapshots are opaque**: Stored as `data_json` (serialized TopicSynthesis). To compute a trend, you'd need to deserialize multiple snapshots and diff them — expensive and fragile
- **No seasonality or baseline**: No model of "normal" event volume for a topic to detect anomalous spikes

### Gap Severity: **Medium-High**

The raw temporal data exists (events have dates, summaries have period boundaries) but it's not being analyzed as a time-series. Adding structured metrics to the compression step and a `thread_snapshots` table would unlock most of the value.

---

## 4. Convergence/Divergence & Confidence

### What Nexus Has

- **`convergence` table** (`schema.py`): `fact_text` + `confirmed_by` (JSON array of outlet names), linked to topics
- **`divergence` table** (`schema.py`): `shared_event`, `source_a`/`source_b`, `framing_a`/`framing_b`, linked to topics
- **Independence validation** (`events.py`): Convergence only counted if sources have different affiliation OR different country
- **Framing data** (`event_sources` table, v9): Per-source `[tone] focus; actor_framing` tuples
- **Divergence variants**: baseline, broadened, structured, encouraged — configurable via experiment framework
- **Post-synthesis validation**: `_validate_convergence()` strips non-independent convergence entries

### What's Missing for Projection

- **No confidence calibration**: Nexus records that sources agree/disagree but doesn't track accuracy over time. "Reuters and BBC agreed on X" — was X actually true? No feedback loop
- **No divergence resolution tracking**: When two sources disagreed last week, which one turned out to be right? No mechanism to record outcomes
- **No per-source reliability scoring**: All outlets are treated equally. A source that has been correct 90% of the time on a topic should carry more weight than one at 50%
- **No systematic pattern extraction**: Can't answer "how often do state-affiliated sources diverge from private media on topic Y?" even though the data to compute this exists
- **No confidence scores on projections**: Forward-looking statements need calibrated uncertainty ("70% likely based on X signals") — no infrastructure for this

### Gap Severity: **Medium**

The convergence/divergence machinery is sophisticated. The main gap is the feedback loop — recording whether convergent facts proved true and whether divergent framings resolved. This requires time (weeks of accumulated outcomes) before it becomes useful.

---

## 5. Cross-Topic Correlation

### What Nexus Has

- **`thread_topics` table** (`schema.py`): Many-to-many between threads and topics, enabling a single thread to span multiple topics
- **Global entity table**: `entities` has no topic_slug column — entity canonicalization is global
- **`get_all_events()`** (`store.py`): Can retrieve events across all topics
- **Topic-scoped pipeline**: `run_pipeline()` in `pipeline.py` processes each topic sequentially and independently

### What's Missing for Projection

- **No automatic cross-topic correlation detection**: Even though entities are global and `thread_topics` supports multi-topic threads, the pipeline never looks across topics to find connections
- **No entity bridging analysis**: Which entities appear in events across 2+ topics? This is a simple SQL query (`event_entities JOIN events GROUP BY entity_id HAVING COUNT(DISTINCT topic_slug) >= 2`) but it doesn't exist
- **No "ripple effect" detection**: An event in topic A (e.g., "US sanctions on China") may cause related events in topic B (e.g., "semiconductor supply chain disruption") — no mechanism to detect this
- **No temporal coincidence analysis**: Events about the same entity in different topics within a short window may be correlated — not checked
- **Pipeline isolation**: Each topic's synthesis is independent. The pipeline has no post-processing step that looks across all topics

### Gap Severity: **High**

This is the second highest-value gap. The data foundation already supports cross-topic analysis (global entities, `thread_topics` table), but the pipeline never exercises this capability. A post-pipeline cross-topic correlation pass would be high-value and relatively straightforward to implement.

---

## 6. Forward-Looking Capabilities

### What Nexus Has

**Nothing explicit.** The system is designed to report on what happened, not predict what's next.

The closest existing features:
- **Breaking news alerts** (`breaking.py`): Signal unexpected developments, but reactive not predictive
- **Divergence detection**: Highlights where sources disagree — useful for identifying uncertainty, but doesn't extrapolate
- **Q&A agent** (`qa.py`): Allows ad-hoc queries that might reveal patterns, but doesn't proactively generate insights
- **Synthesis diffing** (`diff.py`): Shows what changed since last run — a "delta view" but not a "forward view"
- **`relation_to_prior`**: Events include natural language text describing how they relate to prior events — this is implicit causal reasoning but unstructured

### What's Missing for Projection

- **No trend extrapolation**: Can't project where a thread is heading based on its trajectory
- **No causal modeling**: Relationships between events are implicit (co-occurrence, natural language `relation_to_prior`) not structured
- **No impact assessment**: No modeling of cascading effects or second-order consequences
- **No risk scoring**: No evaluation of potential negative outcomes or threat levels
- **No automated recommendations**: System reports what happened, never suggests what to do
- **No scenario generation**: No "if X happens, then Y is likely" reasoning
- **No anomaly detection**: No statistical models to flag unusual activity relative to baseline

### Gap Severity: **Critical** (this is the entire feature request)

---

## Summary: Data Readiness Scores

How much of the data foundation already exists for each projection capability:

| Projection Capability | Readiness | Key Enabler Already Present | Primary Gap |
|---|---|---|---|
| Thread trajectory analysis | **60%** | Thread lifecycle, significance scores, event dates | No significance history table |
| Causal chain extraction | **30%** | `relation_to_prior` text field, entity graph | No structured causal edges |
| Cross-topic correlation | **50%** | Global entity table, `thread_topics` join | Pipeline processes topics independently |
| Confidence calibration | **40%** | Convergence/divergence tables, source metadata | No outcome tracking or feedback loop |
| Impact projection | **20%** | Event significance scores, entity roles | No forward-looking data model |
| Anomaly detection | **45%** | Significance scores, event counts, breaking alerts | No statistical baselines |
| Source reliability | **35%** | Per-source framing data, affiliation tracking | No accuracy history |
| Scenario generation | **15%** | LLMClient infrastructure, knowledge context | No scenario framework |

**Overall assessment**: Nexus is approximately **40% of the way** to supporting projection capabilities, with the strongest foundations in entity tracking and thread lifecycle. The highest-ROI investments are thread trajectory (close the 40% gap with one new table) and cross-topic correlation (close the 50% gap with a post-pipeline analysis pass).
