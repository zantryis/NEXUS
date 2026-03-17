"""SQLite schema for the knowledge graph."""

import logging

import aiosqlite

logger = logging.getLogger(__name__)

CURRENT_VERSION = 15

DDL = """
-- Schema version tracking
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Canonical entities (graph nodes)
CREATE TABLE IF NOT EXISTS entities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    canonical_name TEXT NOT NULL UNIQUE,
    entity_type TEXT NOT NULL DEFAULT 'unknown'
        CHECK(entity_type IN ('person','org','country','treaty','concept','unknown')),
    aliases TEXT NOT NULL DEFAULT '[]',
    first_seen TEXT NOT NULL,
    last_seen TEXT NOT NULL
);

-- Events (core knowledge unit)
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,
    summary TEXT NOT NULL,
    significance INTEGER NOT NULL DEFAULT 5,
    relation_to_prior TEXT NOT NULL DEFAULT '',
    raw_entities TEXT NOT NULL DEFAULT '[]',
    topic_slug TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Event-entity many-to-many
CREATE TABLE IF NOT EXISTS event_entities (
    event_id INTEGER NOT NULL REFERENCES events(id),
    entity_id INTEGER NOT NULL REFERENCES entities(id),
    role TEXT NOT NULL DEFAULT 'mentioned'
        CHECK(role IN ('subject','object','mentioned')),
    PRIMARY KEY (event_id, entity_id)
);

-- Event sources (normalized)
CREATE TABLE IF NOT EXISTS event_sources (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id INTEGER NOT NULL REFERENCES events(id),
    url TEXT NOT NULL,
    outlet TEXT NOT NULL DEFAULT '',
    affiliation TEXT NOT NULL DEFAULT '',
    country TEXT NOT NULL DEFAULT '',
    language TEXT NOT NULL DEFAULT 'en',
    framing TEXT NOT NULL DEFAULT ''
);

-- Persistent narrative threads
CREATE TABLE IF NOT EXISTS threads (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    slug TEXT NOT NULL UNIQUE,
    headline TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'emerging'
        CHECK(status IN ('emerging','active','stale','resolved')),
    significance INTEGER NOT NULL DEFAULT 5,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Thread-event many-to-many
CREATE TABLE IF NOT EXISTS thread_events (
    thread_id INTEGER NOT NULL REFERENCES threads(id),
    event_id INTEGER NOT NULL REFERENCES events(id),
    added_date TEXT NOT NULL DEFAULT (date('now')),
    PRIMARY KEY (thread_id, event_id)
);

-- Thread-topic many-to-many (cross-topic threads)
CREATE TABLE IF NOT EXISTS thread_topics (
    thread_id INTEGER NOT NULL REFERENCES threads(id),
    topic_slug TEXT NOT NULL,
    PRIMARY KEY (thread_id, topic_slug)
);

-- Cross-source fact confirmation
CREATE TABLE IF NOT EXISTS convergence (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id INTEGER NOT NULL REFERENCES threads(id),
    fact_text TEXT NOT NULL,
    confirmed_by TEXT NOT NULL DEFAULT '[]'
);

-- Conflicting framing records
CREATE TABLE IF NOT EXISTS divergence (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id INTEGER NOT NULL REFERENCES threads(id),
    shared_event TEXT NOT NULL,
    source_a TEXT NOT NULL,
    framing_a TEXT NOT NULL,
    source_b TEXT NOT NULL,
    framing_b TEXT NOT NULL
);

-- Compressed period summaries
CREATE TABLE IF NOT EXISTS summaries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    topic_slug TEXT NOT NULL,
    period_type TEXT NOT NULL CHECK(period_type IN ('weekly','monthly')),
    period_start TEXT NOT NULL,
    period_end TEXT NOT NULL,
    text TEXT NOT NULL,
    event_count INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Cached LLM-generated narrative pages
CREATE TABLE IF NOT EXISTS pages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    slug TEXT NOT NULL UNIQUE,
    title TEXT NOT NULL,
    page_type TEXT NOT NULL
        CHECK(page_type IN ('backstory','entity_profile','thread_deepdive','weekly_recap','projection')),
    topic_slug TEXT,
    content_md TEXT NOT NULL,
    generated_at TEXT NOT NULL DEFAULT (datetime('now')),
    stale_after TEXT NOT NULL,
    prompt_hash TEXT NOT NULL DEFAULT ''
);

-- Daily synthesis snapshots
CREATE TABLE IF NOT EXISTS syntheses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    topic_slug TEXT NOT NULL,
    date TEXT NOT NULL,
    data_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Projection support (v10)
CREATE TABLE IF NOT EXISTS thread_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id INTEGER NOT NULL REFERENCES threads(id),
    snapshot_date TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'emerging',
    significance INTEGER NOT NULL DEFAULT 5,
    event_count INTEGER NOT NULL DEFAULT 0,
    latest_event_date TEXT,
    velocity_7d REAL NOT NULL DEFAULT 0.0,
    acceleration_7d REAL NOT NULL DEFAULT 0.0,
    significance_trend_7d REAL NOT NULL DEFAULT 0.0,
    momentum_score REAL NOT NULL DEFAULT 0.0,
    trajectory_label TEXT NOT NULL DEFAULT 'steady',
    UNIQUE(thread_id, snapshot_date)
);

CREATE TABLE IF NOT EXISTS causal_links (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_event_id INTEGER NOT NULL REFERENCES events(id),
    target_event_id INTEGER NOT NULL REFERENCES events(id),
    relation_type TEXT NOT NULL DEFAULT 'follow_on',
    evidence_text TEXT NOT NULL DEFAULT '',
    strength REAL NOT NULL DEFAULT 0.5,
    UNIQUE(source_event_id, target_event_id, relation_type)
);

CREATE TABLE IF NOT EXISTS cross_topic_signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    topic_slug TEXT NOT NULL,
    related_topic_slug TEXT NOT NULL,
    shared_entity TEXT NOT NULL,
    signal_type TEXT NOT NULL DEFAULT 'entity_bridge',
    observed_at TEXT NOT NULL,
    event_ids_json TEXT NOT NULL DEFAULT '[]',
    related_event_ids_json TEXT NOT NULL DEFAULT '[]',
    note TEXT NOT NULL DEFAULT '',
    UNIQUE(topic_slug, related_topic_slug, shared_entity, observed_at)
);

CREATE TABLE IF NOT EXISTS projections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    topic_slug TEXT NOT NULL,
    topic_name TEXT NOT NULL,
    engine TEXT NOT NULL DEFAULT 'native',
    generated_for TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'ready',
    summary TEXT NOT NULL DEFAULT '',
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS projection_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    projection_id INTEGER NOT NULL REFERENCES projections(id),
    claim TEXT NOT NULL,
    confidence TEXT NOT NULL DEFAULT 'medium',
    horizon_days INTEGER NOT NULL DEFAULT 7,
    signpost TEXT NOT NULL,
    signals_cited_json TEXT NOT NULL DEFAULT '[]',
    evidence_event_ids_json TEXT NOT NULL DEFAULT '[]',
    evidence_thread_ids_json TEXT NOT NULL DEFAULT '[]',
    review_after TEXT NOT NULL,
    external_ref TEXT
);

CREATE TABLE IF NOT EXISTS projection_outcomes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    projection_item_id INTEGER NOT NULL REFERENCES projection_items(id),
    outcome_status TEXT NOT NULL DEFAULT 'pending',
    score REAL,
    notes TEXT NOT NULL DEFAULT '',
    reviewed_at TEXT,
    external_ref TEXT
);

CREATE TABLE IF NOT EXISTS forecast_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    topic_slug TEXT NOT NULL,
    topic_name TEXT NOT NULL,
    engine TEXT NOT NULL DEFAULT 'native',
    generated_for TEXT NOT NULL,
    summary TEXT NOT NULL DEFAULT '',
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS forecast_questions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    forecast_run_id INTEGER NOT NULL REFERENCES forecast_runs(id),
    forecast_key TEXT NOT NULL DEFAULT '',
    question TEXT NOT NULL,
    forecast_type TEXT NOT NULL DEFAULT 'binary',
    target_variable TEXT NOT NULL,
    target_metadata_json TEXT NOT NULL DEFAULT '{}',
    probability REAL NOT NULL DEFAULT 0.5,
    base_rate REAL NOT NULL DEFAULT 0.5,
    resolution_criteria TEXT NOT NULL,
    resolution_date TEXT NOT NULL,
    horizon_days INTEGER NOT NULL DEFAULT 7,
    signpost TEXT NOT NULL,
    expected_direction TEXT,
    signals_cited_json TEXT NOT NULL DEFAULT '[]',
    evidence_event_ids_json TEXT NOT NULL DEFAULT '[]',
    evidence_thread_ids_json TEXT NOT NULL DEFAULT '[]',
    cross_topic_signal_ids_json TEXT NOT NULL DEFAULT '[]',
    status TEXT NOT NULL DEFAULT 'open',
    external_ref TEXT
);

CREATE TABLE IF NOT EXISTS forecast_scenarios (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    forecast_run_id INTEGER NOT NULL REFERENCES forecast_runs(id),
    scenario_key TEXT NOT NULL,
    label TEXT NOT NULL,
    probability REAL NOT NULL DEFAULT 0.5,
    description TEXT NOT NULL DEFAULT '',
    signposts_json TEXT NOT NULL DEFAULT '[]',
    status TEXT NOT NULL DEFAULT 'open'
);

CREATE TABLE IF NOT EXISTS forecast_resolutions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    forecast_question_id INTEGER NOT NULL REFERENCES forecast_questions(id),
    outcome_status TEXT NOT NULL DEFAULT 'pending',
    resolved_bool INTEGER,
    realized_direction TEXT,
    actual_value REAL,
    brier_score REAL,
    log_loss REAL,
    notes TEXT NOT NULL DEFAULT '',
    resolved_at TEXT,
    external_ref TEXT
);

CREATE TABLE IF NOT EXISTS forecast_mappings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    forecast_question_id INTEGER NOT NULL REFERENCES forecast_questions(id),
    forecast_key TEXT NOT NULL DEFAULT '',
    mapping_type TEXT NOT NULL DEFAULT 'external_ref',
    external_ref TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_events_date ON events(date);
CREATE INDEX IF NOT EXISTS idx_events_topic_date ON events(topic_slug, date);
CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(canonical_name);
CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(entity_type);
CREATE INDEX IF NOT EXISTS idx_event_entities_entity ON event_entities(entity_id);
CREATE INDEX IF NOT EXISTS idx_event_sources_event ON event_sources(event_id);
CREATE INDEX IF NOT EXISTS idx_threads_status ON threads(status);
CREATE INDEX IF NOT EXISTS idx_thread_events_thread ON thread_events(thread_id);
CREATE INDEX IF NOT EXISTS idx_thread_events_event ON thread_events(event_id);
CREATE INDEX IF NOT EXISTS idx_thread_topics_topic ON thread_topics(topic_slug);
CREATE INDEX IF NOT EXISTS idx_pages_slug ON pages(slug);
CREATE INDEX IF NOT EXISTS idx_pages_stale ON pages(stale_after);
CREATE INDEX IF NOT EXISTS idx_summaries_topic ON summaries(topic_slug, period_type);
CREATE INDEX IF NOT EXISTS idx_syntheses_topic_date ON syntheses(topic_slug, date);
CREATE INDEX IF NOT EXISTS idx_thread_snapshots_thread_date ON thread_snapshots(thread_id, snapshot_date);
CREATE INDEX IF NOT EXISTS idx_causal_links_source ON causal_links(source_event_id);
CREATE INDEX IF NOT EXISTS idx_causal_links_target ON causal_links(target_event_id);
CREATE INDEX IF NOT EXISTS idx_cross_topic_signals_topic_date ON cross_topic_signals(topic_slug, observed_at);
CREATE INDEX IF NOT EXISTS idx_projections_topic_date ON projections(topic_slug, generated_for);
CREATE INDEX IF NOT EXISTS idx_projection_items_projection ON projection_items(projection_id);
CREATE INDEX IF NOT EXISTS idx_projection_outcomes_item ON projection_outcomes(projection_item_id);
CREATE INDEX IF NOT EXISTS idx_forecast_runs_topic_date ON forecast_runs(topic_slug, generated_for);
CREATE INDEX IF NOT EXISTS idx_forecast_questions_run ON forecast_questions(forecast_run_id);
CREATE INDEX IF NOT EXISTS idx_forecast_questions_resolution_date ON forecast_questions(resolution_date);
CREATE INDEX IF NOT EXISTS idx_forecast_resolutions_question ON forecast_resolutions(forecast_question_id);
CREATE INDEX IF NOT EXISTS idx_forecast_mappings_question ON forecast_mappings(forecast_question_id);
"""


MIGRATION_V2 = """
-- Filter decision log (v2)
CREATE TABLE IF NOT EXISTS filter_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_date TEXT NOT NULL,
    topic_slug TEXT NOT NULL,
    url TEXT NOT NULL,
    title TEXT NOT NULL,
    source_id TEXT NOT NULL DEFAULT '',
    source_affiliation TEXT NOT NULL DEFAULT '',
    source_country TEXT NOT NULL DEFAULT '',
    relevance_score REAL,
    relevance_reason TEXT NOT NULL DEFAULT '',
    passed_pass1 INTEGER NOT NULL DEFAULT 0,
    significance_score REAL,
    is_novel INTEGER,
    significance_reason TEXT NOT NULL DEFAULT '',
    passed_pass2 INTEGER,
    final_score REAL,
    outcome TEXT NOT NULL DEFAULT 'rejected'
        CHECK(outcome IN ('accepted','rejected_relevance','rejected_significance','rejected_diversity')),
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_filter_log_run ON filter_log(run_date, topic_slug);
CREATE INDEX IF NOT EXISTS idx_filter_log_outcome ON filter_log(outcome);
"""


MIGRATION_V3 = """
-- Breaking news dedup table (v3)
CREATE TABLE IF NOT EXISTS breaking_alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    headline_hash TEXT NOT NULL UNIQUE,
    headline TEXT NOT NULL,
    source_url TEXT NOT NULL,
    significance_score INTEGER NOT NULL,
    alerted_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- User feedback on briefings (v3)
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    briefing_date TEXT NOT NULL,
    rating TEXT NOT NULL CHECK(rating IN ('up', 'down')),
    comment TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_breaking_alerts_hash ON breaking_alerts(headline_hash);
CREATE INDEX IF NOT EXISTS idx_feedback_date ON feedback(briefing_date);
"""


MIGRATION_V4 = """
-- Usage log for LLM cost tracking (v4)
CREATE TABLE IF NOT EXISTS usage_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    config_key TEXT NOT NULL,
    input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    cost_usd REAL NOT NULL DEFAULT 0.0,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_usage_log_date ON usage_log(date);
"""


MIGRATION_V5 = """
-- Entity thumbnails (v5)
ALTER TABLE entities ADD COLUMN thumbnail_url TEXT NOT NULL DEFAULT '';
"""

MIGRATION_V6 = """
-- Entity Wikipedia URLs (v6)
ALTER TABLE entities ADD COLUMN wikipedia_url TEXT NOT NULL DEFAULT '';
"""

MIGRATION_V7 = """
-- Topic-scoped breaking alerts (v7): recreate table with composite unique
CREATE TABLE IF NOT EXISTS breaking_alerts_new (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    headline_hash TEXT NOT NULL,
    headline TEXT NOT NULL,
    source_url TEXT NOT NULL,
    significance_score INTEGER NOT NULL,
    topic_slug TEXT NOT NULL DEFAULT '',
    alerted_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(headline_hash, topic_slug)
);
INSERT OR IGNORE INTO breaking_alerts_new (id, headline_hash, headline, source_url, significance_score, alerted_at)
    SELECT id, headline_hash, headline, source_url, significance_score, alerted_at FROM breaking_alerts;
DROP TABLE IF EXISTS breaking_alerts;
ALTER TABLE breaking_alerts_new RENAME TO breaking_alerts;
CREATE INDEX IF NOT EXISTS idx_breaking_alerts_hash_topic ON breaking_alerts(headline_hash, topic_slug);
"""


MIGRATION_V8 = """
-- Pipeline run tracking (v8)
CREATE TABLE IF NOT EXISTS pipeline_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT NOT NULL DEFAULT (datetime('now')),
    completed_at TEXT,
    status TEXT NOT NULL DEFAULT 'running'
        CHECK(status IN ('running','completed','failed')),
    topics TEXT NOT NULL DEFAULT '[]',
    article_count INTEGER NOT NULL DEFAULT 0,
    event_count INTEGER NOT NULL DEFAULT 0,
    cost_usd REAL NOT NULL DEFAULT 0.0,
    error TEXT,
    trigger TEXT NOT NULL DEFAULT 'manual'
        CHECK(trigger IN ('manual','scheduled','auto_run','smoke'))
);
CREATE INDEX IF NOT EXISTS idx_pipeline_runs_status ON pipeline_runs(status);
CREATE INDEX IF NOT EXISTS idx_pipeline_runs_started ON pipeline_runs(started_at);
"""


MIGRATION_V9 = """
-- Per-source editorial framing (v9)
ALTER TABLE event_sources ADD COLUMN framing TEXT NOT NULL DEFAULT '';
"""


MIGRATION_V10 = """
-- Projection substrate (v10)
ALTER TABLE events ADD COLUMN raw_entities TEXT NOT NULL DEFAULT '[]';

CREATE TABLE IF NOT EXISTS thread_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id INTEGER NOT NULL REFERENCES threads(id),
    snapshot_date TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'emerging',
    significance INTEGER NOT NULL DEFAULT 5,
    event_count INTEGER NOT NULL DEFAULT 0,
    latest_event_date TEXT,
    velocity_7d REAL NOT NULL DEFAULT 0.0,
    acceleration_7d REAL NOT NULL DEFAULT 0.0,
    significance_trend_7d REAL NOT NULL DEFAULT 0.0,
    momentum_score REAL NOT NULL DEFAULT 0.0,
    trajectory_label TEXT NOT NULL DEFAULT 'steady',
    UNIQUE(thread_id, snapshot_date)
);

CREATE TABLE IF NOT EXISTS causal_links (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_event_id INTEGER NOT NULL REFERENCES events(id),
    target_event_id INTEGER NOT NULL REFERENCES events(id),
    relation_type TEXT NOT NULL DEFAULT 'follow_on',
    evidence_text TEXT NOT NULL DEFAULT '',
    strength REAL NOT NULL DEFAULT 0.5,
    UNIQUE(source_event_id, target_event_id, relation_type)
);

CREATE TABLE IF NOT EXISTS cross_topic_signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    topic_slug TEXT NOT NULL,
    related_topic_slug TEXT NOT NULL,
    shared_entity TEXT NOT NULL,
    signal_type TEXT NOT NULL DEFAULT 'entity_bridge',
    observed_at TEXT NOT NULL,
    event_ids_json TEXT NOT NULL DEFAULT '[]',
    related_event_ids_json TEXT NOT NULL DEFAULT '[]',
    note TEXT NOT NULL DEFAULT '',
    UNIQUE(topic_slug, related_topic_slug, shared_entity, observed_at)
);

CREATE TABLE IF NOT EXISTS projections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    topic_slug TEXT NOT NULL,
    topic_name TEXT NOT NULL,
    engine TEXT NOT NULL DEFAULT 'native',
    generated_for TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'ready',
    summary TEXT NOT NULL DEFAULT '',
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS projection_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    projection_id INTEGER NOT NULL REFERENCES projections(id),
    claim TEXT NOT NULL,
    confidence TEXT NOT NULL DEFAULT 'medium',
    horizon_days INTEGER NOT NULL DEFAULT 7,
    signpost TEXT NOT NULL,
    signals_cited_json TEXT NOT NULL DEFAULT '[]',
    evidence_event_ids_json TEXT NOT NULL DEFAULT '[]',
    evidence_thread_ids_json TEXT NOT NULL DEFAULT '[]',
    review_after TEXT NOT NULL,
    external_ref TEXT
);

CREATE TABLE IF NOT EXISTS projection_outcomes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    projection_item_id INTEGER NOT NULL REFERENCES projection_items(id),
    outcome_status TEXT NOT NULL DEFAULT 'pending',
    score REAL,
    notes TEXT NOT NULL DEFAULT '',
    reviewed_at TEXT,
    external_ref TEXT
);

CREATE TABLE IF NOT EXISTS pages_new (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    slug TEXT NOT NULL UNIQUE,
    title TEXT NOT NULL,
    page_type TEXT NOT NULL
        CHECK(page_type IN ('backstory','entity_profile','thread_deepdive','weekly_recap','projection')),
    topic_slug TEXT,
    content_md TEXT NOT NULL,
    generated_at TEXT NOT NULL DEFAULT (datetime('now')),
    stale_after TEXT NOT NULL,
    prompt_hash TEXT NOT NULL DEFAULT ''
);
INSERT OR IGNORE INTO pages_new (id, slug, title, page_type, topic_slug, content_md, generated_at, stale_after, prompt_hash)
    SELECT id, slug, title, page_type, topic_slug, content_md, generated_at, stale_after, prompt_hash FROM pages;
DROP TABLE IF EXISTS pages;
ALTER TABLE pages_new RENAME TO pages;

CREATE INDEX IF NOT EXISTS idx_thread_snapshots_thread_date ON thread_snapshots(thread_id, snapshot_date);
CREATE INDEX IF NOT EXISTS idx_causal_links_source ON causal_links(source_event_id);
CREATE INDEX IF NOT EXISTS idx_causal_links_target ON causal_links(target_event_id);
CREATE INDEX IF NOT EXISTS idx_cross_topic_signals_topic_date ON cross_topic_signals(topic_slug, observed_at);
CREATE INDEX IF NOT EXISTS idx_projections_topic_date ON projections(topic_slug, generated_for);
CREATE INDEX IF NOT EXISTS idx_projection_items_projection ON projection_items(projection_id);
CREATE INDEX IF NOT EXISTS idx_projection_outcomes_item ON projection_outcomes(projection_item_id);
CREATE INDEX IF NOT EXISTS idx_pages_slug ON pages(slug);
CREATE INDEX IF NOT EXISTS idx_pages_stale ON pages(stale_after);
"""


MIGRATION_V11 = """
-- Quantified forecast substrate (v11)
CREATE TABLE IF NOT EXISTS forecast_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    topic_slug TEXT NOT NULL,
    topic_name TEXT NOT NULL,
    engine TEXT NOT NULL DEFAULT 'native',
    generated_for TEXT NOT NULL,
    summary TEXT NOT NULL DEFAULT '',
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS forecast_questions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    forecast_run_id INTEGER NOT NULL REFERENCES forecast_runs(id),
    forecast_key TEXT NOT NULL DEFAULT '',
    question TEXT NOT NULL,
    forecast_type TEXT NOT NULL DEFAULT 'binary',
    target_variable TEXT NOT NULL,
    target_metadata_json TEXT NOT NULL DEFAULT '{}',
    probability REAL NOT NULL DEFAULT 0.5,
    base_rate REAL NOT NULL DEFAULT 0.5,
    resolution_criteria TEXT NOT NULL,
    resolution_date TEXT NOT NULL,
    horizon_days INTEGER NOT NULL DEFAULT 7,
    signpost TEXT NOT NULL,
    expected_direction TEXT,
    signals_cited_json TEXT NOT NULL DEFAULT '[]',
    evidence_event_ids_json TEXT NOT NULL DEFAULT '[]',
    evidence_thread_ids_json TEXT NOT NULL DEFAULT '[]',
    cross_topic_signal_ids_json TEXT NOT NULL DEFAULT '[]',
    status TEXT NOT NULL DEFAULT 'open',
    external_ref TEXT
);

CREATE TABLE IF NOT EXISTS forecast_scenarios (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    forecast_run_id INTEGER NOT NULL REFERENCES forecast_runs(id),
    scenario_key TEXT NOT NULL,
    label TEXT NOT NULL,
    probability REAL NOT NULL DEFAULT 0.5,
    description TEXT NOT NULL DEFAULT '',
    signposts_json TEXT NOT NULL DEFAULT '[]',
    status TEXT NOT NULL DEFAULT 'open'
);

CREATE TABLE IF NOT EXISTS forecast_resolutions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    forecast_question_id INTEGER NOT NULL REFERENCES forecast_questions(id),
    outcome_status TEXT NOT NULL DEFAULT 'pending',
    resolved_bool INTEGER,
    realized_direction TEXT,
    actual_value REAL,
    brier_score REAL,
    log_loss REAL,
    notes TEXT NOT NULL DEFAULT '',
    resolved_at TEXT,
    external_ref TEXT
);

CREATE TABLE IF NOT EXISTS forecast_mappings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    forecast_question_id INTEGER NOT NULL REFERENCES forecast_questions(id),
    forecast_key TEXT NOT NULL DEFAULT '',
    mapping_type TEXT NOT NULL DEFAULT 'external_ref',
    external_ref TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_forecast_runs_topic_date ON forecast_runs(topic_slug, generated_for);
CREATE INDEX IF NOT EXISTS idx_forecast_questions_run ON forecast_questions(forecast_run_id);
CREATE INDEX IF NOT EXISTS idx_forecast_questions_key ON forecast_questions(forecast_key);
CREATE INDEX IF NOT EXISTS idx_forecast_questions_resolution_date ON forecast_questions(resolution_date);
CREATE INDEX IF NOT EXISTS idx_forecast_resolutions_question ON forecast_resolutions(forecast_question_id);
CREATE INDEX IF NOT EXISTS idx_forecast_mappings_question ON forecast_mappings(forecast_question_id);
CREATE INDEX IF NOT EXISTS idx_forecast_mappings_key ON forecast_mappings(forecast_key);
"""


MIGRATION_V12 = """
-- Forecast readiness contract updates (v12)
ALTER TABLE forecast_questions ADD COLUMN forecast_key TEXT NOT NULL DEFAULT '';
ALTER TABLE forecast_mappings ADD COLUMN forecast_key TEXT NOT NULL DEFAULT '';
CREATE INDEX IF NOT EXISTS idx_forecast_questions_key ON forecast_questions(forecast_key);
CREATE INDEX IF NOT EXISTS idx_forecast_mappings_key ON forecast_mappings(forecast_key);
"""

MIGRATION_V13 = """
-- Entity-entity relationships with bi-temporal validity (v13)
CREATE TABLE IF NOT EXISTS entity_relationships (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_entity_id INTEGER NOT NULL REFERENCES entities(id),
    target_entity_id INTEGER NOT NULL REFERENCES entities(id),
    relation_type TEXT NOT NULL,
    evidence_text TEXT NOT NULL DEFAULT '',
    source_event_id INTEGER REFERENCES events(id),
    strength REAL NOT NULL DEFAULT 0.5,
    valid_from TEXT NOT NULL,
    valid_until TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(source_entity_id, target_entity_id, relation_type, source_event_id)
);
CREATE INDEX IF NOT EXISTS idx_er_source ON entity_relationships(source_entity_id);
CREATE INDEX IF NOT EXISTS idx_er_target ON entity_relationships(target_entity_id);
CREATE INDEX IF NOT EXISTS idx_er_valid ON entity_relationships(valid_from, valid_until);
CREATE INDEX IF NOT EXISTS idx_er_type ON entity_relationships(relation_type);
"""


MIGRATION_V14 = """
-- Breaking news feedback (v14)
CREATE TABLE IF NOT EXISTS breaking_feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    headline_hash TEXT NOT NULL,
    topic_slug TEXT NOT NULL,
    feedback TEXT NOT NULL CHECK(feedback IN ('useful','not_breaking')),
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_breaking_feedback_hash ON breaking_feedback(headline_hash);
"""

MIGRATION_V15 = """
-- Feed health tracking (v15)
CREATE TABLE IF NOT EXISTS feed_health (
    source_url TEXT PRIMARY KEY,
    topic_slug TEXT NOT NULL,
    last_poll_at TEXT NOT NULL DEFAULT (datetime('now')),
    last_success_at TEXT,
    consecutive_failures INTEGER NOT NULL DEFAULT 0,
    total_polls INTEGER NOT NULL DEFAULT 0,
    total_successes INTEGER NOT NULL DEFAULT 0,
    last_error TEXT,
    status TEXT NOT NULL DEFAULT 'healthy'
        CHECK(status IN ('healthy','degraded','dead'))
);
CREATE INDEX IF NOT EXISTS idx_feed_health_status ON feed_health(status);
"""


async def initialize_schema(db: aiosqlite.Connection) -> None:
    """Create all tables and indexes. Idempotent."""
    await db.executescript(DDL)

    # Check current version
    cursor = await db.execute(
        "SELECT MAX(version) FROM schema_version"
    )
    row = await cursor.fetchone()
    current = row[0] if row[0] is not None else 0

    # Run migrations
    if current < 2:
        await db.executescript(MIGRATION_V2)
        logger.info("Applied migration v2: filter_log table")

    if current < 3:
        await db.executescript(MIGRATION_V3)
        logger.info("Applied migration v3: breaking_alerts + feedback tables")

    if current < 4:
        await db.executescript(MIGRATION_V4)
        logger.info("Applied migration v4: usage_log table")

    if current < 5:
        await db.executescript(MIGRATION_V5)
        logger.info("Applied migration v5: entity thumbnail_url column")

    if current < 6:
        await db.executescript(MIGRATION_V6)
        logger.info("Applied migration v6: entity wikipedia_url column")

    if current < 7:
        await db.executescript(MIGRATION_V7)
        logger.info("Applied migration v7: topic-scoped breaking_alerts")

    if current < 8:
        await db.executescript(MIGRATION_V8)
        logger.info("Applied migration v8: pipeline_runs table")

    if current < 9 and not await _has_column(db, "event_sources", "framing"):
        await db.executescript(MIGRATION_V9)
        logger.info("Applied migration v9: event_sources framing column")

    if current < 10 and (
        not await _has_column(db, "events", "raw_entities")
        or not await _has_table(db, "thread_snapshots")
        or not await _has_table(db, "projections")
    ):
        await db.executescript(MIGRATION_V10)
        logger.info("Applied migration v10: future projection substrate")

    if current < 11 and not await _has_table(db, "forecast_runs"):
        await db.executescript(MIGRATION_V11)
        logger.info("Applied migration v11: quantified forecast substrate")

    if current < 12 and (
        not await _has_column(db, "forecast_questions", "forecast_key")
        or not await _has_column(db, "forecast_mappings", "forecast_key")
    ):
        await db.executescript(MIGRATION_V12)
        logger.info("Applied migration v12: forecast readiness contract columns")

    if current < 13 and not await _has_table(db, "entity_relationships"):
        await db.executescript(MIGRATION_V13)
        logger.info("Applied migration v13: entity-entity relationships")

    if current < 14 and not await _has_table(db, "breaking_feedback"):
        await db.executescript(MIGRATION_V14)
        logger.info("Applied migration v14: breaking_feedback table")

    if current < 15 and not await _has_table(db, "feed_health"):
        await db.executescript(MIGRATION_V15)
        logger.info("Applied migration v15: feed_health table")

    if current < CURRENT_VERSION:
        await db.execute(
            "INSERT OR IGNORE INTO schema_version (version) VALUES (?)",
            (CURRENT_VERSION,),
        )
        if await _has_column(db, "forecast_questions", "forecast_key"):
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_forecast_questions_key ON forecast_questions(forecast_key)"
            )
        if await _has_column(db, "forecast_mappings", "forecast_key"):
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_forecast_mappings_key ON forecast_mappings(forecast_key)"
            )
        await db.commit()
        logger.info(f"Knowledge schema initialized at version {CURRENT_VERSION}")
    else:
        if await _has_column(db, "forecast_questions", "forecast_key"):
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_forecast_questions_key ON forecast_questions(forecast_key)"
            )
        if await _has_column(db, "forecast_mappings", "forecast_key"):
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_forecast_mappings_key ON forecast_mappings(forecast_key)"
            )
        await db.commit()
        logger.debug(f"Knowledge schema already at version {current}")


async def get_schema_version(db: aiosqlite.Connection) -> int:
    """Return the current schema version."""
    cursor = await db.execute("SELECT MAX(version) FROM schema_version")
    row = await cursor.fetchone()
    return row[0] if row[0] is not None else 0


async def _has_table(db: aiosqlite.Connection, table_name: str) -> bool:
    cursor = await db.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table_name,),
    )
    return (await cursor.fetchone()) is not None


async def _has_column(db: aiosqlite.Connection, table_name: str, column_name: str) -> bool:
    cursor = await db.execute(f"PRAGMA table_info({table_name})")
    return any(row[1] == column_name for row in await cursor.fetchall())
