"""SQLite schema for the knowledge graph."""

import logging

import aiosqlite

logger = logging.getLogger(__name__)

CURRENT_VERSION = 7

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
    language TEXT NOT NULL DEFAULT 'en'
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
        CHECK(page_type IN ('backstory','entity_profile','thread_deepdive','weekly_recap')),
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

    if current < CURRENT_VERSION:
        await db.execute(
            "INSERT INTO schema_version (version) VALUES (?)",
            (CURRENT_VERSION,),
        )
        await db.commit()
        logger.info(f"Knowledge schema initialized at version {CURRENT_VERSION}")
    else:
        logger.debug(f"Knowledge schema already at version {current}")


async def get_schema_version(db: aiosqlite.Connection) -> int:
    """Return the current schema version."""
    cursor = await db.execute("SELECT MAX(version) FROM schema_version")
    row = await cursor.fetchone()
    return row[0] if row[0] is not None else 0
