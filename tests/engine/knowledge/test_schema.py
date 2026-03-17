"""Tests for knowledge graph SQLite schema."""

import pytest
import aiosqlite
from nexus.engine.knowledge.schema import initialize_schema, get_schema_version, CURRENT_VERSION


@pytest.fixture
async def db(tmp_path):
    """Create an in-memory database for testing."""
    conn = await aiosqlite.connect(str(tmp_path / "test.db"))
    yield conn
    await conn.close()


async def test_initialize_creates_all_tables(db):
    await initialize_schema(db)
    cursor = await db.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    )
    tables = {row[0] for row in await cursor.fetchall()}
    expected = {
        "schema_version", "entities", "events", "event_entities",
        "event_sources", "threads", "thread_events", "thread_topics",
        "convergence", "divergence", "summaries", "pages", "syntheses",
        "thread_snapshots", "causal_links", "cross_topic_signals",
        "projections", "projection_items", "projection_outcomes",
        "forecast_runs", "forecast_questions", "forecast_scenarios",
        "forecast_resolutions", "forecast_mappings",
    }
    assert expected.issubset(tables)


async def test_initialize_is_idempotent(db):
    await initialize_schema(db)
    await initialize_schema(db)
    version = await get_schema_version(db)
    assert version == CURRENT_VERSION


async def test_schema_version_set(db):
    await initialize_schema(db)
    version = await get_schema_version(db)
    assert version == CURRENT_VERSION


async def test_forecast_readiness_columns_exist(db):
    await initialize_schema(db)
    cursor = await db.execute("PRAGMA table_info(forecast_questions)")
    forecast_question_columns = {row[1] for row in await cursor.fetchall()}
    cursor = await db.execute("PRAGMA table_info(forecast_mappings)")
    forecast_mapping_columns = {row[1] for row in await cursor.fetchall()}
    assert "forecast_key" in forecast_question_columns
    assert "forecast_key" in forecast_mapping_columns


async def test_foreign_keys_enforced(db):
    """Inserting into event_sources with a bad event_id should fail."""
    await initialize_schema(db)
    await db.execute("PRAGMA foreign_keys=ON")
    with pytest.raises(aiosqlite.IntegrityError):
        await db.execute(
            "INSERT INTO event_sources (event_id, url) VALUES (9999, 'http://x')"
        )


async def test_entity_type_constraint(db):
    await initialize_schema(db)
    with pytest.raises(aiosqlite.IntegrityError):
        await db.execute(
            "INSERT INTO entities (canonical_name, entity_type, first_seen, last_seen) "
            "VALUES ('Test', 'invalid_type', '2026-01-01', '2026-01-01')"
        )


async def test_thread_status_constraint(db):
    await initialize_schema(db)
    with pytest.raises(aiosqlite.IntegrityError):
        await db.execute(
            "INSERT INTO threads (slug, headline, status) "
            "VALUES ('test', 'Test', 'invalid_status')"
        )


async def test_page_type_constraint(db):
    await initialize_schema(db)
    with pytest.raises(aiosqlite.IntegrityError):
        await db.execute(
            "INSERT INTO pages (slug, title, page_type, content_md, stale_after) "
            "VALUES ('test', 'Test', 'invalid_type', 'content', '2026-01-01')"
        )


async def test_projection_page_type_allowed(db):
    await initialize_schema(db)
    await db.execute(
        "INSERT INTO pages (slug, title, page_type, content_md, stale_after) "
        "VALUES ('projection:test', 'Test', 'projection', 'content', '2026-01-01')"
    )


async def test_entity_unique_name(db):
    await initialize_schema(db)
    await db.execute(
        "INSERT INTO entities (canonical_name, entity_type, first_seen, last_seen) "
        "VALUES ('IAEA', 'org', '2026-01-01', '2026-01-01')"
    )
    with pytest.raises(aiosqlite.IntegrityError):
        await db.execute(
            "INSERT INTO entities (canonical_name, entity_type, first_seen, last_seen) "
            "VALUES ('IAEA', 'org', '2026-01-01', '2026-01-01')"
        )
