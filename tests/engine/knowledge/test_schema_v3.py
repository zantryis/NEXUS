"""Tests for schema v3 migration (breaking_alerts + feedback)."""

import pytest
import aiosqlite
from pathlib import Path

from nexus.engine.knowledge.schema import initialize_schema, get_schema_version, CURRENT_VERSION


async def test_v3_migration_creates_tables(tmp_path):
    """Schema v3 should create breaking_alerts and feedback tables."""
    db_path = tmp_path / "test.db"
    async with aiosqlite.connect(str(db_path)) as db:
        await initialize_schema(db)

        version = await get_schema_version(db)
        assert version == CURRENT_VERSION
        assert version >= 3

        # Verify tables exist
        cursor = await db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name IN "
            "('breaking_alerts', 'feedback') ORDER BY name"
        )
        tables = [r[0] for r in await cursor.fetchall()]
        assert "breaking_alerts" in tables
        assert "feedback" in tables
