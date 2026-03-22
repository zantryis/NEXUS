"""SQLite-backed knowledge store. Returns existing Pydantic models."""

import json
import logging
from datetime import date, datetime, timedelta
from pathlib import Path

import aiosqlite

from nexus.engine.knowledge.events import Event
from nexus.engine.knowledge.compression import Summary
from nexus.engine.projection.analytics import compute_snapshot_metrics, detect_cross_topic_signals
from nexus.engine.projection.models import (
    CausalLink,
    CrossTopicSignal,
    ForecastQuestion,
    ForecastResolution,
    ForecastRun,
    ForecastScenario,
    ProjectionItem,
    ProjectionOutcome,
    ThreadSnapshot,
    TopicProjection,
    build_forecast_key,
)
from nexus.engine.knowledge.schema import initialize_schema

logger = logging.getLogger(__name__)


class KnowledgeStore:
    """Async SQLite knowledge store.

    Wraps the knowledge graph DB and returns existing Pydantic models
    so downstream consumers (renderers, judge, metrics) don't change.
    """

    def __init__(self, db_path: Path, *, read_only: bool = False):
        self._db_path = db_path
        self._read_only = read_only
        self._db: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        """Open connection, create tables if needed."""
        if self._read_only:
            self._db = await aiosqlite.connect(f"file:{self._db_path}?mode=ro", uri=True)
        else:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            self._db = await aiosqlite.connect(str(self._db_path))
        self._db.row_factory = aiosqlite.Row
        if not self._read_only:
            await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA busy_timeout=5000")
        await self._db.execute("PRAGMA foreign_keys=ON")
        if not self._read_only:
            await initialize_schema(self._db)

    async def close(self) -> None:
        """Close the database connection."""
        if self._db:
            await self._db.close()
            self._db = None

    @property
    def db(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError("KnowledgeStore not initialized. Call initialize() first.")
        return self._db

    # ── Events ────────────────────────────────────────────────────────

    async def add_events(
        self,
        events: list[Event],
        topic_slug: str,
        *,
        case_id: int | None = None,
    ) -> list[int]:
        """Insert events and their sources. Returns list of new event row IDs.

        All inserts are wrapped in a transaction — if any insert fails,
        the entire batch is rolled back (all-or-nothing).
        """
        ids = []
        await self.db.execute("BEGIN")
        try:
            for event in events:
                raw_entities = event.raw_entities or event.entities
                cursor = await self.db.execute(
                    "INSERT INTO events (date, summary, significance, relation_to_prior, raw_entities, topic_slug, case_id) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        event.date.isoformat(),
                        event.summary,
                        event.significance,
                        event.relation_to_prior,
                        json.dumps(raw_entities),
                        topic_slug,
                        case_id,
                    ),
                )
                event_id = cursor.lastrowid
                ids.append(event_id)
                event.event_id = event_id

                # Insert sources
                for src in event.sources:
                    await self.db.execute(
                        "INSERT INTO event_sources (event_id, url, outlet, affiliation, country, language, framing) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (
                            event_id,
                            src.get("url", ""),
                            src.get("outlet", ""),
                            src.get("affiliation", ""),
                            src.get("country", ""),
                            src.get("language", "en"),
                            src.get("framing", ""),
                        ),
                    )

                # Backward-compatible path for tests/manual store use:
                # when raw_entities is absent, treat event.entities as canonical enough to link.
                if not event.raw_entities:
                    for entity_name in event.entities:
                        await self.db.execute(
                            "INSERT INTO entities (canonical_name, entity_type, first_seen, last_seen) "
                            "VALUES (?, 'unknown', ?, ?) "
                            "ON CONFLICT(canonical_name) DO UPDATE SET last_seen = excluded.last_seen",
                            (entity_name, event.date.isoformat(), event.date.isoformat()),
                        )
                        cursor2 = await self.db.execute(
                            "SELECT id FROM entities WHERE canonical_name = ?",
                            (entity_name,),
                        )
                        row = await cursor2.fetchone()
                        if row:
                            await self.db.execute(
                                "INSERT OR IGNORE INTO event_entities (event_id, entity_id) VALUES (?, ?)",
                                (event_id, row[0]),
                            )

            await self.db.execute("COMMIT")
        except Exception:
            await self.db.execute("ROLLBACK")
            raise
        return ids

    async def get_events(
        self,
        topic_slug: str | None = None,
        since: date | None = None,
        until: date | None = None,
        limit: int | None = None,
        *,
        case_id: int | None = None,
    ) -> list[Event]:
        """Load events for a topic, optionally filtered by date range."""
        query = (
            "SELECT id, date, summary, significance, relation_to_prior, raw_entities "
            "FROM events WHERE 1=1"
        )
        params: list = []

        if topic_slug is not None:
            query += " AND topic_slug = ?"
            params.append(topic_slug)
        elif case_id is None:
            query += " AND case_id IS NULL"
        if case_id is not None:
            query += " AND case_id = ?"
            params.append(case_id)

        if since:
            query += " AND date >= ?"
            params.append(since.isoformat())
        if until:
            query += " AND date <= ?"
            params.append(until.isoformat())

        query += " ORDER BY date ASC"
        if limit:
            query += " LIMIT ?"
            params.append(limit)

        cursor = await self.db.execute(query, params)
        rows = await cursor.fetchall()

        events = []
        for row in rows:
            event_id = row[0]

            # Fetch sources
            src_cursor = await self.db.execute(
                "SELECT url, outlet, affiliation, country, language, framing "
                "FROM event_sources WHERE event_id = ?",
                (event_id,),
            )
            sources = [
                {
                    "url": s[0],
                    "outlet": s[1],
                    "affiliation": s[2],
                    "country": s[3],
                    "language": s[4],
                    "framing": s[5],
                }
                for s in await src_cursor.fetchall()
            ]

            # Fetch entities
            ent_cursor = await self.db.execute(
                "SELECT e.canonical_name FROM entities e "
                "JOIN event_entities ee ON e.id = ee.entity_id "
                "WHERE ee.event_id = ?",
                (event_id,),
            )
            entities = [e[0] for e in await ent_cursor.fetchall()]
            raw_entities = json.loads(row[5]) if row[5] else []

            events.append(Event(
                event_id=event_id,
                date=date.fromisoformat(row[1]),
                summary=row[2],
                significance=row[3],
                relation_to_prior=row[4],
                sources=sources,
                entities=entities or raw_entities,
                raw_entities=raw_entities,
            ))

        return events

    async def get_recent_events(
        self, topic_slug: str | None = None, days: int = 7, limit: int = 30,
        reference_date: date | None = None,
        *,
        case_id: int | None = None,
    ) -> list[Event]:
        """Get events from the last N days."""
        ref = reference_date or date.today()
        since = ref - timedelta(days=days)
        events = await self.get_events(topic_slug, since=since, until=ref, case_id=case_id)
        return events[-limit:] if len(events) > limit else events

    async def get_all_events(
        self,
        topic_slug: str | None = None,
        *,
        case_id: int | None = None,
    ) -> list[Event]:
        """Get all events, optionally filtered by topic."""
        if case_id is not None:
            return await self.get_events(None, case_id=case_id)
        if topic_slug:
            return await self.get_events(topic_slug)
        # Cross-topic: get all
        cursor = await self.db.execute(
            "SELECT DISTINCT topic_slug FROM events WHERE case_id IS NULL ORDER BY topic_slug"
        )
        slugs = [row[0] for row in await cursor.fetchall()]
        all_events = []
        for slug in slugs:
            all_events.extend(await self.get_events(slug))
        return sorted(all_events, key=lambda e: e.date)

    # ── Entities ──────────────────────────────────────────────────────

    async def upsert_entity(
        self,
        canonical_name: str,
        entity_type: str = "unknown",
        aliases: list[str] | None = None,
        observation_date: date | None = None,
    ) -> int:
        """Insert or update an entity. Returns the entity ID."""
        aliases_json = json.dumps(aliases or [])
        ref_date = (observation_date or date.today()).isoformat()

        await self.db.execute(
            "INSERT INTO entities (canonical_name, entity_type, aliases, first_seen, last_seen) "
            "VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(canonical_name) DO UPDATE SET "
            "entity_type = CASE WHEN excluded.entity_type != 'unknown' THEN excluded.entity_type ELSE entities.entity_type END, "
            "aliases = excluded.aliases, "
            "first_seen = MIN(entities.first_seen, excluded.first_seen), "
            "last_seen = MAX(entities.last_seen, excluded.last_seen)",
            (canonical_name, entity_type, aliases_json, ref_date, ref_date),
        )
        await self.db.commit()

        cursor = await self.db.execute(
            "SELECT id FROM entities WHERE canonical_name = ?",
            (canonical_name,),
        )
        row = await cursor.fetchone()
        return row[0]

    async def find_entity(self, name: str) -> dict | None:
        """Look up entity by canonical name or alias."""
        # Try canonical name first
        cursor = await self.db.execute(
            "SELECT id, canonical_name, entity_type, aliases, first_seen, last_seen, thumbnail_url, wikipedia_url "
            "FROM entities WHERE canonical_name = ?",
            (name,),
        )
        row = await cursor.fetchone()
        if row:
            return _entity_row_to_dict(row)

        # Search aliases (JSON array contains check)
        cursor = await self.db.execute(
            "SELECT id, canonical_name, entity_type, aliases, first_seen, last_seen, thumbnail_url, wikipedia_url "
            "FROM entities WHERE aliases LIKE ?",
            (f'%"{name}"%',),
        )
        row = await cursor.fetchone()
        if row:
            return _entity_row_to_dict(row)

        return None

    async def get_all_entities(
        self,
        topic_slug: str | None = None,
        *,
        case_id: int | None = None,
    ) -> list[dict]:
        """Get all entities, optionally scoped to a topic."""
        if case_id is not None:
            cursor = await self.db.execute(
                "SELECT DISTINCT e.id, e.canonical_name, e.entity_type, e.aliases, "
                "e.first_seen, e.last_seen, e.thumbnail_url, e.wikipedia_url "
                "FROM entities e "
                "JOIN event_entities ee ON e.id = ee.entity_id "
                "JOIN events ev ON ee.event_id = ev.id "
                "WHERE ev.case_id = ? "
                "ORDER BY e.canonical_name",
                (case_id,),
            )
        elif topic_slug:
            cursor = await self.db.execute(
                "SELECT DISTINCT e.id, e.canonical_name, e.entity_type, e.aliases, "
                "e.first_seen, e.last_seen, e.thumbnail_url, e.wikipedia_url "
                "FROM entities e "
                "JOIN event_entities ee ON e.id = ee.entity_id "
                "JOIN events ev ON ee.event_id = ev.id "
                "WHERE ev.topic_slug = ? "
                "ORDER BY e.canonical_name",
                (topic_slug,),
            )
        else:
            cursor = await self.db.execute(
                "SELECT id, canonical_name, entity_type, aliases, first_seen, last_seen, thumbnail_url, wikipedia_url "
                "FROM entities ORDER BY canonical_name"
            )
        rows = await cursor.fetchall()
        return [_entity_row_to_dict(r) for r in rows]

    async def link_event_entities(self, event_id: int, entity_ids: list[int]) -> None:
        """Create event-entity associations."""
        for eid in entity_ids:
            await self.db.execute(
                "INSERT OR IGNORE INTO event_entities (event_id, entity_id) VALUES (?, ?)",
                (event_id, eid),
            )
        await self.db.commit()

    async def get_events_for_entity(self, entity_id: int) -> list[Event]:
        """Get all events involving a specific entity, across all topics."""
        cursor = await self.db.execute(
            "SELECT ev.id, ev.date, ev.summary, ev.significance, ev.relation_to_prior, ev.raw_entities, ev.topic_slug "
            "FROM events ev "
            "JOIN event_entities ee ON ev.id = ee.event_id "
            "WHERE ee.entity_id = ? "
            "ORDER BY ev.date ASC",
            (entity_id,),
        )
        rows = await cursor.fetchall()

        events = []
        for row in rows:
            event_id = row[0]
            src_cursor = await self.db.execute(
                "SELECT url, outlet, affiliation, country, language, framing "
                "FROM event_sources WHERE event_id = ?",
                (event_id,),
            )
            sources = [
                {"url": s[0], "outlet": s[1], "affiliation": s[2], "country": s[3], "language": s[4], "framing": s[5]}
                for s in await src_cursor.fetchall()
            ]
            ent_cursor = await self.db.execute(
                "SELECT e.canonical_name FROM entities e "
                "JOIN event_entities ee ON e.id = ee.entity_id "
                "WHERE ee.event_id = ?",
                (event_id,),
            )
            entities = [e[0] for e in await ent_cursor.fetchall()]

            events.append(Event(
                event_id=event_id,
                date=date.fromisoformat(row[1]),
                summary=row[2],
                significance=row[3],
                relation_to_prior=row[4],
                sources=sources,
                entities=entities or (json.loads(row[5]) if row[5] else []),
                raw_entities=json.loads(row[5]) if row[5] else [],
            ))
        return events

    # ── Summaries ─────────────────────────────────────────────────────

    async def add_summary(
        self, summary: Summary, topic_slug: str, period_type: str,
    ) -> int:
        """Insert a period summary. Returns the row ID."""
        cursor = await self.db.execute(
            "INSERT INTO summaries (topic_slug, period_type, period_start, period_end, text, event_count) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                topic_slug,
                period_type,
                summary.period_start.isoformat(),
                summary.period_end.isoformat(),
                summary.text,
                summary.event_count,
            ),
        )
        await self.db.commit()
        return cursor.lastrowid

    async def get_summaries(self, topic_slug: str, period_type: str) -> list[Summary]:
        """Load summaries for a topic and period type."""
        cursor = await self.db.execute(
            "SELECT period_start, period_end, text, event_count "
            "FROM summaries WHERE topic_slug = ? AND period_type = ? "
            "ORDER BY period_start ASC",
            (topic_slug, period_type),
        )
        rows = await cursor.fetchall()
        return [
            Summary(
                period_start=date.fromisoformat(r[0]),
                period_end=date.fromisoformat(r[1]),
                text=r[2],
                event_count=r[3],
            )
            for r in rows
        ]

    # ── Threads ───────────────────────────────────────────────────────

    async def upsert_thread(
        self,
        slug: str,
        headline: str,
        significance: int = 5,
        status: str = "emerging",
    ) -> int:
        """Insert or update a thread. Returns thread ID."""
        now = datetime.now().isoformat()
        await self.db.execute(
            "INSERT INTO threads (slug, headline, significance, status, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(slug) DO UPDATE SET "
            "headline = excluded.headline, "
            "significance = excluded.significance, "
            "status = excluded.status, "
            "updated_at = CASE "
            "  WHEN threads.headline != excluded.headline "
            "    OR threads.significance != excluded.significance "
            "    OR threads.status != excluded.status "
            "  THEN excluded.updated_at "
            "  ELSE threads.updated_at "
            "END",
            (slug, headline, significance, status, now, now),
        )
        await self.db.commit()
        cursor = await self.db.execute(
            "SELECT id FROM threads WHERE slug = ?", (slug,)
        )
        row = await cursor.fetchone()
        return row[0]

    async def get_active_threads(
        self,
        topic_slug: str | None = None,
        *,
        case_id: int | None = None,
    ) -> list[dict]:
        """Get threads with status 'emerging' or 'active'."""
        if case_id is not None:
            cursor = await self.db.execute(
                "SELECT DISTINCT t.id, t.slug, t.headline, t.status, t.significance, "
                "t.created_at, t.updated_at "
                "FROM threads t "
                "JOIN thread_cases tc ON t.id = tc.thread_id "
                "WHERE tc.case_id = ? AND t.status IN ('emerging', 'active') "
                "ORDER BY t.updated_at DESC",
                (case_id,),
            )
        elif topic_slug:
            cursor = await self.db.execute(
                "SELECT DISTINCT t.id, t.slug, t.headline, t.status, t.significance, "
                "t.created_at, t.updated_at "
                "FROM threads t "
                "JOIN thread_topics tt ON t.id = tt.thread_id "
                "WHERE tt.topic_slug = ? AND t.status IN ('emerging', 'active') "
                "ORDER BY t.updated_at DESC",
                (topic_slug,),
            )
        else:
            cursor = await self.db.execute(
                "SELECT id, slug, headline, status, significance, created_at, updated_at "
                "FROM threads WHERE status IN ('emerging', 'active') "
                "AND id NOT IN (SELECT thread_id FROM thread_cases) "
                "ORDER BY updated_at DESC"
            )
        rows = await cursor.fetchall()

        threads = []
        for r in rows:
            thread_id = r[0]
            # Get key entities for this thread
            ent_cursor = await self.db.execute(
                "SELECT DISTINCT e.canonical_name FROM entities e "
                "JOIN event_entities ee ON e.id = ee.entity_id "
                "JOIN thread_events te ON ee.event_id = te.event_id "
                "WHERE te.thread_id = ?",
                (thread_id,),
            )
            key_entities = [e[0] for e in await ent_cursor.fetchall()]
            count_cursor = await self.db.execute(
                "SELECT COUNT(*) FROM thread_events WHERE thread_id = ?",
                (thread_id,),
            )
            event_count = (await count_cursor.fetchone())[0]

            threads.append({
                "id": thread_id,
                "slug": r[1],
                "headline": r[2],
                "status": r[3],
                "significance": r[4],
                "created_at": r[5],
                "updated_at": r[6],
                "key_entities": key_entities,
                "event_count": event_count,
            })
        enriched = []
        for thread in threads:
            enriched.append(await self._attach_latest_snapshot_fields(thread))
        return enriched

    async def get_matchable_threads(
        self,
        topic_slug: str | None = None,
        *,
        case_id: int | None = None,
    ) -> list[dict]:
        """Get non-terminal threads eligible for event matching."""
        threads = await self.get_all_threads(topic_slug=topic_slug, case_id=case_id)
        return [thread for thread in threads if thread["status"] in ("emerging", "active", "stale")]

    async def link_thread_events(self, thread_id: int, event_ids: list[int]) -> int:
        """Link events to a thread. Returns number of new links inserted."""
        inserted = 0
        for eid in event_ids:
            cursor = await self.db.execute(
                "INSERT OR IGNORE INTO thread_events (thread_id, event_id) VALUES (?, ?)",
                (thread_id, eid),
            )
            inserted += cursor.rowcount or 0
        if inserted:
            await self.db.execute(
                "UPDATE threads SET updated_at = ? WHERE id = ?",
                (datetime.now().isoformat(), thread_id),
            )
        await self.db.commit()
        return inserted

    async def find_event_id(
        self,
        summary: str,
        date: str,
        topic_slug: str | None = None,
        *,
        case_id: int | None = None,
    ) -> int | None:
        """Find an event's DB id by its natural key."""
        query = "SELECT id FROM events WHERE summary = ? AND date = ?"
        params: list = [summary, date]
        if topic_slug is not None:
            query += " AND topic_slug = ?"
            params.append(topic_slug)
        if case_id is not None:
            query += " AND case_id = ?"
            params.append(case_id)
        cursor = await self.db.execute(query, params)
        row = await cursor.fetchone()
        return row[0] if row else None

    async def get_threads_for_event(self, event_id: int) -> list[dict]:
        """Get all threads that contain a given event."""
        cursor = await self.db.execute(
            "SELECT t.id, t.slug, t.headline, t.status, t.significance "
            "FROM threads t "
            "JOIN thread_events te ON t.id = te.thread_id "
            "WHERE te.event_id = ? "
            "ORDER BY t.significance DESC",
            (event_id,),
        )
        return [
            {"id": r[0], "slug": r[1], "headline": r[2], "status": r[3], "significance": r[4]}
            for r in await cursor.fetchall()
        ]

    async def get_topics_for_thread(self, thread_id: int) -> list[str]:
        """Get topic slugs associated with a thread."""
        cursor = await self.db.execute(
            "SELECT topic_slug FROM thread_topics WHERE thread_id = ?",
            (thread_id,),
        )
        return [r[0] for r in await cursor.fetchall()]

    async def get_cases_for_thread(self, thread_id: int) -> list[str]:
        """Get case slugs associated with a thread."""
        cursor = await self.db.execute(
            "SELECT c.slug FROM cases c "
            "JOIN thread_cases tc ON c.id = tc.case_id "
            "WHERE tc.thread_id = ? "
            "ORDER BY c.slug",
            (thread_id,),
        )
        return [r[0] for r in await cursor.fetchall()]

    async def get_related_events(
        self, event_id: int, limit: int = 5,
    ) -> list[dict]:
        """Find events sharing 2+ entities with the given event."""
        cursor = await self.db.execute(
            "SELECT e.id, e.date, e.summary, e.significance, e.topic_slug, "
            "       COUNT(*) as shared "
            "FROM events e "
            "JOIN event_entities ee ON e.id = ee.event_id "
            "WHERE ee.entity_id IN ("
            "  SELECT entity_id FROM event_entities WHERE event_id = ?"
            ") AND e.id != ? "
            "GROUP BY e.id "
            "HAVING shared >= 2 "
            "ORDER BY shared DESC, e.significance DESC "
            "LIMIT ?",
            (event_id, event_id, limit),
        )
        return [
            {"id": r[0], "date": r[1], "summary": r[2], "significance": r[3],
             "topic_slug": r[4], "shared_entities": r[5]}
            for r in await cursor.fetchall()
        ]

    async def get_graph_data(
        self, min_events: int = 3, min_co: int = 2,
    ) -> dict:
        """Return nodes + links for the force-directed graph.

        Nodes: entities appearing in >= min_events events.
        Links: entity pairs co-occurring in >= min_co events.
        """
        # Get qualifying entities
        cursor = await self.db.execute(
            "SELECT ee.entity_id, COUNT(DISTINCT ee.event_id) as evt_count "
            "FROM event_entities ee "
            "GROUP BY ee.entity_id HAVING evt_count >= ?",
            (min_events,),
        )
        entity_counts = {r[0]: r[1] for r in await cursor.fetchall()}
        if not entity_counts:
            return {"nodes": [], "links": []}

        entity_ids = list(entity_counts.keys())

        # Get entity details
        placeholders = ",".join("?" * len(entity_ids))
        cursor = await self.db.execute(
            f"SELECT id, canonical_name, entity_type, thumbnail_url "
            f"FROM entities WHERE id IN ({placeholders})",
            entity_ids,
        )
        nodes = []
        for r in await cursor.fetchall():
            nodes.append({
                "id": r[0],
                "name": r[1],
                "type": r[2],
                "thumbnail_url": r[3] or "",
                "event_count": entity_counts[r[0]],
            })

        # Get co-occurrence links
        cursor = await self.db.execute(
            f"SELECT ee1.entity_id, ee2.entity_id, COUNT(*) as co "
            f"FROM event_entities ee1 "
            f"JOIN event_entities ee2 ON ee1.event_id = ee2.event_id "
            f"  AND ee1.entity_id < ee2.entity_id "
            f"WHERE ee1.entity_id IN ({placeholders}) "
            f"  AND ee2.entity_id IN ({placeholders}) "
            f"GROUP BY ee1.entity_id, ee2.entity_id "
            f"HAVING co >= ?",
            entity_ids + entity_ids + [min_co],
        )
        links = []
        for r in await cursor.fetchall():
            links.append({
                "source": r[0],
                "target": r[1],
                "weight": r[2],
            })

        return {"nodes": nodes, "links": links}

    async def get_case_graph_data(
        self,
        case_id: int,
        *,
        min_events: int = 1,
        min_co: int = 1,
    ) -> dict:
        """Return graph data scoped to one case."""
        cursor = await self.db.execute(
            "SELECT ee.entity_id, COUNT(DISTINCT ee.event_id) as evt_count "
            "FROM event_entities ee "
            "JOIN events ev ON ee.event_id = ev.id "
            "WHERE ev.case_id = ? "
            "GROUP BY ee.entity_id HAVING evt_count >= ?",
            (case_id, min_events),
        )
        entity_counts = {r[0]: r[1] for r in await cursor.fetchall()}
        if not entity_counts:
            return {"nodes": [], "links": []}

        entity_ids = list(entity_counts.keys())
        placeholders = ",".join("?" * len(entity_ids))
        cursor = await self.db.execute(
            f"SELECT id, canonical_name, entity_type, thumbnail_url "
            f"FROM entities WHERE id IN ({placeholders})",
            entity_ids,
        )
        nodes = [
            {
                "id": r[0],
                "name": r[1],
                "type": r[2],
                "thumbnail_url": r[3] or "",
                "event_count": entity_counts[r[0]],
            }
            for r in await cursor.fetchall()
        ]

        cursor = await self.db.execute(
            f"SELECT ee1.entity_id, ee2.entity_id, COUNT(*) as co "
            f"FROM event_entities ee1 "
            f"JOIN event_entities ee2 ON ee1.event_id = ee2.event_id "
            f"  AND ee1.entity_id < ee2.entity_id "
            f"JOIN events ev ON ee1.event_id = ev.id "
            f"WHERE ev.case_id = ? "
            f"  AND ee1.entity_id IN ({placeholders}) "
            f"  AND ee2.entity_id IN ({placeholders}) "
            f"GROUP BY ee1.entity_id, ee2.entity_id "
            f"HAVING co >= ?",
            [case_id, *entity_ids, *entity_ids, min_co],
        )
        links = [{"source": r[0], "target": r[1], "weight": r[2]} for r in await cursor.fetchall()]
        return {"nodes": nodes, "links": links}

    async def merge_threads(self, keep_id: int, absorb_id: int) -> dict:
        """Merge absorb_id thread into keep_id. Reassign events, topics, analysis, snapshots, evidence refs."""
        touched_at = datetime.now().isoformat()
        keep_cursor = await self.db.execute(
            "SELECT significance, created_at FROM threads WHERE id = ?",
            (keep_id,),
        )
        keep_row = await keep_cursor.fetchone()
        absorb_cursor = await self.db.execute(
            "SELECT significance, created_at FROM threads WHERE id = ?",
            (absorb_id,),
        )
        absorb_row = await absorb_cursor.fetchone()
        if not keep_row or not absorb_row:
            return {"items_updated": 0, "questions_updated": 0}

        # Reassign thread_events (ignore duplicates)
        await self.db.execute(
            "UPDATE OR IGNORE thread_events SET thread_id = ? WHERE thread_id = ?",
            (keep_id, absorb_id),
        )
        await self.db.execute(
            "DELETE FROM thread_events WHERE thread_id = ?", (absorb_id,),
        )
        # Reassign thread_topics
        await self.db.execute(
            "UPDATE OR IGNORE thread_topics SET thread_id = ? WHERE thread_id = ?",
            (keep_id, absorb_id),
        )
        await self.db.execute(
            "DELETE FROM thread_topics WHERE thread_id = ?", (absorb_id,),
        )
        # Reassign thread_cases
        await self.db.execute(
            "UPDATE OR IGNORE thread_cases SET thread_id = ? WHERE thread_id = ?",
            (keep_id, absorb_id),
        )
        await self.db.execute(
            "DELETE FROM thread_cases WHERE thread_id = ?", (absorb_id,),
        )
        # Move convergence + divergence
        await self.db.execute(
            "UPDATE convergence SET thread_id = ? WHERE thread_id = ?",
            (keep_id, absorb_id),
        )
        await self.db.execute(
            "UPDATE divergence SET thread_id = ? WHERE thread_id = ?",
            (keep_id, absorb_id),
        )
        # Move thread_snapshots (OR IGNORE handles date conflicts)
        await self.db.execute(
            "UPDATE OR IGNORE thread_snapshots SET thread_id = ? WHERE thread_id = ?",
            (keep_id, absorb_id),
        )
        await self.db.execute(
            "DELETE FROM thread_snapshots WHERE thread_id = ?", (absorb_id,),
        )
        # Rewrite evidence_thread_ids_json in projection_items
        items_updated = await self._rewrite_thread_id_json(
            "projection_items", "evidence_thread_ids_json", keep_id, absorb_id,
        )
        # Rewrite evidence_thread_ids_json in forecast_questions
        questions_updated = await self._rewrite_thread_id_json(
            "forecast_questions", "evidence_thread_ids_json", keep_id, absorb_id,
        )
        await self.db.execute(
            "UPDATE threads SET significance = ?, created_at = ?, updated_at = ? WHERE id = ?",
            (
                max(keep_row[0], absorb_row[0]),
                min(keep_row[1], absorb_row[1]),
                touched_at,
                keep_id,
            ),
        )
        # Mark absorbed thread as merged and record merge target
        await self.db.execute(
            "UPDATE threads SET status = 'merged', merged_into_id = ?, updated_at = ? WHERE id = ?",
            (keep_id, touched_at, absorb_id),
        )
        await self.db.commit()
        return {"items_updated": items_updated, "questions_updated": questions_updated}

    async def _rewrite_thread_id_json(
        self, table: str, column: str, keep_id: int, absorb_id: int,
    ) -> int:
        """Replace absorb_id with keep_id in a JSON array column. Returns rows updated."""
        cursor = await self.db.execute(
            f"SELECT id, {column} FROM {table} WHERE {column} LIKE ?",  # noqa: S608
            (f"%{absorb_id}%",),
        )
        updated = 0
        for row in await cursor.fetchall():
            try:
                ids = json.loads(row[1])
            except (json.JSONDecodeError, TypeError):
                logger.warning("Corrupt JSON in %s.%s row %s, skipping", table, column, row[0])
                continue
            if absorb_id in ids:
                ids = list(dict.fromkeys(keep_id if x == absorb_id else x for x in ids))
                await self.db.execute(
                    f"UPDATE {table} SET {column} = ? WHERE id = ?",  # noqa: S608
                    (json.dumps(ids), row[0]),
                )
                updated += 1
        return updated

    async def link_thread_topic(self, thread_id: int, topic_slug: str) -> None:
        """Link a thread to a topic."""
        await self.db.execute(
            "INSERT OR IGNORE INTO thread_topics (thread_id, topic_slug) VALUES (?, ?)",
            (thread_id, topic_slug),
        )
        await self.db.commit()

    async def link_thread_case(
        self,
        thread_id: int,
        case_id: int,
        *,
        relevance: float = 0.5,
        role: str = "",
    ) -> None:
        """Link a thread to a case."""
        await self.db.execute(
            "INSERT OR REPLACE INTO thread_cases (thread_id, case_id, relevance, role) VALUES (?, ?, ?, ?)",
            (thread_id, case_id, relevance, role),
        )
        await self.db.commit()

    async def mark_stale_threads(self, stale_after_days: int = 14, reference_date: date | None = None) -> int:
        """Mark active/emerging threads as stale if latest linked event > stale_after_days old."""
        ref = reference_date or date.today()
        cutoff = (ref - timedelta(days=stale_after_days)).isoformat()
        cursor = await self.db.execute(
            "UPDATE threads SET status = 'stale', updated_at = ? "
            "WHERE status IN ('emerging', 'active') "
            "AND id NOT IN ("
            "  SELECT te.thread_id FROM thread_events te "
            "  JOIN events ev ON te.event_id = ev.id "
            "  WHERE ev.date >= ?"
            ")",
            (datetime.now().isoformat(), cutoff),
        )
        await self.db.commit()
        return cursor.rowcount

    async def add_convergence(
        self, thread_id: int, fact_text: str, confirmed_by: list[str],
    ) -> int:
        """Add a convergence record to a thread."""
        cursor = await self.db.execute(
            "INSERT INTO convergence (thread_id, fact_text, confirmed_by) VALUES (?, ?, ?)",
            (thread_id, fact_text, json.dumps(confirmed_by)),
        )
        await self.db.commit()
        return cursor.lastrowid

    async def add_divergence(
        self,
        thread_id: int,
        shared_event: str,
        source_a: str,
        framing_a: str,
        source_b: str,
        framing_b: str,
    ) -> int:
        """Add a divergence record to a thread."""
        cursor = await self.db.execute(
            "INSERT INTO divergence (thread_id, shared_event, source_a, framing_a, source_b, framing_b) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (thread_id, shared_event, source_a, framing_a, source_b, framing_b),
        )
        await self.db.commit()
        return cursor.lastrowid

    async def clear_thread_analysis(self, thread_id: int) -> None:
        """Clear stored convergence/divergence records for a thread before replacement."""
        await self.db.execute("DELETE FROM convergence WHERE thread_id = ?", (thread_id,))
        await self.db.execute("DELETE FROM divergence WHERE thread_id = ?", (thread_id,))
        await self.db.commit()

    # ── Syntheses ─────────────────────────────────────────────────────

    async def save_synthesis(
        self, synthesis_data: dict, topic_slug: str, run_date: date,
    ) -> int:
        """Save a TopicSynthesis snapshot as JSON."""
        cursor = await self.db.execute(
            "INSERT INTO syntheses (topic_slug, date, data_json) VALUES (?, ?, ?)",
            (topic_slug, run_date.isoformat(), json.dumps(synthesis_data, default=str)),
        )
        await self.db.commit()
        return cursor.lastrowid

    async def replace_synthesis(
        self, synthesis_data: dict, topic_slug: str, run_date: date,
    ) -> int:
        """Replace all synthesis rows for a topic/date with one refreshed snapshot."""
        await self.db.execute(
            "DELETE FROM syntheses WHERE topic_slug = ? AND date = ?",
            (topic_slug, run_date.isoformat()),
        )
        await self.db.commit()
        return await self.save_synthesis(synthesis_data, topic_slug, run_date)

    async def get_synthesis(self, topic_slug: str, run_date: date) -> dict | None:
        """Load a TopicSynthesis snapshot by topic and date."""
        cursor = await self.db.execute(
            "SELECT data_json FROM syntheses WHERE topic_slug = ? AND date = ? "
            "ORDER BY created_at DESC LIMIT 1",
            (topic_slug, run_date.isoformat()),
        )
        row = await cursor.fetchone()
        if row:
            return json.loads(row[0])
        return None

    async def get_previous_synthesis(
        self, topic_slug: str, before_date: date,
    ) -> dict | None:
        """Load the most recent TopicSynthesis snapshot before the given date."""
        cursor = await self.db.execute(
            "SELECT data_json FROM syntheses WHERE topic_slug = ? AND date < ? "
            "ORDER BY date DESC LIMIT 1",
            (topic_slug, before_date.isoformat()),
        )
        row = await cursor.fetchone()
        if row:
            return json.loads(row[0])
        return None

    async def get_synthesis_dates(self, topic_slug: str) -> list[str]:
        """Get all dates with synthesis snapshots for a topic, most recent first."""
        cursor = await self.db.execute(
            "SELECT DISTINCT date FROM syntheses WHERE topic_slug = ? "
            "ORDER BY date DESC",
            (topic_slug,),
        )
        return [r[0] for r in await cursor.fetchall()]

    async def get_all_synthesis_dates(self) -> list[str]:
        """Get all dates with any synthesis snapshot, most recent first."""
        cursor = await self.db.execute(
            "SELECT DISTINCT date FROM syntheses ORDER BY date DESC",
        )
        return [r[0] for r in await cursor.fetchall()]

    # ── Projection substrate ────────────────────────────────────────

    async def upsert_thread_snapshot(self, snapshot: ThreadSnapshot) -> int:
        """Insert or update a thread snapshot and keep metrics synchronized."""
        history = await self.get_thread_snapshots(snapshot.thread_id)
        computed = compute_snapshot_metrics(history, snapshot)
        await self.db.execute(
            "INSERT INTO thread_snapshots (thread_id, snapshot_date, status, significance, event_count, "
            "latest_event_date, velocity_7d, acceleration_7d, significance_trend_7d, momentum_score, trajectory_label) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(thread_id, snapshot_date) DO UPDATE SET "
            "status = excluded.status, significance = excluded.significance, event_count = excluded.event_count, "
            "latest_event_date = excluded.latest_event_date, velocity_7d = excluded.velocity_7d, "
            "acceleration_7d = excluded.acceleration_7d, significance_trend_7d = excluded.significance_trend_7d, "
            "momentum_score = excluded.momentum_score, trajectory_label = excluded.trajectory_label",
            (
                computed.thread_id,
                computed.snapshot_date.isoformat(),
                computed.status,
                computed.significance,
                computed.event_count,
                computed.latest_event_date.isoformat() if computed.latest_event_date else None,
                computed.velocity_7d,
                computed.acceleration_7d,
                computed.significance_trend_7d,
                computed.momentum_score,
                computed.trajectory_label,
            ),
        )
        await self.db.commit()
        cursor = await self.db.execute(
            "SELECT id FROM thread_snapshots WHERE thread_id = ? AND snapshot_date = ?",
            (snapshot.thread_id, snapshot.snapshot_date.isoformat()),
        )
        row = await cursor.fetchone()
        return row[0]

    async def get_thread_snapshots(self, thread_id: int, *, until: date | None = None) -> list[ThreadSnapshot]:
        """Load all snapshots for a thread ordered by date."""
        query = (
            "SELECT thread_id, snapshot_date, status, significance, event_count, latest_event_date, "
            "velocity_7d, acceleration_7d, significance_trend_7d, momentum_score, trajectory_label "
            "FROM thread_snapshots WHERE thread_id = ?"
        )
        params: list = [thread_id]
        if until:
            query += " AND snapshot_date <= ?"
            params.append(until.isoformat())
        query += " ORDER BY snapshot_date ASC"
        cursor = await self.db.execute(query, params)
        rows = await cursor.fetchall()
        snapshots = []
        for row in rows:
            snapshots.append(ThreadSnapshot(
                thread_id=row[0],
                snapshot_date=date.fromisoformat(row[1]),
                status=row[2],
                significance=row[3],
                event_count=row[4],
                latest_event_date=date.fromisoformat(row[5]) if row[5] else None,
                velocity_7d=row[6],
                acceleration_7d=row[7],
                significance_trend_7d=row[8],
                momentum_score=row[9],
                trajectory_label=row[10],
            ))
        return snapshots

    async def get_latest_thread_snapshot(self, thread_id: int) -> ThreadSnapshot | None:
        """Load the newest snapshot for a thread."""
        snapshots = await self.get_thread_snapshots(thread_id)
        return snapshots[-1] if snapshots else None

    async def get_thread_snapshot_as_of(self, thread_id: int, cutoff: date) -> ThreadSnapshot | None:
        """Load the newest snapshot for a thread on or before the cutoff."""
        snapshots = await self.get_thread_snapshots(thread_id, until=cutoff)
        return snapshots[-1] if snapshots else None

    async def _attach_snapshot_fields(
        self,
        payload: dict,
        *,
        as_of: date | None = None,
        override_canonical: bool = True,
    ) -> dict:
        """Attach trajectory metrics to a thread-like dict."""
        snapshot = await (
            self.get_thread_snapshot_as_of(payload["id"], as_of)
            if as_of else self.get_latest_thread_snapshot(payload["id"])
        )
        if not snapshot:
            payload["trajectory_label"] = None
            payload["momentum_score"] = None
            payload["velocity_7d"] = None
            payload["acceleration_7d"] = None
            payload["significance_trend_7d"] = None
            payload["snapshot_count"] = 0
            return payload

        payload["trajectory_label"] = snapshot.trajectory_label
        payload["momentum_score"] = snapshot.momentum_score
        payload["velocity_7d"] = snapshot.velocity_7d
        payload["acceleration_7d"] = snapshot.acceleration_7d
        payload["significance_trend_7d"] = snapshot.significance_trend_7d
        payload["snapshot_count"] = len(await self.get_thread_snapshots(payload["id"], until=as_of))
        if override_canonical:
            payload["status"] = snapshot.status
            payload["significance"] = snapshot.significance
        return payload

    async def _attach_latest_snapshot_fields(self, payload: dict) -> dict:
        """Attach latest trajectory metrics to a thread-like dict."""
        return await self._attach_snapshot_fields(payload, override_canonical=False)

    async def add_causal_link(self, causal_link: CausalLink) -> int:
        """Persist a structured causal link."""
        await self.db.execute(
            "INSERT OR REPLACE INTO causal_links (source_event_id, target_event_id, relation_type, evidence_text, strength) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                causal_link.source_event_id,
                causal_link.target_event_id,
                causal_link.relation_type,
                causal_link.evidence_text,
                causal_link.strength,
            ),
        )
        await self.db.commit()
        cursor = await self.db.execute(
            "SELECT id FROM causal_links WHERE source_event_id = ? AND target_event_id = ? AND relation_type = ?",
            (
                causal_link.source_event_id,
                causal_link.target_event_id,
                causal_link.relation_type,
            ),
        )
        row = await cursor.fetchone()
        return row[0]

    async def replace_thread_causal_links(self, thread_id: int, causal_links: list[CausalLink]) -> None:
        """Replace causal links for all events attached to a thread."""
        await self.db.execute(
            "DELETE FROM causal_links WHERE source_event_id IN "
            "(SELECT event_id FROM thread_events WHERE thread_id = ?) "
            "AND target_event_id IN (SELECT event_id FROM thread_events WHERE thread_id = ?)",
            (thread_id, thread_id),
        )
        for link in causal_links:
            await self.db.execute(
                "INSERT OR IGNORE INTO causal_links (source_event_id, target_event_id, relation_type, evidence_text, strength) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    link.source_event_id,
                    link.target_event_id,
                    link.relation_type,
                    link.evidence_text,
                    link.strength,
                ),
            )
        await self.db.commit()

    async def get_causal_links_for_thread(self, thread_id: int) -> list[dict]:
        """Load causal links that stay within one thread."""
        cursor = await self.db.execute(
            "SELECT id, source_event_id, target_event_id, relation_type, evidence_text, strength "
            "FROM causal_links WHERE source_event_id IN "
            "(SELECT event_id FROM thread_events WHERE thread_id = ?) "
            "AND target_event_id IN (SELECT event_id FROM thread_events WHERE thread_id = ?) "
            "ORDER BY source_event_id, target_event_id",
            (thread_id, thread_id),
        )
        return [
            {
                "id": row[0],
                "source_event_id": row[1],
                "target_event_id": row[2],
                "relation_type": row[3],
                "evidence_text": row[4],
                "strength": row[5],
            }
            for row in await cursor.fetchall()
        ]

    async def get_causal_links_for_events(self, event_ids: list[int]) -> list[dict]:
        """Load causal links touching a set of event IDs."""
        if not event_ids:
            return []
        placeholders = ", ".join("?" for _ in event_ids)
        cursor = await self.db.execute(
            "SELECT id, source_event_id, target_event_id, relation_type, evidence_text, strength "
            f"FROM causal_links WHERE source_event_id IN ({placeholders}) OR target_event_id IN ({placeholders}) "
            "ORDER BY source_event_id, target_event_id",
            event_ids + event_ids,
        )
        return [
            {
                "id": row[0],
                "source_event_id": row[1],
                "target_event_id": row[2],
                "relation_type": row[3],
                "evidence_text": row[4],
                "strength": row[5],
            }
            for row in await cursor.fetchall()
        ]

    async def replace_cross_topic_signals(self, signals: list[CrossTopicSignal], observed_for: date) -> None:
        """Replace cross-topic signals for a specific observed date."""
        await self.db.execute(
            "DELETE FROM cross_topic_signals WHERE observed_at = ?",
            (observed_for.isoformat(),),
        )
        for signal in signals:
            await self.db.execute(
                "INSERT OR IGNORE INTO cross_topic_signals "
                "(topic_slug, related_topic_slug, shared_entity, signal_type, observed_at, event_ids_json, related_event_ids_json, note) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    signal.topic_slug,
                    signal.related_topic_slug,
                    signal.shared_entity,
                    signal.signal_type,
                    signal.observed_at.isoformat(),
                    json.dumps(signal.event_ids),
                    json.dumps(signal.related_event_ids),
                    signal.note,
                ),
            )
        await self.db.commit()

    async def get_cross_topic_signals(
        self,
        topic_slug: str,
        *,
        since: date | None = None,
        limit: int = 10,
    ) -> list[CrossTopicSignal]:
        """Load recent cross-topic signals for a topic."""
        query = (
            "SELECT id, topic_slug, related_topic_slug, shared_entity, signal_type, observed_at, "
            "event_ids_json, related_event_ids_json, note "
            "FROM cross_topic_signals WHERE topic_slug = ?"
        )
        params: list = [topic_slug]
        if since:
            query += " AND observed_at >= ?"
            params.append(since.isoformat())
        query += " ORDER BY observed_at DESC LIMIT ?"
        params.append(limit)
        cursor = await self.db.execute(query, params)
        rows = await cursor.fetchall()
        return [
            CrossTopicSignal(
                signal_id=row[0],
                topic_slug=row[1],
                related_topic_slug=row[2],
                shared_entity=row[3],
                signal_type=row[4],
                observed_at=date.fromisoformat(row[5]),
                event_ids=json.loads(row[6]) if row[6] else [],
                related_event_ids=json.loads(row[7]) if row[7] else [],
                note=row[8],
            )
            for row in rows
        ]

    async def get_cross_topic_signals_as_of(
        self,
        topic_slug: str,
        cutoff: date,
        *,
        limit: int = 10,
        lookback_days: int = 14,
    ) -> list[CrossTopicSignal]:
        """Derive cross-topic signals using only events available on or before the cutoff."""
        rows = await self.get_recent_entity_activity(reference_date=cutoff, lookback_days=lookback_days)
        signals = detect_cross_topic_signals(rows, lookback_days=lookback_days)
        return [signal for signal in signals if signal.topic_slug == topic_slug][:limit]

    async def get_recent_entity_activity(
        self,
        *,
        reference_date: date | None = None,
        lookback_days: int = 14,
    ) -> list[dict]:
        """Return canonical entity mentions across topics for bridge detection."""
        ref = reference_date or date.today()
        since = (ref - timedelta(days=lookback_days)).isoformat()
        cursor = await self.db.execute(
            "SELECT ev.id, ev.topic_slug, ev.date, e.canonical_name "
            "FROM events ev "
            "JOIN event_entities ee ON ev.id = ee.event_id "
            "JOIN entities e ON ee.entity_id = e.id "
            "WHERE ev.date >= ? AND ev.date <= ? "
            "ORDER BY ev.date ASC",
            (since, ref.isoformat()),
        )
        rows = await cursor.fetchall()
        return [
            {
                "event_id": row[0],
                "topic_slug": row[1],
                "event_date": date.fromisoformat(row[2]),
                "canonical_name": row[3],
            }
            for row in rows
        ]

    async def detect_and_save_cross_topic_signals(
        self,
        *,
        reference_date: date | None = None,
        lookback_days: int = 14,
    ) -> list[CrossTopicSignal]:
        """Compute and persist cross-topic bridge signals."""
        ref = reference_date or date.today()
        rows = await self.get_recent_entity_activity(reference_date=ref, lookback_days=lookback_days)
        signals = detect_cross_topic_signals(rows, lookback_days=lookback_days)
        await self.replace_cross_topic_signals(signals, ref)
        return signals

    async def save_projection(self, projection: TopicProjection) -> int:
        """Persist a topic projection and its items."""
        cursor = await self.db.execute(
            "INSERT INTO projections (topic_slug, topic_name, engine, generated_for, status, summary, metadata_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                projection.topic_slug,
                projection.topic_name,
                projection.engine,
                projection.generated_for.isoformat(),
                projection.status,
                projection.summary,
                json.dumps(projection.metadata),
            ),
        )
        projection_id = cursor.lastrowid
        for item in projection.items:
            item_cursor = await self.db.execute(
                "INSERT INTO projection_items "
                "(projection_id, claim, confidence, horizon_days, signpost, signals_cited_json, "
                "evidence_event_ids_json, evidence_thread_ids_json, review_after, external_ref) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    projection_id,
                    item.claim,
                    item.confidence,
                    item.horizon_days,
                    item.signpost,
                    json.dumps(item.signals_cited),
                    json.dumps(item.evidence_event_ids),
                    json.dumps(item.evidence_thread_ids),
                    item.review_after.isoformat(),
                    item.external_ref,
                ),
            )
            projection_item_id = item_cursor.lastrowid
            await self.db.execute(
                "INSERT INTO projection_outcomes (projection_item_id, outcome_status, external_ref) VALUES (?, 'pending', ?)",
                (projection_item_id, item.external_ref),
            )
        await self.db.commit()
        return projection_id

    async def save_forecast_run(self, forecast_run: ForecastRun) -> int:
        """Persist a structured forecast run and initialize pending resolutions."""
        cursor = await self.db.execute(
            "INSERT INTO forecast_runs (topic_slug, topic_name, engine, generated_for, summary, metadata_json) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                forecast_run.topic_slug,
                forecast_run.topic_name,
                forecast_run.engine,
                forecast_run.generated_for.isoformat(),
                forecast_run.summary,
                json.dumps(forecast_run.metadata),
            ),
        )
        run_id = cursor.lastrowid
        forecast_run.run_id = run_id

        for scenario in forecast_run.scenarios:
            scenario_cursor = await self.db.execute(
                "INSERT INTO forecast_scenarios (forecast_run_id, scenario_key, label, probability, description, signposts_json, status) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    scenario.scenario_key,
                    scenario.label,
                    scenario.probability,
                    scenario.description,
                    json.dumps(scenario.signposts),
                    scenario.status,
                ),
            )
            scenario.scenario_id = scenario_cursor.lastrowid

        for question in forecast_run.questions:
            if not question.forecast_key:
                question.forecast_key = build_forecast_key(
                    topic_slug=forecast_run.topic_slug,
                    generated_for=forecast_run.generated_for,
                    target_variable=question.target_variable,
                    target_metadata=question.target_metadata,
                    resolution_date=question.resolution_date,
                    horizon_days=question.horizon_days,
                    expected_direction=question.expected_direction,
                )
            question_cursor = await self.db.execute(
                "INSERT INTO forecast_questions "
                "(forecast_run_id, forecast_key, question, forecast_type, target_variable, target_metadata_json, probability, base_rate, "
                "resolution_criteria, resolution_date, horizon_days, signpost, expected_direction, signals_cited_json, "
                "evidence_event_ids_json, evidence_thread_ids_json, cross_topic_signal_ids_json, status, external_ref, reasoning_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    question.forecast_key,
                    question.question,
                    question.forecast_type,
                    question.target_variable,
                    json.dumps(question.target_metadata),
                    question.probability,
                    question.base_rate,
                    question.resolution_criteria,
                    question.resolution_date.isoformat(),
                    question.horizon_days,
                    question.signpost,
                    question.expected_direction,
                    json.dumps(question.signals_cited),
                    json.dumps(question.evidence_event_ids),
                    json.dumps(question.evidence_thread_ids),
                    json.dumps(question.cross_topic_signal_ids),
                    question.status,
                    question.external_ref,
                    json.dumps(question.reasoning),
                ),
            )
            question_id = question_cursor.lastrowid
            question.question_id = question_id
            await self.db.execute(
                "INSERT INTO forecast_resolutions (forecast_question_id, outcome_status, external_ref) VALUES (?, 'pending', ?)",
                (question_id, question.external_ref),
            )
            if question.external_ref:
                await self.db.execute(
                    "INSERT INTO forecast_mappings (forecast_question_id, forecast_key, mapping_type, external_ref, metadata_json) "
                    "VALUES (?, ?, 'external_ref', ?, '{}')",
                    (question_id, question.forecast_key or "", question.external_ref),
                )

        await self.db.commit()
        return run_id

    async def get_latest_projection(self, topic_slug: str, engine: str | None = None) -> TopicProjection | None:
        """Load the latest stored projection for a topic."""
        query = (
            "SELECT id, topic_slug, topic_name, engine, generated_for, status, summary, metadata_json "
            "FROM projections WHERE topic_slug = ?"
        )
        params: list = [topic_slug]
        if engine:
            query += " AND engine = ?"
            params.append(engine)
        query += " ORDER BY generated_for DESC, id DESC LIMIT 1"
        cursor = await self.db.execute(query, params)
        row = await cursor.fetchone()
        if not row:
            return None

        items_cursor = await self.db.execute(
            "SELECT id, claim, confidence, horizon_days, signpost, signals_cited_json, "
            "evidence_event_ids_json, evidence_thread_ids_json, review_after, external_ref "
            "FROM projection_items WHERE projection_id = ? ORDER BY id ASC",
            (row[0],),
        )
        items = [
            ProjectionItem(
                claim=item_row[1],
                confidence=item_row[2],
                horizon_days=item_row[3],
                signpost=item_row[4],
                signals_cited=json.loads(item_row[5]) if item_row[5] else [],
                evidence_event_ids=json.loads(item_row[6]) if item_row[6] else [],
                evidence_thread_ids=json.loads(item_row[7]) if item_row[7] else [],
                review_after=date.fromisoformat(item_row[8]),
                external_ref=item_row[9],
            )
            for item_row in await items_cursor.fetchall()
        ]

        return TopicProjection(
            topic_slug=row[1],
            topic_name=row[2],
            engine=row[3],
            generated_for=date.fromisoformat(row[4]),
            status=row[5],
            summary=row[6],
            items=items,
            cross_topic_signals=await self.get_cross_topic_signals(topic_slug, limit=5),
            metadata=json.loads(row[7]) if row[7] else {},
        )

    async def get_latest_forecast_run(self, topic_slug: str, engine: str | None = None) -> ForecastRun | None:
        """Load the latest structured forecast run for a topic."""
        query = (
            "SELECT id, topic_slug, topic_name, engine, generated_for, summary, metadata_json "
            "FROM forecast_runs WHERE topic_slug = ?"
        )
        params: list = [topic_slug]
        if engine:
            query += " AND engine = ?"
            params.append(engine)
        query += " ORDER BY generated_for DESC, id DESC LIMIT 1"
        cursor = await self.db.execute(query, params)
        row = await cursor.fetchone()
        if not row:
            return None

        questions_cursor = await self.db.execute(
            "SELECT id, forecast_key, question, forecast_type, target_variable, target_metadata_json, probability, base_rate, "
            "resolution_criteria, resolution_date, horizon_days, signpost, expected_direction, signals_cited_json, "
            "evidence_event_ids_json, evidence_thread_ids_json, cross_topic_signal_ids_json, status, external_ref "
            "FROM forecast_questions WHERE forecast_run_id = ? ORDER BY id ASC",
            (row[0],),
        )
        questions = [
            ForecastQuestion(
                question_id=question_row[0],
                forecast_key=question_row[1],
                question=question_row[2],
                forecast_type=question_row[3],
                target_variable=question_row[4],
                target_metadata=json.loads(question_row[5]) if question_row[5] else {},
                probability=question_row[6],
                base_rate=question_row[7],
                resolution_criteria=question_row[8],
                resolution_date=date.fromisoformat(question_row[9]),
                horizon_days=question_row[10],
                signpost=question_row[11],
                expected_direction=question_row[12],
                signals_cited=json.loads(question_row[13]) if question_row[13] else [],
                evidence_event_ids=json.loads(question_row[14]) if question_row[14] else [],
                evidence_thread_ids=json.loads(question_row[15]) if question_row[15] else [],
                cross_topic_signal_ids=json.loads(question_row[16]) if question_row[16] else [],
                status=question_row[17],
                external_ref=question_row[18],
            )
            for question_row in await questions_cursor.fetchall()
        ]
        for question in questions:
            if not question.forecast_key:
                question.forecast_key = build_forecast_key(
                    topic_slug=row[1],
                    generated_for=date.fromisoformat(row[4]),
                    target_variable=question.target_variable,
                    target_metadata=question.target_metadata,
                    resolution_date=question.resolution_date,
                    horizon_days=question.horizon_days,
                    expected_direction=question.expected_direction,
                )

        scenarios_cursor = await self.db.execute(
            "SELECT id, scenario_key, label, probability, description, signposts_json, status "
            "FROM forecast_scenarios WHERE forecast_run_id = ? ORDER BY id ASC",
            (row[0],),
        )
        scenarios = [
            ForecastScenario(
                scenario_id=scenario_row[0],
                scenario_key=scenario_row[1],
                label=scenario_row[2],
                probability=scenario_row[3],
                description=scenario_row[4],
                signposts=json.loads(scenario_row[5]) if scenario_row[5] else [],
                status=scenario_row[6],
            )
            for scenario_row in await scenarios_cursor.fetchall()
        ]

        return ForecastRun(
            run_id=row[0],
            topic_slug=row[1],
            topic_name=row[2],
            engine=row[3],
            generated_for=date.fromisoformat(row[4]),
            summary=row[5],
            questions=questions,
            scenarios=scenarios,
            metadata=json.loads(row[6]) if row[6] else {},
        )

    async def get_projection_items_for_thread(self, thread_id: int) -> list[dict]:
        """Return stored projection items that cite a thread as evidence."""
        cursor = await self.db.execute(
            "SELECT pi.id, p.topic_slug, p.engine, p.generated_for, pi.claim, pi.confidence, "
            "pi.horizon_days, pi.signpost, pi.review_after, pi.evidence_thread_ids_json "
            "FROM projection_items pi "
            "JOIN projections p ON pi.projection_id = p.id "
            "ORDER BY p.generated_for DESC, pi.id DESC",
        )
        matches = []
        for row in await cursor.fetchall():
            evidence_thread_ids = json.loads(row[9]) if row[9] else []
            if thread_id not in evidence_thread_ids:
                continue
            matches.append({
                "id": row[0],
                "topic_slug": row[1],
                "engine": row[2],
                "generated_for": row[3],
                "claim": row[4],
                "confidence": row[5],
                "horizon_days": row[6],
                "signpost": row[7],
                "review_after": row[8],
            })
        return matches

    async def get_pending_projection_items(
        self,
        *,
        until: date | None = None,
        engine: str | None = None,
    ) -> list[dict]:
        """Projection items due for automated evaluation."""
        ref = until or date.today()
        query = (
            "SELECT po.id, pi.id, p.topic_slug, p.engine, p.generated_for, pi.claim, pi.signpost, "
            "pi.review_after, pi.confidence, pi.horizon_days "
            "FROM projection_outcomes po "
            "JOIN projection_items pi ON po.projection_item_id = pi.id "
            "JOIN projections p ON pi.projection_id = p.id "
            "WHERE po.outcome_status = 'pending' AND pi.review_after <= ?"
        )
        params: list = [ref.isoformat()]
        if engine:
            query += " AND p.engine = ?"
            params.append(engine)
        cursor = await self.db.execute(query, params)
        rows = await cursor.fetchall()
        return [
            {
                "outcome_id": row[0],
                "projection_item_id": row[1],
                "topic_slug": row[2],
                "engine": row[3],
                "generated_for": row[4],
                "claim": row[5],
                "signpost": row[6],
                "review_after": row[7],
                "confidence": row[8],
                "horizon_days": row[9],
            }
            for row in rows
        ]

    async def get_pending_forecast_questions(
        self,
        *,
        until: date | None = None,
        engine: str | None = None,
    ) -> list[dict]:
        """Forecast questions due for resolution."""
        ref = until or date.today()
        query = (
            "SELECT fr.id, fq.id, fr.topic_slug, fr.engine, fr.generated_for, fq.forecast_key, fq.question, fq.forecast_type, "
            "fq.target_variable, fq.target_metadata_json, fq.probability, fq.base_rate, fq.resolution_criteria, "
            "fq.resolution_date, fq.horizon_days, fq.expected_direction "
            "FROM forecast_resolutions fres "
            "JOIN forecast_questions fq ON fres.forecast_question_id = fq.id "
            "JOIN forecast_runs fr ON fq.forecast_run_id = fr.id "
            "WHERE fres.outcome_status = 'pending' AND fq.resolution_date <= ?"
        )
        params: list = [ref.isoformat()]
        if engine:
            query += " AND fr.engine = ?"
            params.append(engine)
        cursor = await self.db.execute(query, params)
        rows = await cursor.fetchall()
        return [
            {
                "forecast_run_id": row[0],
                "forecast_question_id": row[1],
                "topic_slug": row[2],
                "engine": row[3],
                "generated_for": row[4],
                "forecast_key": row[5] or build_forecast_key(
                    topic_slug=row[2],
                    generated_for=date.fromisoformat(row[4]),
                    target_variable=row[8],
                    target_metadata=json.loads(row[9]) if row[9] else {},
                    resolution_date=date.fromisoformat(row[13]),
                    horizon_days=row[14],
                    expected_direction=row[15],
                ),
                "question": row[6],
                "forecast_type": row[7],
                "target_variable": row[8],
                "target_metadata": json.loads(row[9]) if row[9] else {},
                "probability": row[10],
                "base_rate": row[11],
                "resolution_criteria": row[12],
                "resolution_date": row[13],
                "horizon_days": row[14],
                "expected_direction": row[15],
            }
            for row in rows
        ]

    async def get_open_forecasts(
        self,
        topic_slug: str | None = None,
    ) -> list[dict]:
        """Return open (unresolved) forecast questions with run metadata."""
        query = (
            "SELECT fq.id, fr.topic_slug, fr.engine, fr.generated_for, "
            "fq.question, fq.probability, fq.base_rate, fq.resolution_date, "
            "fq.target_variable, fq.external_ref, fq.updated_at, fq.horizon_days "
            "FROM forecast_questions fq "
            "JOIN forecast_runs fr ON fq.forecast_run_id = fr.id "
            "JOIN forecast_resolutions fres ON fres.forecast_question_id = fq.id "
            "WHERE fres.outcome_status = 'pending'"
        )
        params: list = []
        if topic_slug:
            query += " AND fr.topic_slug = ?"
            params.append(topic_slug)
        query += " ORDER BY fq.id ASC"
        cursor = await self.db.execute(query, params)
        rows = await cursor.fetchall()
        return [
            {
                "forecast_question_id": row[0],
                "topic_slug": row[1],
                "engine": row[2],
                "generated_for": row[3],
                "question": row[4],
                "probability": row[5],
                "base_rate": row[6],
                "resolution_date": row[7],
                "target_variable": row[8],
                "external_ref": row[9],
                "updated_at": row[10],
                "horizon_days": row[11],
            }
            for row in rows
        ]

    async def update_forecast_probability(
        self,
        question_id: int,
        new_probability: float,
        source: str = "reprice",
        *,
        market_probability: float | None = None,
    ) -> None:
        """Update a forecast question's probability and record the change."""
        await self.db.execute(
            "UPDATE forecast_questions SET probability = ?, updated_at = datetime('now') WHERE id = ?",
            (new_probability, question_id),
        )
        await self.db.execute(
            "INSERT INTO forecast_probability_history "
            "(forecast_question_id, probability, market_probability, source) "
            "VALUES (?, ?, ?, ?)",
            (question_id, new_probability, market_probability, source),
        )
        await self.db.commit()

    async def get_forecast_probability_history(
        self, question_id: int,
    ) -> list[dict]:
        """Return probability change history for a forecast question."""
        cursor = await self.db.execute(
            "SELECT probability, market_probability, source, recorded_at "
            "FROM forecast_probability_history "
            "WHERE forecast_question_id = ? ORDER BY recorded_at ASC",
            (question_id,),
        )
        rows = await cursor.fetchall()
        return [
            {
                "probability": row[0],
                "market_probability": row[1],
                "source": row[2],
                "recorded_at": row[3],
            }
            for row in rows
        ]

    async def get_forecast_questions_between(
        self,
        *,
        start: date,
        end: date,
        engine: str | None = None,
    ) -> list[dict]:
        """Return stored forecast questions generated within a date window."""
        query = (
            "SELECT fr.topic_slug, fr.topic_name, fr.engine, fr.generated_for, fq.id, fq.forecast_key, fq.question, "
            "fq.forecast_type, fq.target_variable, fq.target_metadata_json, fq.probability, fq.base_rate, "
            "fq.resolution_criteria, fq.resolution_date, fq.horizon_days, fq.expected_direction, "
            "fres.outcome_status, fres.resolved_bool, fres.brier_score, fres.log_loss, "
            "fq.reasoning_json, fq.external_ref "
            "FROM forecast_questions fq "
            "JOIN forecast_runs fr ON fq.forecast_run_id = fr.id "
            "LEFT JOIN forecast_resolutions fres ON fres.forecast_question_id = fq.id "
            "WHERE fr.generated_for >= ? AND fr.generated_for <= ?"
        )
        params: list = [start.isoformat(), end.isoformat()]
        if engine:
            query += " AND fr.engine = ?"
            params.append(engine)
        query += " ORDER BY fr.generated_for ASC, fq.id ASC"
        cursor = await self.db.execute(query, params)
        rows = await cursor.fetchall()
        return [
            {
                "topic_slug": row[0],
                "topic_name": row[1],
                "engine": row[2],
                "generated_for": row[3],
                "forecast_question_id": row[4],
                "forecast_key": row[5] or build_forecast_key(
                    topic_slug=row[0],
                    generated_for=date.fromisoformat(row[3]),
                    target_variable=row[8],
                    target_metadata=json.loads(row[9]) if row[9] else {},
                    resolution_date=date.fromisoformat(row[13]),
                    horizon_days=row[14],
                    expected_direction=row[15],
                ),
                "question": row[6],
                "forecast_type": row[7],
                "target_variable": row[8],
                "target_metadata": json.loads(row[9]) if row[9] else {},
                "probability": row[10],
                "base_rate": row[11],
                "resolution_criteria": row[12],
                "resolution_date": row[13],
                "horizon_days": row[14],
                "expected_direction": row[15],
                "outcome_status": row[16],
                "resolved_bool": None if row[17] is None else bool(row[17]),
                "brier_score": row[18],
                "log_loss": row[19],
                "reasoning": json.loads(row[20]) if row[20] else {},
                "external_ref": row[21],
            }
            for row in rows
        ]

    async def get_prediction_history(
        self,
        identifier: str,
        *,
        lookback_days: int = 30,
        engine: str | None = None,
        run_label: str | None = None,
    ) -> list[dict]:
        """Return probability trajectory for a question over time.

        Uses external_ref (Kalshi ticker) as identifier.
        Optionally filter by engine and run_label to avoid mixing different
        engine/run combinations (which produces fake "trends").
        Returns list of {generated_for, probability, market_prob} ordered by date.
        """
        cutoff = (date.today() - timedelta(days=lookback_days)).isoformat()
        conditions = ["fq.external_ref = ?", "fr.generated_for >= ?"]
        params: list = [identifier, cutoff]
        if engine:
            conditions.append("fr.engine = ?")
            params.append(engine)
        if run_label:
            conditions.append(
                "json_extract(fq.target_metadata_json, '$.run_label') = ?"
            )
            params.append(run_label)
        where = " AND ".join(conditions)
        cursor = await self.db.execute(
            "SELECT fr.generated_for, fq.probability, "
            "json_extract(fq.target_metadata_json, '$.kalshi_implied') as market_prob "
            "FROM forecast_questions fq "
            "JOIN forecast_runs fr ON fq.forecast_run_id = fr.id "
            f"WHERE {where} "
            "ORDER BY fr.generated_for ASC",
            params,
        )
        rows = await cursor.fetchall()
        return [
            {
                "generated_for": row[0],
                "probability": row[1],
                "market_prob": row[2],
            }
            for row in rows
        ]

    async def get_featured_predictions(
        self,
        topic_slug: str,
        *,
        limit: int = 5,
        engine: str | None = None,
        allowed_target_variables: set[str] | None = None,
    ) -> list[dict]:
        """Return pending predictions for a topic, ordered by recency.

        Returns raw forecast question dicts (caller should enrich and score).
        """
        today = date.today()
        start = today - timedelta(days=90)
        query = (
            "SELECT fr.topic_slug, fr.topic_name, fr.engine, fr.generated_for, "
            "fq.id, fq.forecast_key, fq.question, fq.forecast_type, fq.target_variable, "
            "fq.target_metadata_json, fq.probability, fq.base_rate, "
            "fq.resolution_criteria, fq.resolution_date, fq.horizon_days, fq.expected_direction, "
            "fres.outcome_status, fres.resolved_bool, fres.brier_score, fres.log_loss, "
            "fq.reasoning_json, fq.external_ref "
            "FROM forecast_questions fq "
            "JOIN forecast_runs fr ON fq.forecast_run_id = fr.id "
            "LEFT JOIN forecast_resolutions fres ON fres.forecast_question_id = fq.id "
            "WHERE fr.topic_slug = ? "
            "AND fr.generated_for >= ? "
            "AND (fres.outcome_status IS NULL OR fres.outcome_status = 'pending') "
        )
        params: list = [topic_slug, start.isoformat()]
        if engine:
            query += "AND fr.engine = ? "
            params.append(engine)
        if allowed_target_variables:
            variables = sorted(allowed_target_variables)
            placeholders = ",".join("?" * len(variables))
            query += f"AND fq.target_variable IN ({placeholders}) "
            params.extend(variables)
        query += "ORDER BY fr.generated_for DESC, fq.id DESC LIMIT ?"
        params.append(limit * 3)  # fetch extra for dedup
        cursor = await self.db.execute(query, params)
        rows = await cursor.fetchall()
        seen_questions: set[str] = set()
        results: list[dict] = []
        for row in rows:
            question_text = row[6]
            if question_text in seen_questions:
                continue
            seen_questions.add(question_text)
            results.append({
                "topic_slug": row[0],
                "topic_name": row[1],
                "engine": row[2],
                "generated_for": row[3],
                "forecast_question_id": row[4],
                "forecast_key": row[5],
                "question": question_text,
                "forecast_type": row[7],
                "target_variable": row[8],
                "target_metadata": json.loads(row[9]) if row[9] else {},
                "probability": row[10],
                "base_rate": row[11],
                "resolution_criteria": row[12],
                "resolution_date": row[13],
                "horizon_days": row[14],
                "expected_direction": row[15],
                "outcome_status": row[16],
                "resolved_bool": None if row[17] is None else bool(row[17]),
                "brier_score": row[18],
                "log_loss": row[19],
                "reasoning": json.loads(row[20]) if row[20] else {},
                "external_ref": row[21],
            })
            if len(results) >= limit:
                break
        return results

    async def backfill_forecast_keys(
        self,
        *,
        start: date | None = None,
        end: date | None = None,
        engine: str | None = None,
    ) -> dict:
        """Persist stable forecast keys for existing rows that predate the key contract."""
        query = (
            "SELECT fq.id, fr.topic_slug, fr.generated_for, fq.target_variable, fq.target_metadata_json, "
            "fq.resolution_date, fq.horizon_days, fq.expected_direction "
            "FROM forecast_questions fq "
            "JOIN forecast_runs fr ON fq.forecast_run_id = fr.id "
            "WHERE (fq.forecast_key = '' OR fq.forecast_key IS NULL)"
        )
        params: list = []
        if start is not None:
            query += " AND fr.generated_for >= ?"
            params.append(start.isoformat())
        if end is not None:
            query += " AND fr.generated_for <= ?"
            params.append(end.isoformat())
        if engine:
            query += " AND fr.engine = ?"
            params.append(engine)
        query += " ORDER BY fr.generated_for ASC, fq.id ASC"

        cursor = await self.db.execute(query, params)
        rows = await cursor.fetchall()
        questions_backfilled = 0
        mappings_backfilled = 0
        for row in rows:
            forecast_key = build_forecast_key(
                topic_slug=row[1],
                generated_for=date.fromisoformat(row[2]),
                target_variable=row[3],
                target_metadata=json.loads(row[4]) if row[4] else {},
                resolution_date=date.fromisoformat(row[5]),
                horizon_days=row[6],
                expected_direction=row[7],
            )
            await self.db.execute(
                "UPDATE forecast_questions SET forecast_key = ? WHERE id = ?",
                (forecast_key, row[0]),
            )
            mapping_cursor = await self.db.execute(
                "UPDATE forecast_mappings SET forecast_key = ? "
                "WHERE forecast_question_id = ? AND (forecast_key = '' OR forecast_key IS NULL)",
                (forecast_key, row[0]),
            )
            questions_backfilled += 1
            mappings_backfilled += max(0, mapping_cursor.rowcount or 0)

        if rows:
            await self.db.commit()

        return {
            "questions_scanned": len(rows),
            "questions_backfilled": questions_backfilled,
            "mappings_backfilled": mappings_backfilled,
            "start": start.isoformat() if start else None,
            "end": end.isoformat() if end else None,
            "engine": engine,
        }

    async def set_projection_outcome(self, outcome: ProjectionOutcome) -> None:
        """Update a stored projection outcome."""
        await self.db.execute(
            "UPDATE projection_outcomes SET outcome_status = ?, score = ?, notes = ?, reviewed_at = ?, external_ref = ? "
            "WHERE projection_item_id = ?",
            (
                outcome.outcome_status,
                outcome.score,
                outcome.notes,
                outcome.reviewed_at.isoformat() if outcome.reviewed_at else None,
                outcome.external_ref,
                outcome.projection_item_id,
            ),
        )
        await self.db.commit()

    async def set_forecast_resolution(self, resolution: ForecastResolution) -> None:
        """Persist the resolved outcome for a forecast question."""
        await self.db.execute(
            "UPDATE forecast_resolutions SET outcome_status = ?, resolved_bool = ?, realized_direction = ?, "
            "actual_value = ?, brier_score = ?, log_loss = ?, notes = ?, resolved_at = ?, external_ref = ? "
            "WHERE forecast_question_id = ?",
            (
                resolution.outcome_status,
                None if resolution.resolved_bool is None else int(resolution.resolved_bool),
                resolution.realized_direction,
                resolution.actual_value,
                resolution.brier_score,
                resolution.log_loss,
                resolution.notes,
                resolution.resolved_at.isoformat() if resolution.resolved_at else None,
                resolution.external_ref,
                resolution.forecast_question_id,
            ),
        )
        await self.db.commit()

    async def get_historical_calibration(self, *, as_of: date | None = None) -> list[dict]:
        """Return resolved forecast questions with outcomes for calibration feedback.

        Joins forecast_questions → forecast_resolutions → forecast_runs.
        Returns rows with target_variable, probability, resolved_bool, base_rate.
        If as_of is provided, only includes runs generated on or before that date.
        """
        query = (
            "SELECT fq.target_variable, fq.probability, fq.base_rate, "
            "fres.resolved_bool, fr.generated_for "
            "FROM forecast_questions fq "
            "JOIN forecast_runs fr ON fq.forecast_run_id = fr.id "
            "JOIN forecast_resolutions fres ON fres.forecast_question_id = fq.id "
            "WHERE fres.outcome_status = 'resolved' AND fres.resolved_bool IS NOT NULL"
        )
        params: list = []
        if as_of is not None:
            query += " AND fr.generated_for <= ?"
            params.append(as_of.isoformat())
        query += " ORDER BY fr.generated_for ASC"
        cursor = await self.db.execute(query, params)
        rows = await cursor.fetchall()
        return [
            {
                "target_variable": row[0],
                "probability": row[1],
                "base_rate": row[2],
                "resolved_bool": bool(row[3]),
                "generated_for": row[4],
            }
            for row in rows
        ]

    async def get_interesting_kalshi_markets(
        self,
        limit: int = 5,
        *,
        engine: str | None = None,
        topic_slug: str | None = None,
    ) -> list[dict]:
        """Return open Kalshi-aligned markets ranked by interestingness.

        Score = 0.6 * |our_prob - market_prob| + 0.4 * (1 / days_until_resolution).
        """
        today = date.today().isoformat()
        query = (
            "SELECT fq.question, fq.probability, fq.target_metadata_json, "
            "fq.resolution_date, fq.external_ref "
            "FROM forecast_questions fq "
            "JOIN forecast_runs fr ON fq.forecast_run_id = fr.id "
            "WHERE fq.target_variable = 'kalshi_aligned' "
            "AND fq.status = 'open' "
            "AND fq.resolution_date >= ? "
        )
        params: list = [today]
        if engine:
            query += "AND fr.engine = ? "
            params.append(engine)
        if topic_slug:
            query += "AND fr.topic_slug = ? "
            params.append(topic_slug)
        query += "ORDER BY fq.resolution_date ASC"
        cursor = await self.db.execute(query, params)
        rows = await cursor.fetchall()

        results = []
        for row in rows:
            meta = json.loads(row[2]) if row[2] else {}
            market_prob = meta.get("kalshi_implied", row[1])
            if market_prob is None:
                continue
            resolution_date = row[3]
            days_until = max(
                1,
                (date.fromisoformat(resolution_date) - date.today()).days,
            )
            gap = abs(row[1] - market_prob)
            score = 0.6 * gap + 0.4 * (1.0 / days_until)
            gap_pp = round(gap * 100)
            ticker = meta.get("kalshi_ticker", row[4] or "")
            results.append({
                "question": row[0],
                "probability": row[1],
                "market_prob": market_prob,
                "resolution_date": resolution_date,
                "days_until_resolution": days_until,
                "gap_pp": gap_pp,
                "ticker": ticker,
                "score": score,
            })

        # Deduplicate by ticker — keep highest-scoring entry per market
        seen: dict[str, dict] = {}
        for r in results:
            key = r["ticker"] or r["question"][:60]
            if key not in seen or r["score"] > seen[key]["score"]:
                seen[key] = r
        deduped = sorted(seen.values(), key=lambda r: r["score"], reverse=True)
        return deduped[:limit]

    async def purge_template_projections(self, dry_run: bool = True) -> dict:
        """Delete projection items containing template-mush questions.

        Identifies items matching patterns like:
        - "Will % be involved in significant % developments%"
        - "Will % produce significant new developments%"
        """
        cursor = await self.db.execute(
            "SELECT id, projection_id FROM projection_items "
            "WHERE claim LIKE 'Will % be involved in significant%developments%' "
            "OR claim LIKE 'Will % produce significant new developments%' "
            "OR claim LIKE 'Will a new event be recorded in the%thread by%' "
            "OR claim LIKE 'Will the total number of%events recorded%' "
            "OR claim LIKE 'Will the volume of%reports increase%'"
        )
        items = await cursor.fetchall()
        item_ids = [row[0] for row in items]
        projection_ids = list({row[1] for row in items})

        result = {
            "items_found": len(item_ids),
            "items_deleted": 0,
            "projections_deleted": 0,
            "dry_run": dry_run,
        }

        if dry_run or not item_ids:
            return result

        # Delete outcomes for matched items
        placeholders = ",".join("?" * len(item_ids))
        await self.db.execute(
            f"DELETE FROM projection_outcomes WHERE projection_item_id IN ({placeholders})",
            item_ids,
        )
        # Delete the template items
        await self.db.execute(
            f"DELETE FROM projection_items WHERE id IN ({placeholders})",
            item_ids,
        )
        result["items_deleted"] = len(item_ids)

        # Delete projections left with zero items
        for proj_id in projection_ids:
            cursor = await self.db.execute(
                "SELECT COUNT(*) FROM projection_items WHERE projection_id = ?",
                (proj_id,),
            )
            count = (await cursor.fetchone())[0]
            if count == 0:
                await self.db.execute(
                    "DELETE FROM projections WHERE id = ?", (proj_id,),
                )
                result["projections_deleted"] += 1

        await self.db.commit()
        return result

    async def purge_forecast_runs(
        self,
        *,
        target_variable: str | None = None,
        topic_slug: str | None = None,
        dry_run: bool = True,
    ) -> dict:
        """Delete forecast runs and all dependent rows (resolutions, mappings, questions).

        Filter by target_variable (on forecast_questions) and/or topic_slug (on forecast_runs).
        Deletes in FK order: resolutions → mappings → questions → scenarios → runs.
        """
        # Build filter conditions
        conditions = []
        params: list = []
        if target_variable:
            conditions.append("fq.target_variable = ?")
            params.append(target_variable)
        if topic_slug:
            conditions.append("fr.topic_slug = ?")
            params.append(topic_slug)
        if not conditions:
            return {"runs_deleted": 0, "questions_deleted": 0, "resolutions_deleted": 0,
                    "mappings_deleted": 0, "dry_run": dry_run}

        where = " AND ".join(conditions)

        # Find matching run IDs
        cursor = await self.db.execute(
            f"SELECT DISTINCT fr.id FROM forecast_runs fr "
            f"JOIN forecast_questions fq ON fq.forecast_run_id = fr.id "
            f"WHERE {where}",
            params,
        )
        run_ids = [row[0] for row in await cursor.fetchall()]

        if not run_ids:
            return {"runs_deleted": 0, "questions_deleted": 0, "resolutions_deleted": 0,
                    "mappings_deleted": 0, "dry_run": dry_run,
                    "runs_found": 0, "questions_found": 0}

        # Collect question IDs
        ph_runs = ",".join("?" * len(run_ids))
        cursor = await self.db.execute(
            f"SELECT id FROM forecast_questions WHERE forecast_run_id IN ({ph_runs})",
            run_ids,
        )
        question_ids = [row[0] for row in await cursor.fetchall()]

        if dry_run:
            return {
                "runs_found": len(run_ids),
                "questions_found": len(question_ids),
                "dry_run": True,
            }

        # Delete in FK order
        ph_qs = ",".join("?" * len(question_ids)) if question_ids else "''"
        res_deleted = 0
        map_deleted = 0

        if question_ids:
            cursor = await self.db.execute(
                f"DELETE FROM forecast_resolutions WHERE forecast_question_id IN ({ph_qs})",
                question_ids,
            )
            res_deleted = cursor.rowcount
            cursor = await self.db.execute(
                f"DELETE FROM forecast_mappings WHERE forecast_question_id IN ({ph_qs})",
                question_ids,
            )
            map_deleted = cursor.rowcount

        cursor = await self.db.execute(
            f"DELETE FROM forecast_questions WHERE forecast_run_id IN ({ph_runs})",
            run_ids,
        )
        qs_deleted = cursor.rowcount

        await self.db.execute(
            f"DELETE FROM forecast_scenarios WHERE forecast_run_id IN ({ph_runs})",
            run_ids,
        )

        cursor = await self.db.execute(
            f"DELETE FROM forecast_runs WHERE id IN ({ph_runs})",
            run_ids,
        )
        runs_deleted = cursor.rowcount

        await self.db.commit()
        return {
            "runs_deleted": runs_deleted,
            "questions_deleted": qs_deleted,
            "resolutions_deleted": res_deleted,
            "mappings_deleted": map_deleted,
            "dry_run": False,
        }

    async def get_filter_log_dates(self, topic_slug: str | None = None) -> list[str]:
        """Get all dates with filter log entries, most recent first."""
        if topic_slug:
            cursor = await self.db.execute(
                "SELECT DISTINCT run_date FROM filter_log "
                "WHERE topic_slug = ? ORDER BY run_date DESC",
                (topic_slug,),
            )
        else:
            cursor = await self.db.execute(
                "SELECT DISTINCT run_date FROM filter_log ORDER BY run_date DESC",
            )
        return [r[0] for r in await cursor.fetchall()]

    async def get_filter_log_topics(self) -> list[str]:
        """Get all topic slugs that have filter log entries."""
        cursor = await self.db.execute(
            "SELECT DISTINCT topic_slug FROM filter_log ORDER BY topic_slug",
        )
        return [r[0] for r in await cursor.fetchall()]

    async def get_adjacent_signals(
        self, run_date: date, min_significance: int = 6, max_relevance: int = 4,
    ) -> list[dict]:
        """Find articles with high significance but low relevance — potential niche signals.

        These are articles from existing sources that are significant developments
        but don't match any configured topic well.
        """
        cursor = await self.db.execute(
            "SELECT url, title, source_id, source_affiliation, source_country, "
            "topic_slug, relevance_score, significance_score, significance_reason "
            "FROM filter_log "
            "WHERE run_date = ? "
            "AND significance_score IS NOT NULL "
            "AND significance_score >= ? "
            "AND relevance_score IS NOT NULL "
            "AND relevance_score <= ? "
            "ORDER BY significance_score DESC",
            (run_date.isoformat(), min_significance, max_relevance),
        )
        return [
            {
                "url": r[0], "title": r[1], "source_id": r[2],
                "source_affiliation": r[3], "source_country": r[4],
                "topic_slug": r[5], "relevance_score": r[6],
                "significance_score": r[7], "reason": r[8],
            }
            for r in await cursor.fetchall()
        ]

    # ── Pages ─────────────────────────────────────────────────────────

    async def save_page(
        self,
        slug: str,
        title: str,
        page_type: str,
        content_md: str,
        topic_slug: str | None,
        ttl_days: int,
        prompt_hash: str,
    ) -> int:
        """Save or update a cached narrative page."""
        stale_after = (datetime.now() + timedelta(days=ttl_days)).isoformat()
        await self.db.execute(
            "INSERT INTO pages (slug, title, page_type, topic_slug, content_md, "
            "generated_at, stale_after, prompt_hash) "
            "VALUES (?, ?, ?, ?, ?, datetime('now'), ?, ?) "
            "ON CONFLICT(slug) DO UPDATE SET "
            "title = excluded.title, "
            "content_md = excluded.content_md, "
            "generated_at = datetime('now'), "
            "stale_after = excluded.stale_after, "
            "prompt_hash = excluded.prompt_hash",
            (slug, title, page_type, topic_slug, content_md, stale_after, prompt_hash),
        )
        await self.db.commit()
        cursor = await self.db.execute(
            "SELECT id FROM pages WHERE slug = ?", (slug,)
        )
        row = await cursor.fetchone()
        return row[0]

    async def get_page(self, slug: str) -> dict | None:
        """Get a page by slug."""
        cursor = await self.db.execute(
            "SELECT id, slug, title, page_type, topic_slug, content_md, "
            "generated_at, stale_after, prompt_hash "
            "FROM pages WHERE slug = ?",
            (slug,),
        )
        row = await cursor.fetchone()
        if not row:
            return None
        return {
            "id": row[0],
            "slug": row[1],
            "title": row[2],
            "page_type": row[3],
            "topic_slug": row[4],
            "content_md": row[5],
            "generated_at": row[6],
            "stale_after": row[7],
            "prompt_hash": row[8],
        }

    async def get_stale_pages(self) -> list[dict]:
        """Find pages past their stale_after time."""
        now = datetime.now().isoformat()
        cursor = await self.db.execute(
            "SELECT id, slug, title, page_type, topic_slug, prompt_hash "
            "FROM pages WHERE stale_after < ?",
            (now,),
        )
        return [
            {
                "id": r[0], "slug": r[1], "title": r[2],
                "page_type": r[3], "topic_slug": r[4], "prompt_hash": r[5],
            }
            for r in await cursor.fetchall()
        ]

    # ── Filter Log ────────────────────────────────────────────────────

    async def add_filter_log(self, entries: list[dict]) -> None:
        """Bulk insert filter decision log entries."""
        for e in entries:
            await self.db.execute(
                "INSERT INTO filter_log "
                "(run_date, topic_slug, url, title, source_id, source_affiliation, "
                "source_country, relevance_score, relevance_reason, passed_pass1, "
                "significance_score, is_novel, significance_reason, passed_pass2, "
                "final_score, outcome) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    e.get("run_date", ""),
                    e.get("topic_slug", ""),
                    e.get("url", ""),
                    e.get("title", ""),
                    e.get("source_id", ""),
                    e.get("source_affiliation", ""),
                    e.get("source_country", ""),
                    e.get("relevance_score"),
                    e.get("relevance_reason", ""),
                    1 if e.get("passed_pass1") else 0,
                    e.get("significance_score"),
                    1 if e.get("is_novel") else (0 if e.get("is_novel") is not None else None),
                    e.get("significance_reason", ""),
                    1 if e.get("passed_pass2") else (0 if e.get("passed_pass2") is not None else None),
                    e.get("final_score"),
                    e.get("outcome", "rejected"),
                ),
            )
        await self.db.commit()

    async def get_filter_log(
        self, topic_slug: str, run_date: date,
    ) -> list[dict]:
        """Load filter log entries for a specific run."""
        cursor = await self.db.execute(
            "SELECT id, run_date, topic_slug, url, title, source_id, "
            "source_affiliation, source_country, relevance_score, relevance_reason, "
            "passed_pass1, significance_score, is_novel, significance_reason, "
            "passed_pass2, final_score, outcome "
            "FROM filter_log WHERE topic_slug = ? AND run_date = ? "
            "ORDER BY relevance_score DESC",
            (topic_slug, run_date.isoformat()),
        )
        rows = await cursor.fetchall()
        return [
            {
                "id": r[0], "run_date": r[1], "topic_slug": r[2],
                "url": r[3], "title": r[4], "source_id": r[5],
                "source_affiliation": r[6], "source_country": r[7],
                "relevance_score": r[8], "relevance_reason": r[9],
                "passed_pass1": bool(r[10]),
                "significance_score": r[11],
                "is_novel": bool(r[12]) if r[12] is not None else None,
                "significance_reason": r[13],
                "passed_pass2": bool(r[14]) if r[14] is not None else None,
                "final_score": r[15], "outcome": r[16],
            }
            for r in rows
        ]

    async def get_filter_stats(
        self, topic_slug: str, run_date: date,
    ) -> dict:
        """Get aggregate filter stats for a run."""
        cursor = await self.db.execute(
            "SELECT outcome, COUNT(*) FROM filter_log "
            "WHERE topic_slug = ? AND run_date = ? "
            "GROUP BY outcome",
            (topic_slug, run_date.isoformat()),
        )
        stats = {"total": 0}
        for row in await cursor.fetchall():
            stats[row[0]] = row[1]
            stats["total"] += row[1]
        return stats

    # ── Dashboard Queries ───────────────────────────────────────────

    async def get_event_by_id(self, event_id: int) -> Event | None:
        """Load a single event with full sources and entities."""
        cursor = await self.db.execute(
            "SELECT id, date, summary, significance, relation_to_prior, raw_entities "
            "FROM events WHERE id = ?",
            (event_id,),
        )
        row = await cursor.fetchone()
        if not row:
            return None
        return await self._build_event_from_row(row)

    async def get_events_for_thread(self, thread_id: int) -> list[Event]:
        """Get all events linked to a thread, with full sources+entities."""
        cursor = await self.db.execute(
            "SELECT ev.id, ev.date, ev.summary, ev.significance, ev.relation_to_prior, ev.raw_entities "
            "FROM events ev "
            "JOIN thread_events te ON ev.id = te.event_id "
            "WHERE te.thread_id = ? "
            "ORDER BY ev.date ASC",
            (thread_id,),
        )
        rows = await cursor.fetchall()
        return [await self._build_event_from_row(r) for r in rows]

    async def get_events_for_thread_as_of(self, thread_id: int, cutoff: date) -> list[Event]:
        """Get thread-linked events up to and including the cutoff date."""
        cursor = await self.db.execute(
            "SELECT ev.id, ev.date, ev.summary, ev.significance, ev.relation_to_prior, ev.raw_entities "
            "FROM events ev "
            "JOIN thread_events te ON ev.id = te.event_id "
            "WHERE te.thread_id = ? AND ev.date <= ? "
            "ORDER BY ev.date ASC",
            (thread_id, cutoff.isoformat()),
        )
        rows = await cursor.fetchall()
        return [await self._build_event_from_row(r) for r in rows]

    async def get_convergence_for_thread(self, thread_id: int) -> list[dict]:
        """Get convergence records for a thread."""
        cursor = await self.db.execute(
            "SELECT id, fact_text, confirmed_by FROM convergence WHERE thread_id = ?",
            (thread_id,),
        )
        return [
            {"id": r[0], "fact_text": r[1], "confirmed_by": json.loads(r[2])}
            for r in await cursor.fetchall()
        ]

    async def get_divergence_for_thread(self, thread_id: int) -> list[dict]:
        """Get divergence records for a thread."""
        cursor = await self.db.execute(
            "SELECT id, shared_event, source_a, framing_a, source_b, framing_b "
            "FROM divergence WHERE thread_id = ?",
            (thread_id,),
        )
        return [
            {
                "id": r[0], "shared_event": r[1],
                "source_a": r[2], "framing_a": r[3],
                "source_b": r[4], "framing_b": r[5],
            }
            for r in await cursor.fetchall()
        ]

    async def get_thread(self, slug: str) -> dict | None:
        """Get a single thread by slug (any status)."""
        cursor = await self.db.execute(
            "SELECT id, slug, headline, status, significance, created_at, updated_at "
            "FROM threads WHERE slug = ?",
            (slug,),
        )
        row = await cursor.fetchone()
        if not row:
            return None
        thread_id = row[0]
        ent_cursor = await self.db.execute(
            "SELECT DISTINCT e.canonical_name FROM entities e "
            "JOIN event_entities ee ON e.id = ee.entity_id "
            "JOIN thread_events te ON ee.event_id = te.event_id "
            "WHERE te.thread_id = ?",
            (thread_id,),
        )
        key_entities = [e[0] for e in await ent_cursor.fetchall()]
        payload = {
            "id": thread_id, "slug": row[1], "headline": row[2],
            "status": row[3], "significance": row[4],
            "created_at": row[5], "updated_at": row[6],
            "key_entities": key_entities,
        }
        return await self._attach_latest_snapshot_fields(payload)

    async def get_thread_as_of(self, slug: str, cutoff: date) -> dict | None:
        """Get a single thread by slug with snapshot state pinned to a cutoff."""
        cursor = await self.db.execute(
            "SELECT id, slug, headline, status, significance, created_at, updated_at "
            "FROM threads WHERE slug = ?",
            (slug,),
        )
        row = await cursor.fetchone()
        if not row:
            return None
        thread_id = row[0]
        ent_cursor = await self.db.execute(
            "SELECT DISTINCT e.canonical_name FROM entities e "
            "JOIN event_entities ee ON e.id = ee.entity_id "
            "JOIN thread_events te ON ee.event_id = te.event_id "
            "JOIN events ev ON te.event_id = ev.id "
            "WHERE te.thread_id = ? AND ev.date <= ?",
            (thread_id, cutoff.isoformat()),
        )
        key_entities = [e[0] for e in await ent_cursor.fetchall()]
        payload = {
            "id": thread_id, "slug": row[1], "headline": row[2],
            "status": row[3], "significance": row[4],
            "created_at": row[5], "updated_at": row[6],
            "key_entities": key_entities,
        }
        return await self._attach_snapshot_fields(payload, as_of=cutoff)

    async def get_all_threads(
        self,
        topic_slug: str | None = None,
        status: str | None = None,
        *,
        case_id: int | None = None,
    ) -> list[dict]:
        """Get threads with optional topic and status filters."""
        if case_id is not None:
            query = (
                "SELECT DISTINCT t.id, t.slug, t.headline, t.status, t.significance, "
                "t.created_at, t.updated_at "
                "FROM threads t "
                "JOIN thread_cases tc ON t.id = tc.thread_id "
                "WHERE tc.case_id = ?"
            )
            params: list = [case_id]
            if status:
                query += " AND t.status = ?"
                params.append(status)
            else:
                query += " AND t.status != 'merged'"
            query += " ORDER BY t.updated_at DESC"
        elif topic_slug:
            query = (
                "SELECT DISTINCT t.id, t.slug, t.headline, t.status, t.significance, "
                "t.created_at, t.updated_at "
                "FROM threads t "
                "JOIN thread_topics tt ON t.id = tt.thread_id "
                "WHERE tt.topic_slug = ?"
            )
            params: list = [topic_slug]
            if status:
                query += " AND t.status = ?"
                params.append(status)
            else:
                query += " AND t.status != 'merged'"
            query += " ORDER BY t.updated_at DESC"
        else:
            query = (
                "SELECT id, slug, headline, status, significance, created_at, updated_at "
                "FROM threads"
            )
            params = []
            if status:
                query += " WHERE status = ? AND id NOT IN (SELECT thread_id FROM thread_cases)"
                params.append(status)
            else:
                query += " WHERE status != 'merged' AND id NOT IN (SELECT thread_id FROM thread_cases)"
            query += " ORDER BY updated_at DESC"

        cursor = await self.db.execute(query, params)
        rows = await cursor.fetchall()

        threads = []
        for r in rows:
            thread_id = r[0]
            ent_cursor = await self.db.execute(
                "SELECT DISTINCT e.canonical_name FROM entities e "
                "JOIN event_entities ee ON e.id = ee.entity_id "
                "JOIN thread_events te ON ee.event_id = te.event_id "
                "WHERE te.thread_id = ?",
                (thread_id,),
            )
            key_entities = [e[0] for e in await ent_cursor.fetchall()]
            count_cursor = await self.db.execute(
                "SELECT COUNT(*) FROM thread_events WHERE thread_id = ?",
                (thread_id,),
            )
            event_count = (await count_cursor.fetchone())[0]
            threads.append({
                "id": thread_id, "slug": r[1], "headline": r[2],
                "status": r[3], "significance": r[4],
                "created_at": r[5], "updated_at": r[6],
                "key_entities": key_entities,
                "event_count": event_count,
            })
        enriched = []
        for thread in threads:
            enriched.append(await self._attach_latest_snapshot_fields(thread))
        return enriched

    async def get_all_threads_as_of(
        self,
        topic_slug: str | None = None,
        cutoff: date | None = None,
        status: str | None = None,
        *,
        case_id: int | None = None,
    ) -> list[dict]:
        """Get threads with snapshot state and entity context pinned to a cutoff."""
        if cutoff is None:
            return await self.get_all_threads(topic_slug=topic_slug, status=status, case_id=case_id)

        if case_id is not None:
            query = (
                "SELECT DISTINCT t.id, t.slug, t.headline, t.status, t.significance, "
                "t.created_at, t.updated_at "
                "FROM threads t "
                "JOIN thread_cases tc ON t.id = tc.thread_id "
                "WHERE tc.case_id = ?"
            )
            params: list = [case_id]
            if status:
                query += " AND t.status = ?"
                params.append(status)
            else:
                query += " AND t.status != 'merged'"
        elif topic_slug:
            query = (
                "SELECT DISTINCT t.id, t.slug, t.headline, t.status, t.significance, "
                "t.created_at, t.updated_at "
                "FROM threads t "
                "JOIN thread_topics tt ON t.id = tt.thread_id "
                "WHERE tt.topic_slug = ?"
            )
            params: list = [topic_slug]
            if status:
                query += " AND t.status = ?"
                params.append(status)
            else:
                query += " AND t.status != 'merged'"
        else:
            query = (
                "SELECT id, slug, headline, status, significance, created_at, updated_at "
                "FROM threads"
            )
            params = []
            if status:
                query += " WHERE status = ? AND id NOT IN (SELECT thread_id FROM thread_cases)"
                params.append(status)
            else:
                query += " WHERE status != 'merged' AND id NOT IN (SELECT thread_id FROM thread_cases)"

        cursor = await self.db.execute(query, params)
        rows = await cursor.fetchall()

        threads = []
        for r in rows:
            thread_id = r[0]
            snapshot = await self.get_thread_snapshot_as_of(thread_id, cutoff)
            if not snapshot and status != "merged":
                continue
            if snapshot and status and snapshot.status != status:
                continue
            ent_cursor = await self.db.execute(
                "SELECT DISTINCT e.canonical_name FROM entities e "
                "JOIN event_entities ee ON e.id = ee.entity_id "
                "JOIN thread_events te ON ee.event_id = te.event_id "
                "JOIN events ev ON te.event_id = ev.id "
                "WHERE te.thread_id = ? AND ev.date <= ?",
                (thread_id, cutoff.isoformat()),
            )
            key_entities = [e[0] for e in await ent_cursor.fetchall()]
            count_cursor = await self.db.execute(
                "SELECT COUNT(*) FROM thread_events te "
                "JOIN events ev ON te.event_id = ev.id "
                "WHERE te.thread_id = ? AND ev.date <= ?",
                (thread_id, cutoff.isoformat()),
            )
            event_count = (await count_cursor.fetchone())[0]
            payload = {
                "id": thread_id, "slug": r[1], "headline": r[2],
                "status": snapshot.status if snapshot else r[3],
                "significance": snapshot.significance if snapshot else r[4],
                "created_at": r[5], "updated_at": r[6],
                "key_entities": key_entities,
                "event_count": event_count,
            }
            threads.append(await self._attach_snapshot_fields(payload, as_of=cutoff))
        threads.sort(key=lambda thread: (thread.get("momentum_score") or 0.0, thread["headline"]), reverse=True)
        return threads

    async def get_threads_for_entity(self, entity_id: int) -> list[dict]:
        """Get threads involving a specific entity (via event linkage)."""
        cursor = await self.db.execute(
            "SELECT DISTINCT t.id, t.slug, t.headline, t.status, t.significance, "
            "t.created_at, t.updated_at "
            "FROM threads t "
            "JOIN thread_events te ON t.id = te.thread_id "
            "JOIN event_entities ee ON te.event_id = ee.event_id "
            "WHERE ee.entity_id = ? AND t.status != 'merged' "
            "ORDER BY t.updated_at DESC",
            (entity_id,),
        )
        rows = await cursor.fetchall()
        return [
            {
                "id": r[0], "slug": r[1], "headline": r[2],
                "status": r[3], "significance": r[4],
                "created_at": r[5], "updated_at": r[6],
            }
            for r in rows
        ]

    async def get_source_stats(
        self,
        topic_slug: str | None = None,
        *,
        case_id: int | None = None,
    ) -> list[dict]:
        """Aggregate event sources by outlet/affiliation/country."""
        if case_id is not None:
            cursor = await self.db.execute(
                "SELECT es.outlet, es.affiliation, es.country, es.language, COUNT(*) as cnt "
                "FROM event_sources es "
                "JOIN events ev ON es.event_id = ev.id "
                "WHERE ev.case_id = ? "
                "GROUP BY es.outlet, es.affiliation, es.country, es.language "
                "ORDER BY cnt DESC",
                (case_id,),
            )
        elif topic_slug:
            cursor = await self.db.execute(
                "SELECT es.outlet, es.affiliation, es.country, es.language, COUNT(*) as cnt "
                "FROM event_sources es "
                "JOIN events ev ON es.event_id = ev.id "
                "WHERE ev.topic_slug = ? "
                "GROUP BY es.outlet, es.affiliation, es.country, es.language "
                "ORDER BY cnt DESC",
                (topic_slug,),
            )
        else:
            cursor = await self.db.execute(
                "SELECT es.outlet, es.affiliation, es.country, es.language, COUNT(*) as cnt "
                "FROM event_sources es "
                "JOIN events ev ON es.event_id = ev.id "
                "WHERE ev.case_id IS NULL "
                "GROUP BY outlet, affiliation, country, language "
                "ORDER BY cnt DESC"
            )
        return [
            {
                "outlet": r[0], "affiliation": r[1], "country": r[2],
                "language": r[3], "event_count": r[4],
            }
            for r in await cursor.fetchall()
        ]

    # ── Cases ───────────────────────────────────────────────────────

    async def upsert_case(
        self,
        slug: str,
        title: str,
        question: str,
        *,
        status: str = "active",
        time_bounds: dict | None = None,
        build_defaults: dict | None = None,
        monitoring_enabled: bool = True,
    ) -> int:
        """Insert or update a case registry record."""
        now = datetime.now().isoformat()
        await self.db.execute(
            "INSERT INTO cases (slug, title, question, status, time_bounds_json, build_defaults_json, monitoring_enabled, generated_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(slug) DO UPDATE SET "
            "title = excluded.title, question = excluded.question, status = excluded.status, "
            "time_bounds_json = excluded.time_bounds_json, build_defaults_json = excluded.build_defaults_json, "
            "monitoring_enabled = excluded.monitoring_enabled, updated_at = excluded.updated_at",
            (
                slug,
                title,
                question,
                status,
                json.dumps(time_bounds or {}),
                json.dumps(build_defaults or {}),
                1 if monitoring_enabled else 0,
                now,
                now,
            ),
        )
        await self.db.commit()
        cursor = await self.db.execute("SELECT id FROM cases WHERE slug = ?", (slug,))
        row = await cursor.fetchone()
        return row[0]

    async def get_case(self, slug: str) -> dict | None:
        """Load a case registry row by slug."""
        cursor = await self.db.execute(
            "SELECT id, slug, title, question, status, time_bounds_json, build_defaults_json, "
            "monitoring_enabled, generated_at, updated_at "
            "FROM cases WHERE slug = ?",
            (slug,),
        )
        row = await cursor.fetchone()
        if not row:
            return None
        return {
            "id": row[0],
            "slug": row[1],
            "title": row[2],
            "question": row[3],
            "status": row[4],
            "time_bounds": json.loads(row[5] or "{}"),
            "build_defaults": json.loads(row[6] or "{}"),
            "monitoring_enabled": bool(row[7]),
            "generated_at": row[8],
            "updated_at": row[9],
        }

    async def get_all_cases(self) -> list[dict]:
        """List stored cases."""
        cursor = await self.db.execute(
            "SELECT slug FROM cases ORDER BY updated_at DESC, slug ASC"
        )
        slugs = [row[0] for row in await cursor.fetchall()]
        cases = []
        for slug in slugs:
            case = await self.get_case(slug)
            if case is not None:
                cases.append(case)
        return cases

    async def clear_case_threads(self, case_id: int) -> None:
        """Remove thread links for one case before relinking."""
        await self.db.execute("DELETE FROM thread_cases WHERE case_id = ?", (case_id,))
        await self.db.commit()

    async def reset_case_scope(self, case_id: int) -> None:
        """Clear case-linked operational state before a full rebuild."""
        thread_cursor = await self.db.execute(
            "SELECT thread_id FROM thread_cases WHERE case_id = ?",
            (case_id,),
        )
        thread_ids = [row[0] for row in await thread_cursor.fetchall()]
        event_cursor = await self.db.execute(
            "SELECT id FROM events WHERE case_id = ?",
            (case_id,),
        )
        event_ids = [row[0] for row in await event_cursor.fetchall()]

        if thread_ids:
            thread_placeholders = ",".join("?" for _ in thread_ids)
            await self.db.execute(
                f"DELETE FROM convergence WHERE thread_id IN ({thread_placeholders})",
                thread_ids,
            )
            await self.db.execute(
                f"DELETE FROM divergence WHERE thread_id IN ({thread_placeholders})",
                thread_ids,
            )
            await self.db.execute(
                f"DELETE FROM thread_snapshots WHERE thread_id IN ({thread_placeholders})",
                thread_ids,
            )
            await self.db.execute(
                f"DELETE FROM thread_events WHERE thread_id IN ({thread_placeholders})",
                thread_ids,
            )

        if event_ids:
            event_placeholders = ",".join("?" for _ in event_ids)
            await self.db.execute(
                f"DELETE FROM event_sources WHERE event_id IN ({event_placeholders})",
                event_ids,
            )
            await self.db.execute(
                f"DELETE FROM event_entities WHERE event_id IN ({event_placeholders})",
                event_ids,
            )
            await self.db.execute(
                f"DELETE FROM causal_links WHERE source_event_id IN ({event_placeholders}) "
                f"OR target_event_id IN ({event_placeholders})",
                event_ids + event_ids,
            )
            await self.db.execute(
                f"DELETE FROM entity_relationships WHERE source_event_id IN ({event_placeholders})",
                event_ids,
            )
            await self.db.execute(
                f"DELETE FROM thread_events WHERE event_id IN ({event_placeholders})",
                event_ids,
            )
            await self.db.execute(
                f"DELETE FROM events WHERE id IN ({event_placeholders})",
                event_ids,
            )

        await self.db.execute("DELETE FROM thread_cases WHERE case_id = ?", (case_id,))
        await self.db.execute("DELETE FROM case_documents WHERE case_id = ?", (case_id,))
        await self.db.execute("DELETE FROM case_evidence WHERE case_id = ?", (case_id,))
        await self.db.execute("DELETE FROM case_hypotheses WHERE case_id = ?", (case_id,))
        await self.db.execute("DELETE FROM case_assessments WHERE case_id = ?", (case_id,))
        await self.db.execute("DELETE FROM case_open_questions WHERE case_id = ?", (case_id,))

        if thread_ids:
            thread_placeholders = ",".join("?" for _ in thread_ids)
            await self.db.execute(
                f"UPDATE threads SET merged_into_id = NULL WHERE merged_into_id IN ({thread_placeholders})",
                thread_ids,
            )
            await self.db.execute(
                f"DELETE FROM threads "
                f"WHERE id IN ({thread_placeholders}) "
                f"AND id NOT IN (SELECT thread_id FROM thread_topics) "
                f"AND id NOT IN (SELECT thread_id FROM thread_cases) "
                f"AND id NOT IN (SELECT thread_id FROM thread_events)",
                thread_ids,
            )

        await self.db.commit()

    async def replace_case_documents(self, case_id: int, documents: list[dict]) -> None:
        """Replace stored case documents."""
        await self.db.execute("DELETE FROM case_documents WHERE case_id = ?", (case_id,))
        for document in documents:
            await self.db.execute(
                "INSERT INTO case_documents "
                "(case_id, document_key, title, url, canonical_url, kind, role, source_class, source_label, "
                "priority, notes, discovered_via, published_at, quality_label, summary, time_anchors_json, excerpt, "
                "ingestion_status, ingestion_error) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    case_id,
                    document["id"],
                    document["title"],
                    document["url"],
                    document["canonical_url"],
                    document["kind"],
                    document["role"],
                    document["source_class"],
                    document.get("source_label", ""),
                    document.get("priority", 5),
                    document.get("notes"),
                    document.get("discovered_via"),
                    document.get("published_at"),
                    document.get("quality_label", "medium"),
                    document.get("summary", ""),
                    json.dumps(document.get("time_anchors", [])),
                    document.get("excerpt"),
                    document.get("ingestion_status", "ok"),
                    document.get("ingestion_error"),
                ),
            )
        await self.db.commit()

    async def get_case_documents(self, case_id: int) -> list[dict]:
        """Load stored case documents."""
        cursor = await self.db.execute(
            "SELECT document_key, title, url, canonical_url, kind, role, source_class, source_label, priority, "
            "notes, discovered_via, published_at, quality_label, summary, time_anchors_json, excerpt, "
            "ingestion_status, ingestion_error "
            "FROM case_documents WHERE case_id = ? ORDER BY priority DESC, id ASC",
            (case_id,),
        )
        return [
            {
                "id": row[0],
                "title": row[1],
                "url": row[2],
                "canonical_url": row[3],
                "kind": row[4],
                "role": row[5],
                "source_class": row[6],
                "source_label": row[7],
                "priority": row[8],
                "notes": row[9],
                "discovered_via": row[10],
                "published_at": row[11],
                "quality_label": row[12],
                "summary": row[13],
                "time_anchors": json.loads(row[14] or "[]"),
                "excerpt": row[15],
                "ingestion_status": row[16],
                "ingestion_error": row[17],
            }
            for row in await cursor.fetchall()
        ]

    async def replace_case_evidence(self, case_id: int, evidence_items: list[dict]) -> None:
        """Replace stored case evidence."""
        await self.db.execute("DELETE FROM case_evidence WHERE case_id = ?", (case_id,))
        for item in evidence_items:
            await self.db.execute(
                "INSERT INTO case_evidence "
                "(case_id, evidence_key, claim, stance, quality_label, summary, document_key, document_title, "
                "document_url, source_label, source_class, related_hypotheses_json, excerpt, time_anchors_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    case_id,
                    item["id"],
                    item["claim"],
                    item["stance"],
                    item["quality_label"],
                    item["summary"],
                    item["document_id"],
                    item["document_title"],
                    item["document_url"],
                    item.get("source_label", ""),
                    item.get("source_class", "analysis"),
                    json.dumps(item.get("related_hypotheses", [])),
                    item.get("excerpt"),
                    json.dumps(item.get("time_anchors", [])),
                ),
            )
        await self.db.commit()

    async def get_case_evidence(self, case_id: int) -> list[dict]:
        """Load stored case evidence."""
        cursor = await self.db.execute(
            "SELECT evidence_key, claim, stance, quality_label, summary, document_key, document_title, document_url, "
            "source_label, source_class, related_hypotheses_json, excerpt, time_anchors_json "
            "FROM case_evidence WHERE case_id = ? ORDER BY evidence_key ASC",
            (case_id,),
        )
        return [
            {
                "id": row[0],
                "claim": row[1],
                "stance": row[2],
                "quality_label": row[3],
                "summary": row[4],
                "document_id": row[5],
                "document_title": row[6],
                "document_url": row[7],
                "source_label": row[8],
                "source_class": row[9],
                "related_hypotheses": json.loads(row[10] or "[]"),
                "excerpt": row[11],
                "time_anchors": json.loads(row[12] or "[]"),
            }
            for row in await cursor.fetchall()
        ]

    async def replace_case_hypotheses(self, case_id: int, hypotheses: list[dict]) -> None:
        """Replace stored case hypotheses."""
        await self.db.execute("DELETE FROM case_hypotheses WHERE case_id = ?", (case_id,))
        for item in hypotheses:
            await self.db.execute(
                "INSERT INTO case_hypotheses "
                "(case_id, hypothesis_key, title, summary, confidence_label, evidence_for_json, evidence_against_json, "
                "unresolved_gaps_json, what_would_change_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    case_id,
                    item["id"],
                    item["title"],
                    item.get("summary", ""),
                    item.get("confidence_label", "Contested"),
                    json.dumps(item.get("evidence_for", [])),
                    json.dumps(item.get("evidence_against", [])),
                    json.dumps(item.get("unresolved_gaps", [])),
                    json.dumps(item.get("what_would_change_my_mind", [])),
                ),
            )
        await self.db.commit()

    async def get_case_hypotheses(self, case_id: int) -> list[dict]:
        """Load stored case hypotheses."""
        cursor = await self.db.execute(
            "SELECT hypothesis_key, title, summary, confidence_label, evidence_for_json, evidence_against_json, "
            "unresolved_gaps_json, what_would_change_json "
            "FROM case_hypotheses WHERE case_id = ? ORDER BY id ASC",
            (case_id,),
        )
        return [
            {
                "id": row[0],
                "title": row[1],
                "summary": row[2],
                "confidence_label": row[3],
                "evidence_for": json.loads(row[4] or "[]"),
                "evidence_against": json.loads(row[5] or "[]"),
                "unresolved_gaps": json.loads(row[6] or "[]"),
                "what_would_change_my_mind": json.loads(row[7] or "[]"),
            }
            for row in await cursor.fetchall()
        ]

    async def replace_case_open_questions(self, case_id: int, questions: list[str]) -> None:
        """Replace stored case open questions."""
        await self.db.execute("DELETE FROM case_open_questions WHERE case_id = ?", (case_id,))
        for index, question in enumerate(questions):
            await self.db.execute(
                "INSERT INTO case_open_questions (case_id, ordinal, question) VALUES (?, ?, ?)",
                (case_id, index, question),
            )
        await self.db.commit()

    async def get_case_open_questions(self, case_id: int) -> list[str]:
        """Load stored case open questions."""
        cursor = await self.db.execute(
            "SELECT question FROM case_open_questions WHERE case_id = ? ORDER BY ordinal ASC, id ASC",
            (case_id,),
        )
        return [row[0] for row in await cursor.fetchall()]

    async def replace_case_assessments(self, case_id: int, assessments: list[dict]) -> None:
        """Replace stored case assessments."""
        await self.db.execute("DELETE FROM case_assessments WHERE case_id = ?", (case_id,))
        for item in assessments:
            await self.db.execute(
                "INSERT INTO case_assessments "
                "(case_id, assessment_key, target_hypothesis_key, mode, question, probability, confidence, rationale, "
                "counterarguments_json, evidence_ids_json, evidence_thread_ids_json, signposts_json, metadata_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    case_id,
                    item["id"],
                    item["target_hypothesis_id"],
                    item.get("mode", "posterior"),
                    item["question"],
                    item.get("probability", 0.5),
                    item.get("confidence", "medium"),
                    item.get("rationale", ""),
                    json.dumps(item.get("counterarguments", [])),
                    json.dumps(item.get("evidence_ids", [])),
                    json.dumps(item.get("evidence_thread_ids", [])),
                    json.dumps(item.get("signposts", [])),
                    json.dumps(item.get("metadata", {})),
                ),
            )
        await self.db.commit()

    async def get_case_assessments(self, case_id: int) -> list[dict]:
        """Load stored case assessments."""
        cursor = await self.db.execute(
            "SELECT assessment_key, target_hypothesis_key, mode, question, probability, confidence, rationale, "
            "counterarguments_json, evidence_ids_json, evidence_thread_ids_json, signposts_json "
            "FROM case_assessments WHERE case_id = ? ORDER BY id ASC",
            (case_id,),
        )
        return [
            {
                "id": row[0],
                "target_hypothesis_id": row[1],
                "mode": row[2],
                "question": row[3],
                "probability": row[4],
                "confidence": row[5],
                "rationale": row[6],
                "counterarguments": json.loads(row[7] or "[]"),
                "evidence_ids": json.loads(row[8] or "[]"),
                "evidence_thread_ids": json.loads(row[9] or "[]"),
                "signposts": json.loads(row[10] or "[]"),
            }
            for row in await cursor.fetchall()
        ]

    async def get_threads_for_case(self, case_id: int) -> list[dict]:
        """Return threads linked to a case."""
        return await self.get_all_threads(case_id=case_id)

    async def get_case_divergence(self, case_id: int) -> list[dict]:
        """Return divergence rows across all threads in a case."""
        cursor = await self.db.execute(
            "SELECT d.thread_id, t.slug, t.headline, d.shared_event, d.source_a, d.framing_a, d.source_b, d.framing_b "
            "FROM divergence d "
            "JOIN thread_cases tc ON d.thread_id = tc.thread_id "
            "JOIN threads t ON d.thread_id = t.id "
            "WHERE tc.case_id = ? "
            "ORDER BY t.updated_at DESC, d.id ASC",
            (case_id,),
        )
        return [
            {
                "thread_id": row[0],
                "thread_slug": row[1],
                "thread_headline": row[2],
                "shared_event": row[3],
                "source_a": row[4],
                "framing_a": row[5],
                "source_b": row[6],
                "framing_b": row[7],
            }
            for row in await cursor.fetchall()
        ]

    async def get_case_convergence(self, case_id: int) -> list[dict]:
        """Return convergence rows across all threads in a case."""
        cursor = await self.db.execute(
            "SELECT c.thread_id, t.slug, t.headline, c.fact_text, c.confirmed_by "
            "FROM convergence c "
            "JOIN thread_cases tc ON c.thread_id = tc.thread_id "
            "JOIN threads t ON c.thread_id = t.id "
            "WHERE tc.case_id = ? "
            "ORDER BY t.updated_at DESC, c.id ASC",
            (case_id,),
        )
        return [
            {
                "thread_id": row[0],
                "thread_slug": row[1],
                "thread_headline": row[2],
                "fact_text": row[3],
                "confirmed_by": json.loads(row[4] or "[]"),
            }
            for row in await cursor.fetchall()
        ]

    async def find_entity_by_id(self, entity_id: int) -> dict | None:
        """Look up entity by ID."""
        cursor = await self.db.execute(
            "SELECT id, canonical_name, entity_type, aliases, first_seen, last_seen, thumbnail_url, wikipedia_url "
            "FROM entities WHERE id = ?",
            (entity_id,),
        )
        row = await cursor.fetchone()
        if row:
            return _entity_row_to_dict(row)
        return None

    async def update_entity_thumbnail(self, entity_id: int, url: str) -> None:
        """Set the thumbnail URL for an entity."""
        await self.db.execute(
            "UPDATE entities SET thumbnail_url = ? WHERE id = ?", (url, entity_id),
        )
        await self.db.commit()

    async def update_entity_media(
        self, entity_id: int, thumbnail_url: str = "", wikipedia_url: str = "",
    ) -> None:
        """Set thumbnail and/or Wikipedia URL for an entity."""
        updates, params = [], []
        if thumbnail_url:
            updates.append("thumbnail_url = ?")
            params.append(thumbnail_url)
        if wikipedia_url:
            updates.append("wikipedia_url = ?")
            params.append(wikipedia_url)
        if not updates:
            return
        params.append(entity_id)
        await self.db.execute(
            f"UPDATE entities SET {', '.join(updates)} WHERE id = ?", params,
        )
        await self.db.commit()

    async def search_entities(self, query: str, limit: int = 20) -> list[dict]:
        """Search entities by canonical name or alias (LIKE match)."""
        pattern = f"%{query}%"
        cursor = await self.db.execute(
            "SELECT id, canonical_name, entity_type, aliases, first_seen, last_seen, thumbnail_url, wikipedia_url "
            "FROM entities "
            "WHERE canonical_name LIKE ? OR aliases LIKE ? "
            "ORDER BY canonical_name "
            "LIMIT ?",
            (pattern, pattern, limit),
        )
        return [_entity_row_to_dict(r) for r in await cursor.fetchall()]

    async def get_topic_stats(self) -> list[dict]:
        """Per-topic aggregate stats for dashboard landing page."""
        cursor = await self.db.execute(
            "SELECT topic_slug, COUNT(*) as event_count, MAX(date) as latest_date "
            "FROM events WHERE case_id IS NULL GROUP BY topic_slug ORDER BY topic_slug"
        )
        rows = await cursor.fetchall()
        stats = []
        for r in rows:
            slug = r[0]
            # Count distinct entities for this topic
            ent_cursor = await self.db.execute(
                "SELECT COUNT(DISTINCT ee.entity_id) FROM event_entities ee "
                "JOIN events ev ON ee.event_id = ev.id WHERE ev.topic_slug = ?",
                (slug,),
            )
            entity_count = (await ent_cursor.fetchone())[0]
            # Count threads for this topic
            thread_cursor = await self.db.execute(
                "SELECT COUNT(DISTINCT tt.thread_id) FROM thread_topics tt "
                "JOIN threads t ON tt.thread_id = t.id "
                "WHERE tt.topic_slug = ? AND t.status != 'merged'",
                (slug,),
            )
            thread_count = (await thread_cursor.fetchone())[0]
            stats.append({
                "topic_slug": slug,
                "event_count": r[1],
                "entity_count": entity_count,
                "thread_count": thread_count,
                "latest_date": r[2],
            })
        return stats

    async def get_topic_event_range(self, topic_slug: str, *, until: date | None = None) -> dict:
        """Return first/last event dates plus event count for a topic."""
        query = "SELECT MIN(date), MAX(date), COUNT(*) FROM events WHERE topic_slug = ?"
        params: list = [topic_slug]
        if until:
            query += " AND date <= ?"
            params.append(until.isoformat())
        cursor = await self.db.execute(query, params)
        row = await cursor.fetchone()
        return {
            "first_date": date.fromisoformat(row[0]) if row and row[0] else None,
            "last_date": date.fromisoformat(row[1]) if row and row[1] else None,
            "event_count": row[2] if row else 0,
        }

    async def get_thread_event_stats(self, thread_id: int, *, until: date | None = None) -> dict:
        """Return event count and latest event date for a thread."""
        query = (
            "SELECT COUNT(*), MAX(ev.date) FROM thread_events te "
            "JOIN events ev ON te.event_id = ev.id WHERE te.thread_id = ?"
        )
        params: list = [thread_id]
        if until:
            query += " AND ev.date <= ?"
            params.append(until.isoformat())
        cursor = await self.db.execute(query, params)
        row = await cursor.fetchone()
        return {
            "event_count": row[0] if row else 0,
            "latest_event_date": date.fromisoformat(row[1]) if row and row[1] else None,
        }

    async def get_related_entities(
        self, entity_id: int, limit: int = 20,
    ) -> list[dict]:
        """Find entities that co-appear in events with the given entity."""
        cursor = await self.db.execute(
            "SELECT e.id, e.canonical_name, e.entity_type, COUNT(*) as co_count "
            "FROM entities e "
            "JOIN event_entities ee ON e.id = ee.entity_id "
            "WHERE ee.event_id IN ("
            "  SELECT event_id FROM event_entities WHERE entity_id = ?"
            ") AND e.id != ? "
            "GROUP BY e.id "
            "ORDER BY co_count DESC "
            "LIMIT ?",
            (entity_id, entity_id, limit),
        )
        return [
            {
                "id": r[0], "canonical_name": r[1],
                "entity_type": r[2], "co_occurrence_count": r[3],
            }
            for r in await cursor.fetchall()
        ]

    # ── Entity Relationships ──────────────────────────────────────────

    async def save_entity_relationship(self, rel: dict) -> int:
        """Insert an entity-entity relationship. Returns ID.

        Uses INSERT OR IGNORE so duplicates (same source, target, type, event)
        are idempotent.
        """
        await self.db.execute(
            "INSERT OR IGNORE INTO entity_relationships "
            "(source_entity_id, target_entity_id, relation_type, evidence_text, "
            "source_event_id, strength, valid_from) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                rel["source_entity_id"],
                rel["target_entity_id"],
                rel["relation_type"],
                rel.get("evidence_text", ""),
                rel.get("source_event_id"),
                rel.get("strength", 0.5),
                rel["valid_from"],
            ),
        )
        await self.db.commit()
        # Return the ID (either new or existing due to UNIQUE constraint)
        cursor = await self.db.execute(
            "SELECT id FROM entity_relationships "
            "WHERE source_entity_id = ? AND target_entity_id = ? "
            "AND relation_type = ? AND source_event_id IS ?",
            (
                rel["source_entity_id"],
                rel["target_entity_id"],
                rel["relation_type"],
                rel.get("source_event_id"),
            ),
        )
        row = await cursor.fetchone()
        return row[0] if row else 0

    async def invalidate_relationship(
        self, relationship_id: int, valid_until: date,
    ) -> None:
        """Set valid_until on an existing relationship (bi-temporal invalidation)."""
        await self.db.execute(
            "UPDATE entity_relationships SET valid_until = ? WHERE id = ?",
            (valid_until.isoformat(), relationship_id),
        )
        await self.db.commit()

    async def get_active_relationships_for_entity(
        self, entity_id: int, *, as_of: date | None = None,
    ) -> list[dict]:
        """All active relationships where entity is source or target.

        Active means: valid_from <= as_of AND (valid_until IS NULL OR valid_until > as_of).
        """
        cutoff = (as_of or date.today()).isoformat()
        cursor = await self.db.execute(
            "SELECT r.id, r.source_entity_id, r.target_entity_id, "
            "r.relation_type, r.evidence_text, r.source_event_id, "
            "r.strength, r.valid_from, r.valid_until, "
            "se.canonical_name AS source_entity_name, "
            "te.canonical_name AS target_entity_name "
            "FROM entity_relationships r "
            "JOIN entities se ON r.source_entity_id = se.id "
            "JOIN entities te ON r.target_entity_id = te.id "
            "WHERE (r.source_entity_id = ? OR r.target_entity_id = ?) "
            "AND r.valid_from <= ? "
            "AND (r.valid_until IS NULL OR r.valid_until > ?)",
            (entity_id, entity_id, cutoff, cutoff),
        )
        return [dict(r) for r in await cursor.fetchall()]

    async def get_relationships_between(
        self,
        entity_a_id: int,
        entity_b_id: int,
        *,
        as_of: date | None = None,
    ) -> list[dict]:
        """Direct relationships between two specific entities (either direction)."""
        cutoff = (as_of or date.today()).isoformat()
        cursor = await self.db.execute(
            "SELECT r.id, r.source_entity_id, r.target_entity_id, "
            "r.relation_type, r.evidence_text, r.source_event_id, "
            "r.strength, r.valid_from, r.valid_until "
            "FROM entity_relationships r "
            "WHERE ((r.source_entity_id = ? AND r.target_entity_id = ?) "
            "   OR (r.source_entity_id = ? AND r.target_entity_id = ?)) "
            "AND r.valid_from <= ? "
            "AND (r.valid_until IS NULL OR r.valid_until > ?)",
            (entity_a_id, entity_b_id, entity_b_id, entity_a_id, cutoff, cutoff),
        )
        return [dict(r) for r in await cursor.fetchall()]

    async def get_entity_neighborhood(
        self,
        entity_id: int,
        *,
        hops: int = 2,
        as_of: date | None = None,
        limit: int = 50,
    ) -> dict:
        """BFS traversal: entities within N hops with their relationships.

        Returns {entities: [{id, name, type, distance}], relationships: [...]}.
        """
        cutoff = (as_of or date.today()).isoformat()
        visited: set[int] = set()
        entities_out: list[dict] = []
        relationships_out: list[dict] = []
        frontier = {entity_id}

        for hop in range(1, hops + 1):
            if not frontier or len(entities_out) >= limit:
                break
            placeholders = ",".join("?" for _ in frontier)
            cursor = await self.db.execute(
                f"SELECT r.id, r.source_entity_id, r.target_entity_id, "
                f"r.relation_type, r.evidence_text, r.strength, "
                f"r.valid_from, r.valid_until, r.source_event_id "
                f"FROM entity_relationships r "
                f"WHERE (r.source_entity_id IN ({placeholders}) "
                f"   OR r.target_entity_id IN ({placeholders})) "
                f"AND r.valid_from <= ? "
                f"AND (r.valid_until IS NULL OR r.valid_until > ?)",
                (*frontier, *frontier, cutoff, cutoff),
            )
            rows = await cursor.fetchall()
            next_frontier: set[int] = set()
            for r in rows:
                rel = dict(r)
                relationships_out.append(rel)
                # Find the neighbor (the entity that's NOT in our current search set)
                for eid in (rel["source_entity_id"], rel["target_entity_id"]):
                    if eid != entity_id and eid not in visited:
                        next_frontier.add(eid)

            visited |= frontier
            frontier = next_frontier - visited

        # Fetch entity details for all discovered neighbors
        all_neighbor_ids = {
            eid
            for r in relationships_out
            for eid in (r["source_entity_id"], r["target_entity_id"])
            if eid != entity_id
        }
        if all_neighbor_ids:
            placeholders = ",".join("?" for _ in all_neighbor_ids)
            cursor = await self.db.execute(
                f"SELECT id, canonical_name, entity_type "
                f"FROM entities WHERE id IN ({placeholders})",
                tuple(all_neighbor_ids),
            )
            for row in await cursor.fetchall():
                entities_out.append({
                    "id": row[0],
                    "name": row[1],
                    "type": row[2],
                })

        # Deduplicate relationships by id
        seen_rel_ids: set[int] = set()
        unique_rels = []
        for r in relationships_out:
            if r["id"] not in seen_rel_ids:
                seen_rel_ids.add(r["id"])
                unique_rels.append(r)

        return {
            "entities": entities_out[:limit],
            "relationships": unique_rels,
        }

    async def get_relationship_timeline(
        self,
        entity_id: int,
        *,
        days: int = 30,
        reference_date: date | None = None,
    ) -> list[dict]:
        """Relationships created or invalidated recently for an entity.

        Returns relationships sorted by valid_from DESC, showing what changed.
        Includes both active and recently invalidated relationships.
        """
        ref = (reference_date or date.today()).isoformat()
        cutoff = (
            (reference_date or date.today()) - timedelta(days=days)
        ).isoformat()
        cursor = await self.db.execute(
            "SELECT r.id, r.source_entity_id, r.target_entity_id, "
            "r.relation_type, r.evidence_text, r.strength, "
            "r.valid_from, r.valid_until, r.source_event_id, "
            "se.canonical_name AS source_entity_name, "
            "te.canonical_name AS target_entity_name "
            "FROM entity_relationships r "
            "JOIN entities se ON r.source_entity_id = se.id "
            "JOIN entities te ON r.target_entity_id = te.id "
            "WHERE (r.source_entity_id = ? OR r.target_entity_id = ?) "
            "AND r.valid_from <= ? "
            "AND (r.valid_from >= ? OR (r.valid_until IS NOT NULL AND r.valid_until >= ?)) "
            "ORDER BY r.valid_from DESC",
            (entity_id, entity_id, ref, cutoff, cutoff),
        )
        return [dict(r) for r in await cursor.fetchall()]

    # ── Breaking Alerts ─────────────────────────────────────────────

    async def add_breaking_alert(
        self, headline_hash: str, headline: str,
        source_url: str, significance_score: int,
        topic_slug: str = "",
    ) -> int:
        """Record a breaking news alert (dedup by headline_hash + topic_slug)."""
        cursor = await self.db.execute(
            "INSERT OR IGNORE INTO breaking_alerts "
            "(headline_hash, headline, source_url, significance_score, topic_slug) "
            "VALUES (?, ?, ?, ?, ?)",
            (headline_hash, headline, source_url, significance_score, topic_slug),
        )
        await self.db.commit()
        return cursor.lastrowid or 0

    async def is_alerted(self, headline_hash: str, topic_slug: str = "") -> bool:
        """Check if a headline has already been alerted for a topic."""
        cursor = await self.db.execute(
            "SELECT 1 FROM breaking_alerts WHERE headline_hash = ? AND topic_slug = ?",
            (headline_hash, topic_slug),
        )
        return (await cursor.fetchone()) is not None

    async def get_alerted_hashes(
        self, hashes: list[str], topic_slug: str = "",
    ) -> set[str]:
        """Batch check which headline hashes are already alerted for a topic."""
        if not hashes:
            return set()
        placeholders = ",".join("?" for _ in hashes)
        cursor = await self.db.execute(
            f"SELECT headline_hash FROM breaking_alerts "
            f"WHERE topic_slug = ? AND headline_hash IN ({placeholders})",
            [topic_slug, *hashes],
        )
        return {r[0] for r in await cursor.fetchall()}

    async def get_recent_breaking_alerts(
        self, hours: int = 24, topic_slug: str | None = None,
    ) -> list[dict]:
        """Get recent breaking alerts, optionally filtered by topic."""
        if topic_slug is not None:
            cursor = await self.db.execute(
                "SELECT headline, source_url, significance_score, topic_slug, alerted_at "
                "FROM breaking_alerts "
                "WHERE alerted_at >= datetime('now', ?) AND topic_slug = ? "
                "ORDER BY alerted_at DESC",
                (f"-{hours} hours", topic_slug),
            )
        else:
            cursor = await self.db.execute(
                "SELECT headline, source_url, significance_score, topic_slug, alerted_at "
                "FROM breaking_alerts "
                "WHERE alerted_at >= datetime('now', ?) "
                "ORDER BY alerted_at DESC",
                (f"-{hours} hours",),
            )
        return [
            {
                "headline": r[0], "source_url": r[1],
                "significance_score": r[2], "topic_slug": r[3],
                "alerted_at": r[4],
            }
            for r in await cursor.fetchall()
        ]

    # ── Feedback ─────────────────────────────────────────────────────

    async def add_feedback(
        self, briefing_date: str, rating: str, comment: str | None = None,
    ) -> int:
        """Record user feedback on a briefing."""
        cursor = await self.db.execute(
            "INSERT INTO feedback (briefing_date, rating, comment) VALUES (?, ?, ?)",
            (briefing_date, rating, comment),
        )
        await self.db.commit()
        return cursor.lastrowid

    async def get_feedback(self, briefing_date: str | None = None) -> list[dict]:
        """Get feedback, optionally filtered by briefing date."""
        if briefing_date:
            cursor = await self.db.execute(
                "SELECT id, briefing_date, rating, comment, created_at "
                "FROM feedback WHERE briefing_date = ? ORDER BY created_at DESC",
                (briefing_date,),
            )
        else:
            cursor = await self.db.execute(
                "SELECT id, briefing_date, rating, comment, created_at "
                "FROM feedback ORDER BY created_at DESC"
            )
        return [
            {
                "id": r[0], "briefing_date": r[1], "rating": r[2],
                "comment": r[3], "created_at": r[4],
            }
            for r in await cursor.fetchall()
        ]

    # ── Breaking Feedback ─────────────────────────────────────────────

    async def add_breaking_feedback(
        self, headline_hash: str, topic_slug: str, feedback: str,
    ) -> int:
        """Record user feedback on a breaking news alert."""
        cursor = await self.db.execute(
            "INSERT INTO breaking_feedback (headline_hash, topic_slug, feedback) VALUES (?, ?, ?)",
            (headline_hash, topic_slug, feedback),
        )
        await self.db.commit()
        return cursor.lastrowid

    async def get_breaking_fp_rate(self) -> float:
        """Compute the false-positive rate from breaking feedback.

        FP rate = not_breaking / total feedback entries.
        Returns 0.0 if no feedback exists.
        """
        cursor = await self.db.execute(
            "SELECT COUNT(*) FROM breaking_feedback"
        )
        total = (await cursor.fetchone())[0]
        if total == 0:
            return 0.0
        cursor = await self.db.execute(
            "SELECT COUNT(*) FROM breaking_feedback WHERE feedback = 'not_breaking'"
        )
        not_breaking = (await cursor.fetchone())[0]
        return not_breaking / total

    # ── Feed Health ───────────────────────────────────────────────────

    async def record_feed_poll(
        self, source_url: str, topic_slug: str, *, success: bool, error: str | None = None,
    ) -> None:
        """Record the result of a feed poll for health tracking."""
        existing = await self.get_feed_health(source_url)
        if existing is None:
            consecutive = 0 if success else 1
            status = "healthy"
            await self.db.execute(
                "INSERT INTO feed_health "
                "(source_url, topic_slug, last_poll_at, last_success_at, "
                "consecutive_failures, total_polls, total_successes, last_error, status) "
                "VALUES (?, ?, datetime('now'), ?, ?, 1, ?, ?, ?)",
                (
                    source_url, topic_slug,
                    "datetime('now')" if success else None,
                    consecutive, 1 if success else 0,
                    error, status,
                ),
            )
            # Fix last_success_at for successful first poll
            if success:
                await self.db.execute(
                    "UPDATE feed_health SET last_success_at = datetime('now') WHERE source_url = ?",
                    (source_url,),
                )
        else:
            total_polls = existing["total_polls"] + 1
            total_successes = existing["total_successes"] + (1 if success else 0)
            if success:
                consecutive = 0
            else:
                consecutive = existing["consecutive_failures"] + 1

            if consecutive >= 5:
                status = "dead"
            elif consecutive >= 3:
                status = "degraded"
            else:
                status = "healthy"

            if success:
                await self.db.execute(
                    "UPDATE feed_health SET last_poll_at = datetime('now'), "
                    "last_success_at = datetime('now'), consecutive_failures = 0, "
                    "total_polls = ?, total_successes = ?, last_error = NULL, status = ? "
                    "WHERE source_url = ?",
                    (total_polls, total_successes, status, source_url),
                )
            else:
                await self.db.execute(
                    "UPDATE feed_health SET last_poll_at = datetime('now'), "
                    "consecutive_failures = ?, total_polls = ?, total_successes = ?, "
                    "last_error = ?, status = ? WHERE source_url = ?",
                    (consecutive, total_polls, total_successes, error, status, source_url),
                )
        await self.db.commit()

    async def get_feed_health(self, source_url: str) -> dict | None:
        """Get health record for a single feed."""
        cursor = await self.db.execute(
            "SELECT source_url, topic_slug, last_poll_at, last_success_at, "
            "consecutive_failures, total_polls, total_successes, last_error, status "
            "FROM feed_health WHERE source_url = ?",
            (source_url,),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return {
            "source_url": row[0], "topic_slug": row[1],
            "last_poll_at": row[2], "last_success_at": row[3],
            "consecutive_failures": row[4], "total_polls": row[5],
            "total_successes": row[6], "last_error": row[7], "status": row[8],
        }

    async def get_dead_feeds(self) -> list[dict]:
        """Get all feeds flagged as dead."""
        cursor = await self.db.execute(
            "SELECT source_url, topic_slug, last_poll_at, last_success_at, "
            "consecutive_failures, total_polls, total_successes, last_error, status "
            "FROM feed_health WHERE status = 'dead'"
        )
        return [
            {
                "source_url": r[0], "topic_slug": r[1],
                "last_poll_at": r[2], "last_success_at": r[3],
                "consecutive_failures": r[4], "total_polls": r[5],
                "total_successes": r[6], "last_error": r[7], "status": r[8],
            }
            for r in await cursor.fetchall()
        ]

    async def get_all_feed_health(self) -> list[dict]:
        """Get health records for all tracked feeds."""
        cursor = await self.db.execute(
            "SELECT source_url, topic_slug, last_poll_at, last_success_at, "
            "consecutive_failures, total_polls, total_successes, last_error, status "
            "FROM feed_health ORDER BY status DESC, consecutive_failures DESC"
        )
        return [
            {
                "source_url": r[0], "topic_slug": r[1],
                "last_poll_at": r[2], "last_success_at": r[3],
                "consecutive_failures": r[4], "total_polls": r[5],
                "total_successes": r[6], "last_error": r[7], "status": r[8],
            }
            for r in await cursor.fetchall()
        ]

    async def _build_event_from_row(self, row) -> Event:
        """Build an Event object from a DB row, loading sources and entities."""
        event_id = row[0]
        src_cursor = await self.db.execute(
            "SELECT url, outlet, affiliation, country, language, framing "
            "FROM event_sources WHERE event_id = ?",
            (event_id,),
        )
        sources = [
            {"url": s[0], "outlet": s[1], "affiliation": s[2], "country": s[3], "language": s[4], "framing": s[5]}
            for s in await src_cursor.fetchall()
        ]
        ent_cursor = await self.db.execute(
            "SELECT e.canonical_name FROM entities e "
            "JOIN event_entities ee ON e.id = ee.entity_id "
            "WHERE ee.event_id = ?",
            (event_id,),
        )
        entities = [e[0] for e in await ent_cursor.fetchall()]
        raw_entities = json.loads(row[5]) if len(row) > 5 and row[5] else []
        return Event(
            event_id=event_id,
            date=date.fromisoformat(row[1]),
            summary=row[2],
            significance=row[3],
            relation_to_prior=row[4],
            sources=sources,
            entities=entities or raw_entities,
            raw_entities=raw_entities,
        )

    # ── Usage Log ─────────────────────────────────────────────────────

    async def add_usage_record(
        self, date: str, provider: str, model: str, config_key: str,
        input_tokens: int, output_tokens: int, cost_usd: float,
    ) -> int:
        """Insert a usage log record. Returns the row ID."""
        if self.db is None:
            return 0  # store already closed — silently drop the record
        cursor = await self.db.execute(
            "INSERT INTO usage_log (date, provider, model, config_key, "
            "input_tokens, output_tokens, cost_usd) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (date, provider, model, config_key, input_tokens, output_tokens, cost_usd),
        )
        await self.db.commit()
        return cursor.lastrowid

    async def get_usage_summary(self, since_date: str | None = None) -> list[dict]:
        """Aggregate usage by date."""
        query = (
            "SELECT date, SUM(input_tokens), SUM(output_tokens), SUM(cost_usd), COUNT(*) "
            "FROM usage_log"
        )
        params: list = []
        if since_date:
            query += " WHERE date >= ?"
            params.append(since_date)
        query += " GROUP BY date ORDER BY date ASC"

        cursor = await self.db.execute(query, params)
        return [
            {
                "date": r[0],
                "total_input_tokens": r[1],
                "total_output_tokens": r[2],
                "total_cost_usd": r[3],
                "call_count": r[4],
            }
            for r in await cursor.fetchall()
        ]

    async def count_events_for_date(self, target_date) -> int:
        """Count events ingested on a specific date."""
        date_str = target_date.isoformat() if hasattr(target_date, "isoformat") else str(target_date)
        cursor = await self.db.execute(
            "SELECT COUNT(*) FROM events WHERE date = ?", (date_str,),
        )
        row = await cursor.fetchone()
        return row[0] if row else 0

    async def count_threads_updated_since(self, target_date) -> int:
        """Count threads updated on or after a given date."""
        date_str = target_date.isoformat() if hasattr(target_date, "isoformat") else str(target_date)
        cursor = await self.db.execute(
            "SELECT COUNT(*) FROM threads WHERE updated_at >= ?", (date_str,),
        )
        row = await cursor.fetchone()
        return row[0] if row else 0

    async def count_threads_created_since(self, target_date) -> int:
        """Count threads created on or after a given date."""
        date_str = target_date.isoformat() if hasattr(target_date, "isoformat") else str(target_date)
        cursor = await self.db.execute(
            "SELECT COUNT(*) FROM threads WHERE created_at >= ?", (date_str,),
        )
        row = await cursor.fetchone()
        return row[0] if row else 0

    async def purge_empty_threads(self, dry_run: bool = True) -> dict:
        """Delete threads with 0 events (orphaned by thread merges).

        Returns: {purged: int, thread_ids: list}
        """
        cursor = await self.db.execute(
            "SELECT t.id FROM threads t "
            "LEFT JOIN thread_events te ON te.thread_id = t.id "
            "LEFT JOIN threads merged_children ON merged_children.merged_into_id = t.id "
            "WHERE t.status != 'merged' "
            "GROUP BY t.id "
            "HAVING COUNT(DISTINCT te.event_id) = 0 "
            "AND COUNT(DISTINCT merged_children.id) = 0"
        )
        rows = await cursor.fetchall()
        thread_ids = [r[0] for r in rows]

        if dry_run or not thread_ids:
            return {"purged": len(thread_ids), "thread_ids": thread_ids, "dry_run": dry_run}

        placeholders = ",".join("?" for _ in thread_ids)
        # FK-ordered delete: child tables first, then threads
        await self.db.execute(
            f"DELETE FROM thread_snapshots WHERE thread_id IN ({placeholders})",
            thread_ids,
        )
        await self.db.execute(
            f"DELETE FROM convergence WHERE thread_id IN ({placeholders})",
            thread_ids,
        )
        await self.db.execute(
            f"DELETE FROM divergence WHERE thread_id IN ({placeholders})",
            thread_ids,
        )
        await self.db.execute(
            f"DELETE FROM thread_topics WHERE thread_id IN ({placeholders})",
            thread_ids,
        )
        await self.db.execute(
            f"DELETE FROM threads WHERE id IN ({placeholders})",
            thread_ids,
        )
        await self.db.commit()
        return {"purged": len(thread_ids), "thread_ids": thread_ids, "dry_run": False}

    async def get_daily_cost(self, date: str) -> float:
        """Get total cost for a specific date. Returns 0.0 if no data."""
        cursor = await self.db.execute(
            "SELECT SUM(cost_usd) FROM usage_log WHERE date = ?",
            (date,),
        )
        row = await cursor.fetchone()
        return row[0] if row[0] is not None else 0.0

    # ── Migration ─────────────────────────────────────────────────────

    async def import_events_from_yaml(self, events: list[Event], topic_slug: str) -> int:
        """Import a list of Event Pydantic models into the store. Returns count imported."""
        ids = await self.add_events(events, topic_slug)
        return len(ids)

    async def import_summaries_from_yaml(
        self, summaries: list[Summary], topic_slug: str, period_type: str,
    ) -> int:
        """Import Summary models into the store. Returns count imported."""
        for s in summaries:
            await self.add_summary(s, topic_slug, period_type)
        return len(summaries)


    # ── Pipeline Runs ──────────────────────────────────────────────

    async def start_pipeline_run(
        self, topics: list[str], trigger: str = "manual",
    ) -> int:
        """Record pipeline start. Returns run ID."""
        # Clean up stale runs (running > 2 hours = presumed crashed)
        await self.db.execute(
            "UPDATE pipeline_runs SET status = 'failed', "
            "error = 'Stale: exceeded 3h timeout', "
            "completed_at = datetime('now') "
            "WHERE status = 'running' "
            "AND started_at < datetime('now', '-3 hours')"
        )
        cursor = await self.db.execute(
            "INSERT INTO pipeline_runs (topics, trigger) VALUES (?, ?)",
            (json.dumps(topics), trigger),
        )
        await self.db.commit()
        return cursor.lastrowid

    async def complete_pipeline_run(
        self, run_id: int, article_count: int = 0,
        event_count: int = 0, cost_usd: float = 0.0,
        skipped_topics: list[dict] | None = None,
    ) -> None:
        """Mark pipeline run as completed."""
        await self.db.execute(
            "UPDATE pipeline_runs SET status = 'completed', "
            "completed_at = datetime('now'), "
            "article_count = ?, event_count = ?, cost_usd = ?, "
            "skipped_topics = ? "
            "WHERE id = ?",
            (article_count, event_count, cost_usd,
             json.dumps(skipped_topics or []), run_id),
        )
        await self.db.commit()

    async def fail_pipeline_run(self, run_id: int, error: str) -> None:
        """Mark pipeline run as failed."""
        await self.db.execute(
            "UPDATE pipeline_runs SET status = 'failed', "
            "completed_at = datetime('now'), error = ? "
            "WHERE id = ?",
            (error, run_id),
        )
        await self.db.commit()

    async def get_last_pipeline_run(self) -> dict | None:
        """Get most recent pipeline run."""
        cursor = await self.db.execute(
            "SELECT id, started_at, completed_at, status, topics, "
            "article_count, event_count, cost_usd, error, trigger, "
            "skipped_topics "
            "FROM pipeline_runs ORDER BY id DESC LIMIT 1"
        )
        row = await cursor.fetchone()
        if not row:
            return None
        return {
            "id": row[0], "started_at": row[1], "completed_at": row[2],
            "status": row[3], "topics": json.loads(row[4]),
            "article_count": row[5], "event_count": row[6],
            "cost_usd": row[7], "error": row[8], "trigger": row[9],
            "skipped_topics": json.loads(row[10]) if row[10] else [],
        }

    async def is_pipeline_running(self) -> bool:
        """Check if any pipeline run is in 'running' status."""
        # First clean up stale runs
        await self.db.execute(
            "UPDATE pipeline_runs SET status = 'failed', "
            "error = 'Stale: exceeded 3h timeout', "
            "completed_at = datetime('now') "
            "WHERE status = 'running' "
            "AND started_at < datetime('now', '-3 hours')"
        )
        await self.db.commit()
        cursor = await self.db.execute(
            "SELECT COUNT(*) FROM pipeline_runs WHERE status = 'running'"
        )
        row = await cursor.fetchone()
        return row[0] > 0


def _entity_row_to_dict(row) -> dict:
    """Convert an entity row to a dict."""
    d = {
        "id": row[0],
        "canonical_name": row[1],
        "entity_type": row[2],
        "aliases": json.loads(row[3]) if row[3] else [],
        "first_seen": row[4],
        "last_seen": row[5],
    }
    d["thumbnail_url"] = row[6] if len(row) > 6 else ""
    d["wikipedia_url"] = row[7] if len(row) > 7 else ""
    return d
