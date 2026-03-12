"""SQLite-backed knowledge store. Returns existing Pydantic models."""

import json
import logging
from datetime import date, datetime, timedelta
from pathlib import Path

import aiosqlite

from nexus.engine.knowledge.events import Event
from nexus.engine.knowledge.compression import Summary
from nexus.engine.knowledge.schema import initialize_schema

logger = logging.getLogger(__name__)


class KnowledgeStore:
    """Async SQLite knowledge store.

    Wraps the knowledge graph DB and returns existing Pydantic models
    so downstream consumers (renderers, judge, metrics) don't change.
    """

    def __init__(self, db_path: Path):
        self._db_path = db_path
        self._db: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        """Open connection, create tables if needed."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(str(self._db_path))
        self._db.row_factory = aiosqlite.Row
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA foreign_keys=ON")
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

    async def add_events(self, events: list[Event], topic_slug: str) -> list[int]:
        """Insert events and their sources. Returns list of new event row IDs."""
        ids = []
        for event in events:
            cursor = await self.db.execute(
                "INSERT INTO events (date, summary, significance, relation_to_prior, topic_slug) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    event.date.isoformat(),
                    event.summary,
                    event.significance,
                    event.relation_to_prior,
                    topic_slug,
                ),
            )
            event_id = cursor.lastrowid
            ids.append(event_id)

            # Insert sources
            for src in event.sources:
                await self.db.execute(
                    "INSERT INTO event_sources (event_id, url, outlet, affiliation, country, language) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        event_id,
                        src.get("url", ""),
                        src.get("outlet", ""),
                        src.get("affiliation", ""),
                        src.get("country", ""),
                        src.get("language", "en"),
                    ),
                )

            # Insert raw entity strings (unresolved — linked to entity table in Phase 3)
            # For now, store entity names in a lightweight join table
            # that just tracks the raw string per event for reconstruction.
            for entity_name in event.entities:
                # Upsert into entities table with 'unknown' type
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

        await self.db.commit()
        return ids

    async def get_events(
        self,
        topic_slug: str,
        since: date | None = None,
        until: date | None = None,
        limit: int | None = None,
    ) -> list[Event]:
        """Load events for a topic, optionally filtered by date range."""
        query = "SELECT id, date, summary, significance, relation_to_prior FROM events WHERE topic_slug = ?"
        params: list = [topic_slug]

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
                "SELECT url, outlet, affiliation, country, language "
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

            events.append(Event(
                date=date.fromisoformat(row[1]),
                summary=row[2],
                significance=row[3],
                relation_to_prior=row[4],
                sources=sources,
                entities=entities,
            ))

        return events

    async def get_recent_events(
        self, topic_slug: str, days: int = 7, limit: int = 30,
        reference_date: date | None = None,
    ) -> list[Event]:
        """Get events from the last N days."""
        ref = reference_date or date.today()
        since = ref - timedelta(days=days)
        events = await self.get_events(topic_slug, since=since, until=ref)
        return events[-limit:] if len(events) > limit else events

    async def get_all_events(self, topic_slug: str | None = None) -> list[Event]:
        """Get all events, optionally filtered by topic."""
        if topic_slug:
            return await self.get_events(topic_slug)
        # Cross-topic: get all
        cursor = await self.db.execute(
            "SELECT DISTINCT topic_slug FROM events ORDER BY topic_slug"
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
    ) -> int:
        """Insert or update an entity. Returns the entity ID."""
        aliases_json = json.dumps(aliases or [])
        today = date.today().isoformat()

        await self.db.execute(
            "INSERT INTO entities (canonical_name, entity_type, aliases, first_seen, last_seen) "
            "VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(canonical_name) DO UPDATE SET "
            "entity_type = CASE WHEN excluded.entity_type != 'unknown' THEN excluded.entity_type ELSE entities.entity_type END, "
            "aliases = excluded.aliases, "
            "last_seen = excluded.last_seen",
            (canonical_name, entity_type, aliases_json, today, today),
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

    async def get_all_entities(self, topic_slug: str | None = None) -> list[dict]:
        """Get all entities, optionally scoped to a topic."""
        if topic_slug:
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
            "SELECT ev.id, ev.date, ev.summary, ev.significance, ev.relation_to_prior, ev.topic_slug "
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
                "SELECT url, outlet, affiliation, country, language "
                "FROM event_sources WHERE event_id = ?",
                (event_id,),
            )
            sources = [
                {"url": s[0], "outlet": s[1], "affiliation": s[2], "country": s[3], "language": s[4]}
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
                date=date.fromisoformat(row[1]),
                summary=row[2],
                significance=row[3],
                relation_to_prior=row[4],
                sources=sources,
                entities=entities,
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
            "updated_at = excluded.updated_at",
            (slug, headline, significance, status, now, now),
        )
        await self.db.commit()
        cursor = await self.db.execute(
            "SELECT id FROM threads WHERE slug = ?", (slug,)
        )
        row = await cursor.fetchone()
        return row[0]

    async def get_active_threads(self, topic_slug: str | None = None) -> list[dict]:
        """Get threads with status 'emerging' or 'active'."""
        if topic_slug:
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

            threads.append({
                "id": thread_id,
                "slug": r[1],
                "headline": r[2],
                "status": r[3],
                "significance": r[4],
                "created_at": r[5],
                "updated_at": r[6],
                "key_entities": key_entities,
            })
        return threads

    async def link_thread_events(self, thread_id: int, event_ids: list[int]) -> None:
        """Link events to a thread."""
        for eid in event_ids:
            await self.db.execute(
                "INSERT OR IGNORE INTO thread_events (thread_id, event_id) VALUES (?, ?)",
                (thread_id, eid),
            )
        await self.db.commit()

    async def find_event_id(
        self, summary: str, date: str, topic_slug: str,
    ) -> int | None:
        """Find an event's DB id by its natural key (summary + date + topic)."""
        cursor = await self.db.execute(
            "SELECT id FROM events WHERE summary = ? AND date = ? AND topic_slug = ?",
            (summary, date, topic_slug),
        )
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

    async def merge_threads(self, keep_id: int, absorb_id: int) -> None:
        """Merge absorb_id thread into keep_id. Reassign events, topics, analysis."""
        # Reassign thread_events (ignore duplicates)
        await self.db.execute(
            "UPDATE OR IGNORE thread_events SET thread_id = ? WHERE thread_id = ?",
            (keep_id, absorb_id),
        )
        # Remove any remaining (duplicate) rows for absorbed thread
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
        # Move convergence + divergence
        await self.db.execute(
            "UPDATE convergence SET thread_id = ? WHERE thread_id = ?",
            (keep_id, absorb_id),
        )
        await self.db.execute(
            "UPDATE divergence SET thread_id = ? WHERE thread_id = ?",
            (keep_id, absorb_id),
        )
        # Mark absorbed thread as resolved
        await self.db.execute(
            "UPDATE threads SET status = 'resolved' WHERE id = ?", (absorb_id,),
        )
        await self.db.commit()

    async def link_thread_topic(self, thread_id: int, topic_slug: str) -> None:
        """Link a thread to a topic."""
        await self.db.execute(
            "INSERT OR IGNORE INTO thread_topics (thread_id, topic_slug) VALUES (?, ?)",
            (thread_id, topic_slug),
        )
        await self.db.commit()

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
            "SELECT id, date, summary, significance, relation_to_prior "
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
            "SELECT ev.id, ev.date, ev.summary, ev.significance, ev.relation_to_prior "
            "FROM events ev "
            "JOIN thread_events te ON ev.id = te.event_id "
            "WHERE te.thread_id = ? "
            "ORDER BY ev.date ASC",
            (thread_id,),
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
        return {
            "id": thread_id, "slug": row[1], "headline": row[2],
            "status": row[3], "significance": row[4],
            "created_at": row[5], "updated_at": row[6],
            "key_entities": key_entities,
        }

    async def get_all_threads(
        self, topic_slug: str | None = None, status: str | None = None,
    ) -> list[dict]:
        """Get threads with optional topic and status filters."""
        if topic_slug:
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
            query += " ORDER BY t.updated_at DESC"
        else:
            query = (
                "SELECT id, slug, headline, status, significance, created_at, updated_at "
                "FROM threads"
            )
            params = []
            if status:
                query += " WHERE status = ?"
                params.append(status)
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
            threads.append({
                "id": thread_id, "slug": r[1], "headline": r[2],
                "status": r[3], "significance": r[4],
                "created_at": r[5], "updated_at": r[6],
                "key_entities": key_entities,
            })
        return threads

    async def get_threads_for_entity(self, entity_id: int) -> list[dict]:
        """Get threads involving a specific entity (via event linkage)."""
        cursor = await self.db.execute(
            "SELECT DISTINCT t.id, t.slug, t.headline, t.status, t.significance, "
            "t.created_at, t.updated_at "
            "FROM threads t "
            "JOIN thread_events te ON t.id = te.thread_id "
            "JOIN event_entities ee ON te.event_id = ee.event_id "
            "WHERE ee.entity_id = ? "
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

    async def get_source_stats(self, topic_slug: str | None = None) -> list[dict]:
        """Aggregate event sources by outlet/affiliation/country."""
        if topic_slug:
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
                "SELECT outlet, affiliation, country, language, COUNT(*) as cnt "
                "FROM event_sources "
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
            "FROM events GROUP BY topic_slug ORDER BY topic_slug"
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
                "WHERE tt.topic_slug = ?",
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

    async def _build_event_from_row(self, row) -> Event:
        """Build an Event object from a DB row, loading sources and entities."""
        event_id = row[0]
        src_cursor = await self.db.execute(
            "SELECT url, outlet, affiliation, country, language "
            "FROM event_sources WHERE event_id = ?",
            (event_id,),
        )
        sources = [
            {"url": s[0], "outlet": s[1], "affiliation": s[2], "country": s[3], "language": s[4]}
            for s in await src_cursor.fetchall()
        ]
        ent_cursor = await self.db.execute(
            "SELECT e.canonical_name FROM entities e "
            "JOIN event_entities ee ON e.id = ee.entity_id "
            "WHERE ee.event_id = ?",
            (event_id,),
        )
        entities = [e[0] for e in await ent_cursor.fetchall()]
        return Event(
            date=date.fromisoformat(row[1]),
            summary=row[2],
            significance=row[3],
            relation_to_prior=row[4],
            sources=sources,
            entities=entities,
        )

    # ── Usage Log ─────────────────────────────────────────────────────

    async def add_usage_record(
        self, date: str, provider: str, model: str, config_key: str,
        input_tokens: int, output_tokens: int, cost_usd: float,
    ) -> int:
        """Insert a usage log record. Returns the row ID."""
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
