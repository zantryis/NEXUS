"""Demo database seeder — exports a subset of live data for demo/showcase use.

Usage:
    python -m nexus demo seed              # Seed from live data
    python -m nexus demo seed --from-scratch  # Seed with minimal static data
    python -m nexus demo serve             # Start demo server
"""

import asyncio
import json
import logging
import shutil
from datetime import date, timedelta
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


async def _seed_from_live(src_db: Path, dest_db: Path, data_dir: Path) -> dict:
    """Export a sanitized subset of live knowledge.db for demo use."""
    import aiosqlite
    from nexus.engine.knowledge.schema import initialize_schema

    stats = {}

    # Initialize fresh schema
    db = await aiosqlite.connect(str(dest_db))
    await db.execute("PRAGMA journal_mode=WAL")
    await db.execute("PRAGMA foreign_keys=ON")
    await initialize_schema(db)

    src = await aiosqlite.connect(str(src_db))
    src.row_factory = aiosqlite.Row

    today = date.today()
    cutoff = (today - timedelta(days=14)).isoformat()

    # Copy recent events (last 14 days, all topics)
    cursor = await src.execute(
        "SELECT * FROM events WHERE date >= ? ORDER BY date DESC LIMIT 200", (cutoff,)
    )
    events = await cursor.fetchall()
    event_ids = set()
    for e in events:
        event_ids.add(e["id"])
        await db.execute(
            "INSERT INTO events (id, date, summary, significance, relation_to_prior, raw_entities, topic_slug, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (e["id"], e["date"], e["summary"], e["significance"],
             e["relation_to_prior"], e["raw_entities"], e["topic_slug"], e["created_at"]),
        )
    stats["events"] = len(events)

    # Copy entities referenced by copied events
    if event_ids:
        placeholders = ",".join("?" * len(event_ids))
        cursor = await src.execute(
            f"SELECT DISTINCT entity_id FROM event_entities WHERE event_id IN ({placeholders})",
            list(event_ids),
        )
        entity_ids = {row[0] for row in await cursor.fetchall()}

        for eid in entity_ids:
            cursor = await src.execute("SELECT * FROM entities WHERE id = ?", (eid,))
            ent = await cursor.fetchone()
            if ent:
                await db.execute(
                    "INSERT OR IGNORE INTO entities (id, canonical_name, entity_type, aliases, first_seen, last_seen, thumbnail_url, wikipedia_url) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (ent["id"], ent["canonical_name"], ent["entity_type"], ent["aliases"],
                     ent["first_seen"], ent["last_seen"], ent["thumbnail_url"], ent["wikipedia_url"]),
                )

        # Copy event_entities junction
        cursor = await src.execute(
            f"SELECT * FROM event_entities WHERE event_id IN ({placeholders})",
            list(event_ids),
        )
        for row in await cursor.fetchall():
            await db.execute(
                "INSERT OR IGNORE INTO event_entities (event_id, entity_id, role) VALUES (?, ?, ?)",
                (row["event_id"], row["entity_id"], row["role"]),
            )
        stats["entities"] = len(entity_ids)

    # Copy active threads
    cursor = await src.execute(
        "SELECT * FROM threads WHERE status IN ('emerging', 'active') ORDER BY updated_at DESC LIMIT 30"
    )
    threads = await cursor.fetchall()
    thread_ids = set()
    for t in threads:
        thread_ids.add(t["id"])
        await db.execute(
            "INSERT INTO threads (id, slug, headline, status, significance, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (t["id"], t["slug"], t["headline"], t["status"],
             t["significance"], t["created_at"], t["updated_at"]),
        )
    stats["threads"] = len(threads)

    # Copy thread_topics and thread_events for copied threads
    if thread_ids:
        t_placeholders = ",".join("?" * len(thread_ids))
        cursor = await src.execute(
            f"SELECT * FROM thread_topics WHERE thread_id IN ({t_placeholders})",
            list(thread_ids),
        )
        for row in await cursor.fetchall():
            await db.execute(
                "INSERT OR IGNORE INTO thread_topics (thread_id, topic_slug) VALUES (?, ?)",
                (row["thread_id"], row["topic_slug"]),
            )

        cursor = await src.execute(
            f"SELECT * FROM thread_events WHERE thread_id IN ({t_placeholders})",
            list(thread_ids),
        )
        for row in await cursor.fetchall():
            if row["event_id"] in event_ids:
                await db.execute(
                    "INSERT OR IGNORE INTO thread_events (thread_id, event_id, added_date) VALUES (?, ?, ?)",
                    (row["thread_id"], row["event_id"], row["added_date"]),
                )

    # Copy recent syntheses (latest per topic)
    cursor = await src.execute(
        "SELECT * FROM syntheses WHERE date >= ? ORDER BY date DESC LIMIT 10", (cutoff,)
    )
    syntheses = await cursor.fetchall()
    for s in syntheses:
        await db.execute(
            "INSERT INTO syntheses (id, topic_slug, date, data_json, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (s["id"], s["topic_slug"], s["date"], s["data_json"], s["created_at"]),
        )
    stats["syntheses"] = len(syntheses)

    # Copy recent breaking alerts
    cursor = await src.execute(
        "SELECT * FROM breaking_alerts ORDER BY alerted_at DESC LIMIT 20"
    )
    alerts = await cursor.fetchall()
    for a in alerts:
        await db.execute(
            "INSERT OR IGNORE INTO breaking_alerts (id, headline_hash, headline, source_url, significance_score, topic_slug, alerted_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (a["id"], a["headline_hash"], a["headline"], a["source_url"],
             a["significance_score"], a["topic_slug"], a["alerted_at"]),
        )
    stats["breaking_alerts"] = len(alerts)

    # Copy forecast runs + questions + resolutions
    cursor = await src.execute(
        "SELECT * FROM forecast_runs WHERE generated_for >= ? ORDER BY generated_for DESC LIMIT 10",
        (cutoff,),
    )
    runs = await cursor.fetchall()
    run_ids = set()
    for r in runs:
        run_ids.add(r["id"])
        await db.execute(
            "INSERT INTO forecast_runs (id, topic_slug, topic_name, engine, generated_for, summary, metadata_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (r["id"], r["topic_slug"], r["topic_name"], r["engine"],
             r["generated_for"], r["summary"], r["metadata_json"], r["created_at"]),
        )

    q_count = 0
    if run_ids:
        r_placeholders = ",".join("?" * len(run_ids))
        cursor = await src.execute(
            f"SELECT * FROM forecast_questions WHERE forecast_run_id IN ({r_placeholders})",
            list(run_ids),
        )
        questions = await cursor.fetchall()
        q_ids = set()
        for q in questions:
            q_ids.add(q["id"])
            cols = [desc[0] for desc in cursor.description]
            vals = [q[c] for c in cols]
            placeholders_q = ",".join("?" * len(cols))
            await db.execute(
                f"INSERT INTO forecast_questions ({','.join(cols)}) VALUES ({placeholders_q})",
                vals,
            )
        q_count = len(questions)

        # Resolutions
        if q_ids:
            q_placeholders = ",".join("?" * len(q_ids))
            cursor = await src.execute(
                f"SELECT * FROM forecast_resolutions WHERE forecast_question_id IN ({q_placeholders})",
                list(q_ids),
            )
            for row in await cursor.fetchall():
                cols = [desc[0] for desc in cursor.description]
                vals = [row[c] for c in cols]
                placeholders_r = ",".join("?" * len(cols))
                await db.execute(
                    f"INSERT OR IGNORE INTO forecast_resolutions ({','.join(cols)}) VALUES ({placeholders_r})",
                    vals,
                )

    stats["forecast_questions"] = q_count

    # Copy recent pipeline runs
    cursor = await src.execute(
        "SELECT * FROM pipeline_runs ORDER BY started_at DESC LIMIT 5"
    )
    for row in await cursor.fetchall():
        cols = [desc[0] for desc in cursor.description]
        vals = [row[c] for c in cols]
        placeholders_p = ",".join("?" * len(cols))
        await db.execute(
            f"INSERT INTO pipeline_runs ({','.join(cols)}) VALUES ({placeholders_p})",
            vals,
        )

    # Copy entity relationships
    if entity_ids:
        e_placeholders = ",".join("?" * len(entity_ids))
        try:
            cursor = await src.execute(
                f"SELECT * FROM entity_relationships WHERE source_entity_id IN ({e_placeholders}) OR target_entity_id IN ({e_placeholders})",
                list(entity_ids) + list(entity_ids),
            )
            for row in await cursor.fetchall():
                cols = [desc[0] for desc in cursor.description]
                vals = [row[c] for c in cols]
                placeholders_er = ",".join("?" * len(cols))
                await db.execute(
                    f"INSERT OR IGNORE INTO entity_relationships ({','.join(cols)}) VALUES ({placeholders_er})",
                    vals,
                )
        except Exception:
            pass  # Table may not exist in older DBs

    await db.commit()
    await db.close()
    await src.close()

    # Copy config (sanitized — strip API keys)
    config_path = data_dir / "config.yaml"
    demo_config_path = dest_db.parent / "config.yaml"
    if config_path.exists():
        config = yaml.safe_load(config_path.read_text())
        # Keep topics and user prefs, strip anything sensitive
        safe_config = {
            "user": config.get("user", {"name": "Demo User", "timezone": "UTC"}),
            "topics": config.get("topics", []),
        }
        demo_config_path.write_text(yaml.dump(safe_config, default_flow_style=False))

    # Copy source registries
    sources_dir = data_dir / "sources"
    demo_sources = dest_db.parent / "sources"
    if sources_dir.exists():
        if demo_sources.exists():
            shutil.rmtree(demo_sources)
        shutil.copytree(sources_dir, demo_sources)

    return stats


async def _seed_from_scratch(dest_db: Path) -> dict:
    """Create a minimal demo database with static seed data."""
    import aiosqlite
    from nexus.engine.knowledge.schema import initialize_schema

    db = await aiosqlite.connect(str(dest_db))
    await db.execute("PRAGMA journal_mode=WAL")
    await db.execute("PRAGMA foreign_keys=ON")
    await initialize_schema(db)

    today = date.today()

    # Events
    for i in range(5):
        d = (today - timedelta(days=i)).isoformat()
        await db.execute(
            "INSERT INTO events (topic_slug, date, summary, significance, raw_entities) "
            "VALUES (?, ?, ?, ?, ?)",
            ("ai-research", d, f"AI development #{i}: Major LLM breakthrough in reasoning capabilities", 7 + (i % 3), "[]"),
        )
    for i in range(3):
        d = (today - timedelta(days=i)).isoformat()
        await db.execute(
            "INSERT INTO events (topic_slug, date, summary, significance, raw_entities) "
            "VALUES (?, ?, ?, ?, ?)",
            ("geopolitics", d, f"Geopolitics #{i}: Trade negotiations shift dynamics", 6 + i, "[]"),
        )

    # Threads
    for tid, (slug, headline) in enumerate([
        ("ai-research", "Foundation Model Scaling Race"),
        ("ai-research", "AI Safety Regulation"),
        ("geopolitics", "US-China Trade Dynamics"),
    ], start=1):
        await db.execute(
            "INSERT INTO threads (id, headline, significance, status, slug, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (tid, headline, 8, "active", headline.lower().replace(" ", "-"), today.isoformat(), today.isoformat()),
        )
        await db.execute(
            "INSERT INTO thread_topics (thread_id, topic_slug) VALUES (?, ?)",
            (tid, slug),
        )

    # Entities
    for eid, name in enumerate(["OpenAI", "European Union", "China", "Google DeepMind"], start=1):
        await db.execute(
            "INSERT INTO entities (id, canonical_name, first_seen, last_seen) VALUES (?, ?, ?, ?)",
            (eid, name, today.isoformat(), today.isoformat()),
        )

    # Breaking alerts
    for i in range(3):
        await db.execute(
            "INSERT INTO breaking_alerts (headline_hash, headline, source_url, significance_score, topic_slug, alerted_at) "
            "VALUES (?, ?, ?, ?, ?, datetime('now'))",
            (f"demo-hash-{i}", f"Breaking: AI safety announcement #{i}", "https://example.com", 8, "ai-research"),
        )

    # Synthesis
    synthesis_data = json.dumps({
        "topic_name": "AI Research",
        "threads": [
            {"headline": "Foundation Model Scaling Race", "significance": 9},
            {"headline": "AI Safety Regulation", "significance": 8},
        ],
    })
    await db.execute(
        "INSERT INTO syntheses (topic_slug, date, data_json) VALUES (?, ?, ?)",
        ("ai-research", today.isoformat(), synthesis_data),
    )

    # Pipeline run
    await db.execute(
        "INSERT INTO pipeline_runs (status, started_at, completed_at) "
        "VALUES (?, datetime('now', '-1 hour'), datetime('now'))",
        ("completed",),
    )

    # Forecast run + questions
    await db.execute(
        "INSERT INTO forecast_runs (id, topic_slug, topic_name, engine, generated_for, summary, metadata_json) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (1, "ai-research", "AI Research", "structural", today.isoformat(), "Demo predictions", "{}"),
    )
    for qid, (question, prob) in enumerate([
        ("Will a major AI lab release a new frontier model within 30 days?", 0.42),
        ("Will the EU finalize AI Act implementation guidelines by Q2?", 0.68),
    ], start=1):
        res_date = (today + timedelta(days=30)).isoformat()
        await db.execute(
            "INSERT INTO forecast_questions "
            "(id, forecast_run_id, question, forecast_type, target_variable, probability, "
            "resolution_criteria, resolution_date, horizon_days, signpost, status, "
            "base_rate, target_metadata_json, signals_cited_json, evidence_event_ids_json, "
            "evidence_thread_ids_json, cross_topic_signal_ids_json, reasoning_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (qid, 1, question, "binary", "kg_native", prob,
             "Observable outcome", res_date, 30, "Key indicator", "open",
             0.5, "{}", "[]", "[]", "[]", "[]", "{}"),
        )
        await db.execute(
            "INSERT INTO forecast_resolutions (forecast_question_id, outcome_status) VALUES (?, 'pending')",
            (qid,),
        )

    # Config
    config = {
        "user": {"name": "Demo User", "timezone": "UTC"},
        "topics": [
            {"name": "AI Research", "priority": "high"},
            {"name": "Geopolitics", "priority": "medium"},
        ],
    }
    (dest_db.parent / "config.yaml").write_text(yaml.dump(config, default_flow_style=False))

    # Source registries
    for slug in ["ai-research", "geopolitics"]:
        src_dir = dest_db.parent / "sources" / slug
        src_dir.mkdir(parents=True, exist_ok=True)
        (src_dir / "registry.yaml").write_text(
            yaml.dump({"sources": [{"url": "https://example.com/rss", "id": f"demo-{slug}", "type": "rss"}]})
        )

    await db.commit()
    await db.close()
    return {"events": 8, "threads": 3, "entities": 4, "breaking_alerts": 3, "forecast_questions": 2, "syntheses": 1}


def run_demo_seed(data_dir: Path, from_scratch: bool = False):
    """Seed a demo database."""
    demo_dir = data_dir / "demo"
    demo_dir.mkdir(parents=True, exist_ok=True)
    dest_db = demo_dir / "knowledge.db"

    # Clean up any existing demo DB and WAL/SHM files
    for suffix in ("", "-wal", "-shm"):
        p = Path(str(dest_db) + suffix)
        if p.exists():
            p.unlink()

    if from_scratch:
        print("Seeding demo database from scratch...")
        stats = asyncio.run(_seed_from_scratch(dest_db))
    else:
        src_db = data_dir / "knowledge.db"
        if not src_db.exists():
            print(f"No live database at {src_db}. Use --from-scratch for static seed data.")
            return
        print(f"Seeding demo database from {src_db}...")
        stats = asyncio.run(_seed_from_live(src_db, dest_db, data_dir))

    print(f"\nDemo database created: {dest_db}")
    print(f"  Events:      {stats.get('events', 0)}")
    print(f"  Threads:     {stats.get('threads', 0)}")
    print(f"  Entities:    {stats.get('entities', 0)}")
    print(f"  Alerts:      {stats.get('breaking_alerts', 0)}")
    print(f"  Forecasts:   {stats.get('forecast_questions', 0)}")
    print(f"  Syntheses:   {stats.get('syntheses', 0)}")
    print(f"\nRun demo: NEXUS_DEMO_MODE=1 python -m nexus demo serve")


def run_demo_serve(data_dir: Path, host: str = "127.0.0.1", port: int = 8000):
    """Start the dashboard in demo mode with the demo database."""
    import os
    import uvicorn
    from nexus.web.app import create_app

    demo_dir = data_dir / "demo"
    db_path = demo_dir / "knowledge.db"

    if not db_path.exists():
        print("No demo database found. Run 'python -m nexus demo seed' first.")
        return

    os.environ["NEXUS_DEMO_MODE"] = "1"
    app = create_app(db_path=db_path, data_dir=demo_dir)
    print(f"Starting demo server at http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)
