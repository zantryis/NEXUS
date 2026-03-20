"""Playwright E2E test fixtures — demo server with seeded database."""

import asyncio
import json
import logging
import os
import shutil
import socket
import subprocess
import threading
import time
from datetime import date, timedelta
from pathlib import Path

import pytest
import yaml

logger = logging.getLogger(__name__)

pytest.importorskip("playwright", reason="playwright not installed — skip E2E tests")

# Mark all tests in this directory as e2e
pytestmark = pytest.mark.e2e


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _seed_database(db_path: Path) -> None:
    """Seed a knowledge.db with representative data for E2E testing."""

    async def _seed():
        import aiosqlite
        from nexus.engine.knowledge.schema import initialize_schema

        db = await aiosqlite.connect(str(db_path))
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
                ("ai-research", d, f"AI development #{i}: Major LLM breakthrough", 7 + (i % 3), "[]"),
            )
        for i in range(3):
            d = (today - timedelta(days=i)).isoformat()
            await db.execute(
                "INSERT INTO events (topic_slug, date, summary, significance, raw_entities) "
                "VALUES (?, ?, ?, ?, ?)",
                ("energy", d, f"Energy shift #{i}: OPEC+ production decision", 6 + i, "[]"),
            )

        # Threads
        for tid, (slug, headline) in enumerate([
            ("ai-research", "GPT-5 Development Race"),
            ("ai-research", "AI Regulation in the EU"),
            ("energy", "OPEC+ Production Cuts"),
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
        for eid, name in enumerate(["OpenAI", "EU", "OPEC", "Google DeepMind"], start=1):
            await db.execute(
                "INSERT INTO entities (id, canonical_name, first_seen, last_seen) VALUES (?, ?, ?, ?)",
                (eid, name, today.isoformat(), today.isoformat()),
            )

        # Breaking alerts
        for i in range(3):
            await db.execute(
                "INSERT INTO breaking_alerts (headline_hash, headline, source_url, significance_score, topic_slug, alerted_at) "
                "VALUES (?, ?, ?, ?, ?, datetime('now'))",
                (f"hash-{i}", f"Breaking: AI Safety announcement #{i}", "https://example.com", 8, "ai-research"),
            )

        # Forecast run + questions
        await db.execute(
            "INSERT INTO forecast_runs (id, topic_slug, topic_name, engine, generated_for, summary, metadata_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (1, "ai-research", "AI Research", "structural", today.isoformat(), "AI predictions", "{}"),
        )
        for qid, (question, prob, ext_ref) in enumerate([
            ("Will OpenAI release GPT-5 within 30 days?", 0.35, None),
            ("Will EU pass AI Act amendments by Q2?", 0.72, "KXEUAI-26Q2"),
        ], start=1):
            res_date = (today + timedelta(days=30)).isoformat()
            await db.execute(
                "INSERT INTO forecast_questions "
                "(id, forecast_run_id, question, forecast_type, target_variable, probability, "
                "resolution_criteria, resolution_date, horizon_days, signpost, status, external_ref, "
                "base_rate, target_metadata_json, signals_cited_json, evidence_event_ids_json, "
                "evidence_thread_ids_json, cross_topic_signal_ids_json, reasoning_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (qid, 1, question, "binary", "kg_native", prob,
                 "observable outcome", res_date, 14, "Key indicator", "open", ext_ref,
                 0.5, "{}", "[]", "[]", "[]", "[]", "{}"),
            )
            await db.execute(
                "INSERT INTO forecast_resolutions (forecast_question_id, outcome_status, external_ref) "
                "VALUES (?, 'pending', ?)",
                (qid, ext_ref),
            )

        # Synthesis
        synthesis_data = json.dumps({
            "topic_name": "AI Research",
            "threads": [
                {"headline": "GPT-5 Development Race", "significance": 9},
                {"headline": "AI Regulation in the EU", "significance": 8},
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

        await db.commit()
        await db.close()

    asyncio.run(_seed())


@pytest.fixture(scope="session")
def demo_dir(tmp_path_factory):
    """Create a temporary data directory with seeded demo database."""
    data_dir = tmp_path_factory.mktemp("demo_data")
    db_path = data_dir / "knowledge.db"
    _seed_database(db_path)

    config = {
        "user": {"name": "Demo User", "timezone": "UTC"},
        "topics": [
            {"name": "AI Research", "priority": "high"},
            {"name": "Energy", "priority": "medium"},
        ],
    }
    (data_dir / "config.yaml").write_text(yaml.dump(config))

    for slug in ["ai-research", "energy"]:
        src_dir = data_dir / "sources" / slug
        src_dir.mkdir(parents=True)
        (src_dir / "registry.yaml").write_text(
            yaml.dump({"sources": [{"url": "https://example.com/rss", "id": f"test-{slug}", "type": "rss"}]})
        )

    return data_dir


@pytest.fixture(scope="session")
def demo_server(demo_dir):
    """Start a uvicorn server in demo mode on a random port."""
    import uvicorn

    port = _find_free_port()
    os.environ["NEXUS_DEMO_MODE"] = "1"

    from nexus.web.app import create_app
    app = create_app(db_path=demo_dir / "knowledge.db", data_dir=demo_dir)

    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)

    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    # Wait for server to be ready
    import httpx
    base_url = f"http://127.0.0.1:{port}"
    for _ in range(50):
        try:
            resp = httpx.get(f"{base_url}/", timeout=2.0, follow_redirects=True)
            if resp.status_code in (200, 302, 307):
                break
        except (httpx.ConnectError, httpx.ReadTimeout):
            pass
        time.sleep(0.2)
    else:
        raise RuntimeError(f"Demo server failed to start on port {port}")

    yield base_url

    os.environ.pop("NEXUS_DEMO_MODE", None)
    server.should_exit = True
    thread.join(timeout=5)


@pytest.fixture(scope="session")
def setup_dir(tmp_path_factory):
    """Create a temporary data directory with NO config.yaml (fresh install state)."""
    data_dir = tmp_path_factory.mktemp("setup_data")
    # Touch an empty knowledge.db — wizard doesn't need data, just needs the file
    (data_dir / "knowledge.db").touch()
    return data_dir


@pytest.fixture(scope="session")
def setup_server(setup_dir):
    """Start a server with no config — triggers setup wizard redirect."""
    import uvicorn

    port = _find_free_port()
    os.environ["NEXUS_SKIP_AUTO_PIPELINE"] = "1"
    # Ensure demo mode is off — demo_server may have set it in this process
    saved_demo = os.environ.pop("NEXUS_DEMO_MODE", None)

    from nexus.web.app import create_app
    app = create_app(db_path=setup_dir / "knowledge.db", data_dir=setup_dir)

    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)

    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    import httpx
    base_url = f"http://127.0.0.1:{port}"
    for _ in range(50):
        try:
            resp = httpx.get(f"{base_url}/setup", timeout=2.0, follow_redirects=True)
            if resp.status_code in (200, 302, 307):
                break
        except (httpx.ConnectError, httpx.ReadTimeout):
            pass
        time.sleep(0.2)
    else:
        raise RuntimeError(f"Setup server failed to start on port {port}")

    yield base_url

    os.environ.pop("NEXUS_SKIP_AUTO_PIPELINE", None)
    if saved_demo is not None:
        os.environ["NEXUS_DEMO_MODE"] = saved_demo
    server.should_exit = True
    thread.join(timeout=5)


@pytest.fixture()
def clean_setup(setup_dir):
    """Ensure no config.yaml exists before each setup wizard test."""
    config_path = setup_dir / "config.yaml"
    config_path.unlink(missing_ok=True)
    yield setup_dir
    config_path.unlink(missing_ok=True)


@pytest.fixture(scope="session")
def browser_context_args():
    """Configure Playwright browser context."""
    return {
        "viewport": {"width": 1280, "height": 800},
        "ignore_https_errors": True,
    }


# ── Docker-based E2E fixtures ────────────────────────────────────────

_E2E_CONTAINERS = (
    "web-gemini", "web-deepseek", "web-audio", "web-wild",
    "cli-gemini", "cli-deepseek",
)

_E2E_URLS = {
    "web_gemini": "http://localhost:8091",
    "web_deepseek": "http://localhost:8092",
    "web_audio": "http://localhost:8093",
    "web_wild": "http://localhost:8094",
    "cli_gemini": "http://localhost:8095",
    "cli_deepseek": "http://localhost:8096",
}


@pytest.fixture(scope="session")
def e2e_containers():
    """Build and start E2E Docker containers, yield URL map, teardown.

    Web containers start with no config.yaml (setup wizard triggers).
    CLI containers pipe stdin into `nexus setup` then start the server.
    """
    import httpx
    from dotenv import load_dotenv

    project_root = Path(__file__).resolve().parents[2]

    # Load .env so API keys are available for Playwright-driven wizard steps
    load_dotenv(project_root / ".env", override=False)
    compose = ["docker", "compose", "-f", "docker-compose.e2e.yml"]

    # Clean data dirs to ensure fresh state (Docker creates files as root)
    def _clean_data_dirs():
        for suffix in _E2E_CONTAINERS:
            d = project_root / f"data-e2e-{suffix}"
            if d.exists():
                try:
                    shutil.rmtree(d)
                except PermissionError:
                    subprocess.run(
                        ["docker", "run", "--rm", "-v", f"{d}:/cleanup", "alpine",
                         "sh", "-c", "rm -rf /cleanup/*"],
                        check=False,
                    )
                    shutil.rmtree(d, ignore_errors=True)
            d.mkdir(exist_ok=True)

    _clean_data_dirs()

    # Build image (shared) + start all 6 containers
    logger.info("Building and starting E2E containers...")
    subprocess.run(
        [*compose, "up", "--build", "-d"],
        cwd=project_root, check=True, timeout=300,
    )

    # Wait for each container to respond
    for name, base in _E2E_URLS.items():
        # CLI containers should respond on /api/health once setup + server start
        # Web containers redirect to /setup when no config exists
        endpoint = "/api/health" if "cli" in name else "/setup"
        for attempt in range(90):  # up to 180s for CLI containers (setup takes time)
            try:
                r = httpx.get(
                    f"{base}{endpoint}", timeout=3.0, follow_redirects=True,
                )
                if r.status_code in (200, 307, 503):
                    logger.info(f"Container {name} ready (attempt {attempt})")
                    break
            except (httpx.ConnectError, httpx.ReadTimeout, httpx.ReadError, httpx.RemoteProtocolError):
                pass
            time.sleep(2)
        else:
            # Dump logs for the failed container before failing
            subprocess.run(
                [*compose, "logs", f"e2e-{name.replace('_', '-')}"],
                cwd=project_root,
            )
            pytest.fail(f"Container {name} did not start within 180s")

    yield _E2E_URLS

    # Teardown
    logger.info("Tearing down E2E containers...")
    subprocess.run([*compose, "down", "-v"], cwd=project_root, check=True)
    _clean_data_dirs()
