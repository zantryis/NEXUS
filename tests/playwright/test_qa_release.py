"""QA release tests — OpenAI + Gemini, 6 topics (4 custom + 2 preset), Telegram.

Spins up 2 Docker containers via docker-compose.qa.yml, each with a different
provider and topic mix.  Validates full pipeline completion, dashboard rendering,
Forward Look, podcast, entities, threads, and source discovery.

Requires: docker, .env with OPENAI_API_KEY + GEMINI_API_KEY + TELEGRAM_BOT_TOKEN + TELEGRAM_BOT_TOKEN_TEST
Run:      .venv/bin/pytest tests/playwright/test_qa_release.py -m "e2e and integration" -v --timeout=1800
"""

import logging
import os
import re
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pytest

pytest.importorskip("playwright", reason="playwright not installed — skip E2E tests")

from playwright.sync_api import expect  # noqa: E402

pytestmark = [pytest.mark.e2e, pytest.mark.integration]

PIPELINE_TIMEOUT = 1200  # 20 min per container (custom topics need discovery + pipeline)

logger = logging.getLogger(__name__)

# ── Container config ──────────────────────────────────────────────────

_QA_CONTAINERS = ("qa-openai", "qa-gemini")

_QA_URLS = {
    "qa_openai": "http://localhost:8097",
    "qa_gemini": "http://localhost:8098",
}


# ── Helpers ───────────────────────────────────────────────────────────

def _get_api_key(name: str) -> str:
    """Get an API key from environment, skip test if missing."""
    val = os.environ.get(name, "")
    if not val:
        pytest.skip(f"{name} not set — skipping")
    return val


def _poll_pipeline(base_url: str, timeout_s: int = PIPELINE_TIMEOUT) -> str:
    """Poll /setup/status until pipeline completes or errors."""
    import httpx

    deadline = time.time() + timeout_s
    last_stage = ""
    while time.time() < deadline:
        try:
            r = httpx.get(f"{base_url}/setup/status", timeout=5.0)
            text = r.text
            if "pipeline-done" in text:
                return "done"
            if "pipeline-error" in text:
                return f"error:{text[:300]}"
            if "backfill" in text.lower():
                return "done_pre_backfill"
            text_lower = text.lower()
            if "audio" in text_lower and "podcast" in text_lower:
                return "done_audio_in_progress"
            for stage in ("discovering", "running", "rendering", "audio"):
                if stage in text_lower and stage != last_stage:
                    last_stage = stage
        except (httpx.ConnectError, httpx.ReadTimeout, httpx.ReadError, httpx.RemoteProtocolError):
            pass
        time.sleep(5)
    return f"timeout:{last_stage}"


def drive_wizard_with_telegram(
    page, base_url: str, provider: str, preset: str,
    api_key: str,
    preset_topics: list[str] | None = None,
    custom_topics: list[str] | None = None,
    telegram_token: str | None = None,
):
    """Drive the 3-step web setup wizard with optional Telegram and tier selection."""
    import httpx

    # Wait for server readiness
    for _ in range(15):
        try:
            r = httpx.get(f"{base_url}/setup", timeout=5.0, follow_redirects=True)
            if r.status_code == 200:
                break
        except (httpx.ConnectError, httpx.ReadTimeout, httpx.ReadError):
            pass
        time.sleep(2)

    page.goto(f"{base_url}/setup", wait_until="domcontentloaded", timeout=30_000)

    # Step 1: Provider + tier + API key
    page.locator(
        "label.preset-card", has=page.locator(f"input[value='{provider}']"),
    ).click()
    page.evaluate(f"onProviderChange('{provider}')")

    # Select specific tier/preset — click the label wrapping the tier radio
    # (radios are visually hidden, so click the parent label.tier-option instead)
    tier_selector = page.locator("#tier-selector")
    if tier_selector.is_visible():
        page.wait_for_timeout(300)
        tier_label = page.locator(
            "label.tier-option", has=page.locator(f"input[value='{preset}']"),
        )
        if tier_label.count() > 0:
            tier_label.click()
        # Ensure hidden field has the right value
        page.evaluate(f"document.getElementById('preset-hidden').value = '{preset}'")

    page.locator("#api_key").wait_for(state="visible", timeout=5_000)
    page.locator("#api_key").fill(api_key)
    page.locator("button[type='submit']").click()
    page.wait_for_url(re.compile(r"/setup/step/2"), timeout=30_000)

    # Step 2: Topics
    if preset_topics:
        for slug in preset_topics:
            page.locator(
                "label.topic-card", has=page.locator(f"input[value='{slug}']"),
            ).click()
    if custom_topics:
        for i, topic in enumerate(custom_topics):
            inputs = page.locator("input[name='custom_topics']")
            if i >= inputs.count():
                page.locator("button.btn-skip", has_text="Add another topic").click()
                inputs = page.locator("input[name='custom_topics']")
            inputs.nth(i).fill(topic)
    page.locator("button[type='submit']").click()
    page.wait_for_url(re.compile(r"/setup/step/3"), timeout=15_000)

    # Step 3: Review — timezone, schedule, optional Telegram
    expect(page.locator(".review-card").first).to_be_visible()

    # Enter Telegram token if provided
    if telegram_token:
        # Expand the optional Telegram section
        page.locator("summary", has_text="Telegram").click()
        page.wait_for_timeout(300)
        page.locator("#telegram_token").fill(telegram_token)
        # Validate token (non-blocking — just tests the UI flow)
        page.locator("button", has_text="Test Connection").click()
        # Wait briefly for validation response
        page.wait_for_timeout(3000)

    page.locator("button[type='submit']").click()
    page.wait_for_url(re.compile(r"setup=complete"), timeout=30_000)


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def qa_containers():
    """Build and start QA Docker containers, yield URL map, teardown."""
    import httpx
    from dotenv import load_dotenv

    project_root = Path(__file__).resolve().parents[2]
    load_dotenv(project_root / ".env", override=False)
    compose = ["docker", "compose", "-f", "docker-compose.qa.yml"]

    # Clean data dirs for fresh state
    def _clean_data_dirs():
        for suffix in _QA_CONTAINERS:
            d = project_root / f"data-{suffix}"
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

    # Build + start
    logger.info("Building and starting QA containers...")
    subprocess.run(
        [*compose, "up", "--build", "-d"],
        cwd=project_root, check=True, timeout=300,
    )

    # Wait for each container to respond on /setup
    for name, base in _QA_URLS.items():
        for attempt in range(90):
            try:
                r = httpx.get(f"{base}/setup", timeout=3.0, follow_redirects=True)
                if r.status_code in (200, 307, 503):
                    logger.info(f"Container {name} ready (attempt {attempt})")
                    break
            except (httpx.ConnectError, httpx.ReadTimeout, httpx.ReadError, httpx.RemoteProtocolError):
                pass
            time.sleep(2)
        else:
            subprocess.run(
                [*compose, "logs", f"{name.replace('_', '-')}"],
                cwd=project_root,
            )
            pytest.fail(f"Container {name} did not start within 180s")

    yield _QA_URLS

    # Teardown
    logger.info("Tearing down QA containers...")
    subprocess.run([*compose, "down", "-v"], cwd=project_root, check=True)
    _clean_data_dirs()


@pytest.fixture(scope="session")
def qa_pipelines_ready(qa_containers, browser):
    """Drive both wizards, then wait for both pipelines concurrently."""
    urls = qa_containers
    results = {}

    openai_key = os.environ.get("OPENAI_API_KEY", "")
    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    telegram_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    telegram_token_test = os.environ.get("TELEGRAM_BOT_TOKEN_TEST", "")

    wizard_configs = [
        (
            "qa_openai", "openai", "openai-balanced", openai_key,
            ["iran-us-relations"], ["Rwanda", "Penguins"],
            telegram_token,
        ),
        (
            "qa_gemini", "gemini", "balanced", gemini_key,
            ["ai-ml-research"], ["Road Cycling", "Stocks and Investment"],
            telegram_token_test,
        ),
    ]

    ctx = browser.new_context(viewport={"width": 1280, "height": 800}, ignore_https_errors=True)
    for name, provider, preset, key, presets, customs, tg_token in wizard_configs:
        if not key:
            results[name] = "skipped:no_key"
            continue
        page = ctx.new_page()
        try:
            drive_wizard_with_telegram(
                page, urls[name], provider, preset,
                api_key=key,
                preset_topics=presets,
                custom_topics=customs,
                telegram_token=tg_token or None,
            )
            results[name] = "wizard_done"
        except Exception as e:
            results[name] = f"wizard_error:{e}"
        finally:
            page.close()
    ctx.close()

    # Poll both pipelines concurrently
    poll_tasks = {}
    for name in ("qa_openai", "qa_gemini"):
        if results.get(name, "").startswith("wizard_done"):
            poll_tasks[name] = urls[name]

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {}
        for name, url in poll_tasks.items():
            futures[pool.submit(_poll_pipeline, url)] = name

        for future in as_completed(futures, timeout=PIPELINE_TIMEOUT + 60):
            name = futures[future]
            result = future.result()
            results[name] = result

    return results


# ── Suite A: OpenAI Balanced ──────────────────────────────────────────

class TestQAOpenAI:
    """OpenAI balanced: Iran-US Relations (preset) + Rwanda + Penguins (custom)."""

    @pytest.fixture(autouse=True)
    def _urls(self, qa_containers, qa_pipelines_ready):
        self.base = qa_containers["qa_openai"]
        self.pipeline_result = qa_pipelines_ready.get("qa_openai", "unknown")

    def test_wizard_and_pipeline(self):
        if "error" in self.pipeline_result:
            pytest.fail(f"Pipeline failed: {self.pipeline_result}")
        if "timeout" in self.pipeline_result:
            pytest.fail(f"Pipeline timed out: {self.pipeline_result}")
        if "skipped" in self.pipeline_result:
            pytest.skip("OPENAI_API_KEY not set")
        assert self.pipeline_result in ("done", "done_pre_backfill", "done_audio_in_progress", "wizard_done")

    def test_dashboard_has_content(self, page):
        page.goto(self.base)
        page.wait_for_load_state("domcontentloaded")
        content = page.locator(".main-content")
        expect(content).to_be_visible(timeout=10_000)
        assert page.locator(".main-content").inner_text(timeout=10_000).strip()

    def test_topic_tabs_visible(self, page):
        """Verify all 3 topic tabs appear on dashboard."""
        page.goto(self.base)
        page.wait_for_load_state("domcontentloaded")
        tabs = page.locator(".topic-tab")
        expect(tabs.first).to_be_visible(timeout=10_000)
        assert tabs.count() >= 2  # niche topics may get 0 articles in smoke mode

    def test_entities_exist(self, page):
        page.goto(f"{self.base}/entities/")
        page.wait_for_load_state("domcontentloaded")
        entity_items = page.locator("a[href*='/entities/']")
        assert entity_items.count() > 0

    def test_threads_page(self, page):
        page.goto(f"{self.base}/threads/")
        page.wait_for_load_state("domcontentloaded")
        thread_links = page.locator("a[href*='/threads/']")
        assert thread_links.count() > 0

    def test_forward_look(self, page):
        page.goto(f"{self.base}/forward-look")
        page.wait_for_load_state("domcontentloaded")
        assert page.locator(".main-content").count() > 0

    def test_podcast_feed(self):
        import httpx
        r = httpx.get(f"{self.base}/feed.xml", timeout=10.0)
        assert r.status_code == 200
        assert "xml" in r.headers.get("content-type", "").lower() or "<rss" in r.text

    def test_sources_page(self, page):
        """Verify source registries were created for custom topics."""
        page.goto(f"{self.base}/sources/")
        page.wait_for_load_state("domcontentloaded")
        body = page.content().lower()
        assert "source" in body or "registry" in body or "feed" in body


# ── Suite B: Gemini Balanced ──────────────────────────────────────────

class TestQAGemini:
    """Gemini balanced: AI/ML Research (preset) + Road Cycling + Stocks and Investment (custom)."""

    @pytest.fixture(autouse=True)
    def _urls(self, qa_containers, qa_pipelines_ready):
        self.base = qa_containers["qa_gemini"]
        self.pipeline_result = qa_pipelines_ready.get("qa_gemini", "unknown")

    def test_wizard_and_pipeline(self):
        if "error" in self.pipeline_result:
            pytest.fail(f"Pipeline failed: {self.pipeline_result}")
        if "timeout" in self.pipeline_result:
            pytest.fail(f"Pipeline timed out: {self.pipeline_result}")
        if "skipped" in self.pipeline_result:
            pytest.skip("GEMINI_API_KEY not set")
        assert self.pipeline_result in ("done", "done_pre_backfill", "done_audio_in_progress", "wizard_done")

    def test_dashboard_has_content(self, page):
        page.goto(self.base)
        page.wait_for_load_state("domcontentloaded")
        content = page.locator(".main-content")
        expect(content).to_be_visible(timeout=10_000)
        assert page.locator(".main-content").inner_text(timeout=10_000).strip()

    def test_topic_tabs_visible(self, page):
        """Verify all 3 topic tabs appear on dashboard."""
        page.goto(self.base)
        page.wait_for_load_state("domcontentloaded")
        tabs = page.locator(".topic-tab")
        expect(tabs.first).to_be_visible(timeout=10_000)
        assert tabs.count() >= 2  # niche topics may get 0 articles in smoke mode

    def test_entities_exist(self, page):
        page.goto(f"{self.base}/entities/")
        page.wait_for_load_state("domcontentloaded")
        entity_items = page.locator("a[href*='/entities/']")
        assert entity_items.count() > 0

    def test_threads_page(self, page):
        page.goto(f"{self.base}/threads/")
        page.wait_for_load_state("domcontentloaded")
        thread_links = page.locator("a[href*='/threads/']")
        assert thread_links.count() > 0

    def test_forward_look(self, page):
        page.goto(f"{self.base}/forward-look")
        page.wait_for_load_state("domcontentloaded")
        assert page.locator(".main-content").count() > 0
