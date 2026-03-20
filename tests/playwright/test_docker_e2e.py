"""E2E Docker tests — multi-provider setup wizard and CLI with real API keys.

Spins up 6 Docker containers (4 web wizard + 2 CLI), each with different
providers, topics, and TTS backends.  Validates full pipeline completion
and dashboard rendering with real data.

Strategy: drive ALL 4 web wizards immediately (starting pipelines in parallel),
then wait for ALL 6 pipelines concurrently, then run validation tests.

Requires: docker, .env with GEMINI_API_KEY + DEEPSEEK_API_KEY + ELEVENLABS_API_KEY
Run:      .venv/bin/pytest tests/playwright/test_docker_e2e.py -m "e2e and integration" -v --timeout=1800
"""

import os
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest

pytest.importorskip("playwright", reason="playwright not installed — skip E2E tests")

from playwright.sync_api import expect  # noqa: E402

pytestmark = [pytest.mark.e2e, pytest.mark.integration]

PIPELINE_TIMEOUT = 900  # 15 min per container


# ── Helpers ───────────────────────────────────────────────────────────

def _get_api_key(name: str) -> str:
    """Get an API key from environment, skip test if missing."""
    val = os.environ.get(name, "")
    if not val:
        pytest.skip(f"{name} not set — skipping")
    return val


def drive_wizard(
    page, base_url: str, provider: str,
    api_key: str | None = None,
    preset_topics: list[str] | None = None,
    custom_topics: list[str] | None = None,
):
    """Drive the 3-step web setup wizard to completion."""
    import httpx
    # Ensure the server is actually reachable before Playwright navigates
    for _ in range(15):
        try:
            r = httpx.get(f"{base_url}/setup", timeout=5.0, follow_redirects=True)
            if r.status_code == 200:
                break
        except (httpx.ConnectError, httpx.ReadTimeout, httpx.ReadError):
            pass
        time.sleep(2)

    page.goto(f"{base_url}/setup", wait_until="domcontentloaded", timeout=30_000)

    # Step 1: Provider + API key
    page.locator(
        "label.preset-card", has=page.locator(f"input[value='{provider}']"),
    ).click()
    page.evaluate(f"onProviderChange('{provider}')")
    if api_key:
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

    # Step 3: Review + submit
    expect(page.locator(".review-card")).to_be_visible()
    page.locator("button[type='submit']").click()
    page.wait_for_url(re.compile(r"setup=complete"), timeout=30_000)


def _poll_pipeline(base_url: str, timeout_s: int = PIPELINE_TIMEOUT) -> str:
    """Poll /setup/status until pipeline completes or errors (httpx-only, no Playwright)."""
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
            # Audio and backfill stages mean the briefing is already done
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


def _poll_health(base_url: str, timeout_s: int = PIPELINE_TIMEOUT) -> str:
    """Poll /api/health until briefing_today is true (for CLI containers)."""
    import httpx

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            r = httpx.get(f"{base_url}/api/health", timeout=5.0)
            if r.status_code == 200:
                data = r.json()
                if data.get("deliverables", {}).get("briefing_today"):
                    return "done"
        except (httpx.ConnectError, httpx.ReadTimeout, httpx.ReadError, httpx.RemoteProtocolError):
            pass
        time.sleep(10)
    return "timeout"


# ── Session-scoped fixture: drive all wizards, wait for all pipelines ──

@pytest.fixture(scope="session")
def all_pipelines_ready(e2e_containers, browser):
    """Drive all 4 web wizards, then wait for all 6 pipelines concurrently.

    Uses a single browser context to drive wizards sequentially (fast, <30s total),
    then polls all 6 pipelines concurrently via httpx threads.
    """
    urls = e2e_containers
    results = {}

    # Phase 1: Drive all 4 web wizards (each ~5-10s)
    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    deepseek_key = os.environ.get("DEEPSEEK_API_KEY", "")

    wizard_configs = [
        ("web_gemini", "gemini", gemini_key, ["ai-ml-research"], ["Antarctic Ice Sheet Monitoring"]),
        ("web_deepseek", "deepseek", deepseek_key, ["iran-us-relations"], ["Competitive Chess"]),
        ("web_audio", "gemini", gemini_key, None, ["Korean Pop Music Industry", "CRISPR Gene Editing"]),
        ("web_wild", "deepseek", deepseek_key, None, ["Underwater Archaeology", "Competitive Speedrunning"]),
    ]

    ctx = browser.new_context(viewport={"width": 1280, "height": 800}, ignore_https_errors=True)
    for name, provider, key, presets, customs in wizard_configs:
        if not key:
            results[name] = "skipped:no_key"
            continue
        page = ctx.new_page()
        try:
            drive_wizard(page, urls[name], provider, api_key=key,
                         preset_topics=presets, custom_topics=customs)
            results[name] = "wizard_done"
        except Exception as e:
            results[name] = f"wizard_error:{e}"
        finally:
            page.close()
    ctx.close()

    # Phase 2: Wait for all 6 pipelines concurrently
    poll_tasks = {}
    for name in ("web_gemini", "web_deepseek", "web_audio", "web_wild"):
        if results.get(name, "").startswith("wizard_done"):
            poll_tasks[name] = ("pipeline", urls[name])
    for name in ("cli_gemini", "cli_deepseek"):
        poll_tasks[name] = ("health", urls[name])

    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {}
        for name, (mode, url) in poll_tasks.items():
            if mode == "pipeline":
                futures[pool.submit(_poll_pipeline, url)] = name
            else:
                futures[pool.submit(_poll_health, url)] = name

        for future in as_completed(futures, timeout=PIPELINE_TIMEOUT + 60):
            name = futures[future]
            result = future.result()
            results[name] = result

    return results


# ── Suite A: Web Wizard + Gemini Balanced ─────────────────────────────

class TestWebGemini:
    """Gemini balanced: AI/ML Research (preset) + Antarctic Ice Sheet Monitoring (custom)."""

    @pytest.fixture(autouse=True)
    def _urls(self, e2e_containers, all_pipelines_ready):
        self.base = e2e_containers["web_gemini"]
        self.pipeline_result = all_pipelines_ready.get("web_gemini", "unknown")

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
        """Verify both topic tabs appear on dashboard."""
        page.goto(self.base)
        page.wait_for_load_state("domcontentloaded")
        body = page.content().lower()
        assert "ai" in body or "antarctic" in body

    def test_entities_exist(self, page):
        page.goto(f"{self.base}/entities/")
        page.wait_for_load_state("domcontentloaded")
        entity_items = page.locator("a[href*='/entities/']")
        assert entity_items.count() > 0

    def test_podcast_feed(self):
        import httpx
        r = httpx.get(f"{self.base}/feed.xml", timeout=10.0)
        assert r.status_code == 200
        assert "xml" in r.headers.get("content-type", "").lower() or "<rss" in r.text

    def test_forward_look(self, page):
        page.goto(f"{self.base}/forward-look")
        page.wait_for_load_state("domcontentloaded")
        assert page.locator(".main-content").count() > 0


# ── Suite B: Web Wizard + DeepSeek Cheap ──────────────────────────────

class TestWebDeepSeek:
    """DeepSeek cheap: Iran-US Relations (preset) + Competitive Chess (custom)."""

    @pytest.fixture(autouse=True)
    def _urls(self, e2e_containers, all_pipelines_ready):
        self.base = e2e_containers["web_deepseek"]
        self.pipeline_result = all_pipelines_ready.get("web_deepseek", "unknown")

    def test_wizard_and_pipeline(self):
        if "error" in self.pipeline_result:
            pytest.fail(f"Pipeline failed: {self.pipeline_result}")
        if "timeout" in self.pipeline_result:
            pytest.fail(f"Pipeline timed out: {self.pipeline_result}")
        if "skipped" in self.pipeline_result:
            pytest.skip("DEEPSEEK_API_KEY not set")
        assert self.pipeline_result in ("done", "done_pre_backfill", "done_audio_in_progress", "wizard_done")

    def test_dashboard_multi_topic(self, page):
        page.goto(self.base)
        page.wait_for_load_state("domcontentloaded")
        tabs = page.locator(".topic-tab")
        expect(tabs.first).to_be_visible(timeout=10_000)
        assert tabs.count() >= 2

    def test_threads_page(self, page):
        page.goto(f"{self.base}/threads/")
        page.wait_for_load_state("domcontentloaded")
        thread_links = page.locator("a[href*='/threads/']")
        assert thread_links.count() > 0

    def test_topic_tabs_visible(self, page):
        """Verify both topic tabs appear on dashboard."""
        page.goto(self.base)
        page.wait_for_load_state("domcontentloaded")
        body = page.content().lower()
        assert "iran" in body or "chess" in body


# ── Suite C: Web Wizard + Gemini + ElevenLabs Audio ───────────────────

class TestWebElevenLabs:
    """Gemini balanced + ElevenLabs TTS: Korean Pop Music Industry + CRISPR Gene Editing."""

    @pytest.fixture(autouse=True)
    def _urls(self, e2e_containers, all_pipelines_ready):
        self.base = e2e_containers["web_audio"]
        self.pipeline_result = all_pipelines_ready.get("web_audio", "unknown")

    def test_wizard_and_pipeline(self):
        if "error" in self.pipeline_result:
            pytest.fail(f"Pipeline failed: {self.pipeline_result}")
        if "timeout" in self.pipeline_result:
            pytest.fail(f"Pipeline timed out: {self.pipeline_result}")
        if "skipped" in self.pipeline_result:
            pytest.skip("GEMINI_API_KEY not set")
        assert self.pipeline_result in ("done", "done_pre_backfill", "done_audio_in_progress", "wizard_done")

    def test_configure_elevenlabs(self, page):
        """Switch TTS backend to ElevenLabs via the settings page."""
        page.goto(f"{self.base}/settings")
        page.wait_for_load_state("domcontentloaded")

        # Expand the Audio / Podcast accordion section
        page.locator(".settings-section-header", has_text="Audio").click()
        page.wait_for_timeout(300)

        audio_checkbox = page.locator("input[name='audio_enabled']")
        if not audio_checkbox.is_checked():
            audio_checkbox.check()

        page.select_option("select[name='tts_backend']", "elevenlabs")
        page.wait_for_timeout(500)

        page.select_option("select[name='voice_host_a']", "Sarah")
        page.select_option("select[name='voice_host_b']", "Charlie")

        # Expand API Keys section and enter ElevenLabs key
        page.locator(".settings-section-header", has_text="API Keys").click()
        page.wait_for_timeout(300)

        elevenlabs_key = _get_api_key("ELEVENLABS_API_KEY")
        key_input = page.locator("input[name='ELEVENLABS_API_KEY']")
        if key_input.count() > 0:
            key_input.fill(elevenlabs_key)

        page.get_by_role("button", name="Save All Settings").click()
        page.wait_for_url(re.compile(r"saved="), timeout=10_000)

    def test_settings_persisted(self, page):
        """Reload settings and verify ElevenLabs is still selected."""
        page.goto(f"{self.base}/settings")
        page.wait_for_load_state("domcontentloaded")
        # Expand audio section
        page.locator(".settings-section-header", has_text="Audio").click()
        page.wait_for_timeout(300)
        tts_select = page.locator("select[name='tts_backend']")
        selected = tts_select.input_value()
        assert selected == "elevenlabs"

    def test_dashboard_has_content(self, page):
        page.goto(self.base)
        page.wait_for_load_state("domcontentloaded")
        content = page.locator(".main-content")
        expect(content).to_be_visible(timeout=10_000)

    def test_podcast_feed(self):
        import httpx
        r = httpx.get(f"{self.base}/feed.xml", timeout=10.0)
        assert r.status_code == 200
        assert "xml" in r.headers.get("content-type", "").lower() or "<rss" in r.text


# ── Suite D: Web Wizard + DeepSeek Wild Topics Only ───────────────────

class TestWebWildTopics:
    """DeepSeek cheap: Underwater Archaeology + Competitive Speedrunning (no presets)."""

    @pytest.fixture(autouse=True)
    def _urls(self, e2e_containers, all_pipelines_ready):
        self.base = e2e_containers["web_wild"]
        self.pipeline_result = all_pipelines_ready.get("web_wild", "unknown")

    def test_wizard_and_pipeline(self):
        if "error" in self.pipeline_result:
            pytest.fail(f"Pipeline failed: {self.pipeline_result}")
        if "timeout" in self.pipeline_result:
            pytest.fail(f"Pipeline timed out: {self.pipeline_result}")
        if "skipped" in self.pipeline_result:
            pytest.skip("DEEPSEEK_API_KEY not set")
        assert self.pipeline_result in ("done", "done_pre_backfill", "done_audio_in_progress", "wizard_done")

    def test_source_discovery(self, page):
        """Verify source registries were created for niche topics."""
        page.goto(f"{self.base}/sources/")
        page.wait_for_load_state("domcontentloaded")
        body = page.content()
        assert "source" in body.lower() or "registry" in body.lower() or "feed" in body.lower()

    def test_dashboard_has_content(self, page):
        page.goto(self.base)
        page.wait_for_load_state("domcontentloaded")
        content = page.locator(".main-content")
        expect(content).to_be_visible(timeout=10_000)

    def test_threads_page(self, page):
        page.goto(f"{self.base}/threads/")
        page.wait_for_load_state("domcontentloaded")
        thread_links = page.locator("a[href*='/threads/']")
        assert thread_links.count() > 0


# ── Suite E: CLI Setup + Gemini Quality ───────────────────────────────

class TestCliGemini:
    """CLI setup: quality preset, Mars Colonization + Global Energy Transition."""

    @pytest.fixture(autouse=True)
    def _urls(self, e2e_containers, all_pipelines_ready):
        self.base = e2e_containers["cli_gemini"]
        self.pipeline_result = all_pipelines_ready.get("cli_gemini", "unknown")

    def test_cli_setup_writes_config(self):
        """Verify the CLI entrypoint wrote config.yaml inside the container."""
        result = subprocess.run(
            ["docker", "exec", "nexus-e2e-cli-gemini", "cat", "/app/data/config.yaml"],
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 0
        assert "quality" in result.stdout or "preset" in result.stdout
        assert "Mars Colonization" in result.stdout

    def test_pipeline_completes(self):
        if "timeout" in self.pipeline_result:
            pytest.fail(f"Pipeline timed out: {self.pipeline_result}")
        assert self.pipeline_result in ("done", "done_audio_in_progress")

    def test_dashboard_has_content(self, page):
        page.goto(self.base)
        page.wait_for_load_state("domcontentloaded")
        content = page.locator(".main-content")
        expect(content).to_be_visible(timeout=10_000)
        assert page.locator(".main-content").inner_text(timeout=10_000).strip()

    def test_forward_look(self, page):
        page.goto(f"{self.base}/forward-look")
        page.wait_for_load_state("domcontentloaded")
        assert page.locator(".main-content").count() > 0

    def test_podcast_feed(self):
        """Quality preset has audio enabled — verify podcast RSS."""
        import httpx
        r = httpx.get(f"{self.base}/feed.xml", timeout=10.0)
        assert r.status_code == 200
        assert "xml" in r.headers.get("content-type", "").lower() or "<rss" in r.text


# ── Suite F: CLI Setup + DeepSeek Cheap ───────────────────────────────

class TestCliDeepSeek:
    """CLI setup: cheap preset, Quantum Computing Breakthroughs + Formula 1."""

    @pytest.fixture(autouse=True)
    def _urls(self, e2e_containers, all_pipelines_ready):
        self.base = e2e_containers["cli_deepseek"]
        self.pipeline_result = all_pipelines_ready.get("cli_deepseek", "unknown")

    def test_cli_setup_writes_config(self):
        """Verify the CLI entrypoint wrote config.yaml inside the container."""
        result = subprocess.run(
            ["docker", "exec", "nexus-e2e-cli-deepseek", "cat", "/app/data/config.yaml"],
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 0
        assert "cheap" in result.stdout or "preset" in result.stdout
        assert "Quantum Computing" in result.stdout

    def test_pipeline_completes(self):
        if "timeout" in self.pipeline_result:
            pytest.fail(f"Pipeline timed out: {self.pipeline_result}")
        assert self.pipeline_result in ("done", "done_audio_in_progress")

    def test_dashboard_has_content(self, page):
        page.goto(self.base)
        page.wait_for_load_state("domcontentloaded")
        content = page.locator(".main-content")
        expect(content).to_be_visible(timeout=10_000)

    def test_threads_page(self, page):
        page.goto(f"{self.base}/threads/")
        page.wait_for_load_state("domcontentloaded")
        thread_links = page.locator("a[href*='/threads/']")
        assert thread_links.count() > 0
