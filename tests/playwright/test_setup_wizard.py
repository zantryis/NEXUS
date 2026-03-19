"""Playwright E2E tests for the setup wizard (fresh install flow)."""

import re

import pytest
from playwright.sync_api import expect

pytestmark = [pytest.mark.e2e]


def _select_ollama(page):
    """Click the Ollama provider card (radio inputs are CSS-hidden behind labels)."""
    page.locator("label.preset-card", has=page.locator("input[value='ollama']")).click()


def _select_gemini(page):
    """Click the Gemini provider card."""
    page.locator("label.preset-card", has=page.locator("input[value='gemini']")).click()


def _walk_to_step2(page, base_url):
    """Navigate from start through step 1 (Ollama) to step 2."""
    page.goto(f"{base_url}/setup")
    _select_ollama(page)
    page.locator("button[type='submit']").click()
    page.wait_for_url(re.compile(r"/setup/step/2"))


def _walk_to_step3(page, base_url):
    """Navigate through steps 1-2 to step 3 (review)."""
    _walk_to_step2(page, base_url)
    page.locator("input[name='custom_topics']").first.fill("Test Topic")
    page.locator("button[type='submit']").click()
    page.wait_for_url(re.compile(r"/setup/step/3"))


def test_fresh_install_redirects_to_setup(page, setup_server, clean_setup):
    """Visiting / with no config should redirect to /setup."""
    page.goto(setup_server)
    assert "/setup" in page.url


def test_setup_step1_renders(page, setup_server, clean_setup):
    """Step 1 should show provider selection cards."""
    page.goto(f"{setup_server}/setup")
    # Preset cards should be visible (radio inputs are hidden behind them)
    expect(page.locator(".preset-card").first).to_be_visible()
    # Submit button present
    expect(page.locator("button[type='submit']")).to_be_visible()


def test_setup_step1_provider_selection(page, setup_server, clean_setup):
    """Selecting Ollama (no key needed) and submitting should advance to step 2."""
    page.goto(f"{setup_server}/setup")
    _select_ollama(page)
    page.locator("button[type='submit']").click()
    page.wait_for_url(re.compile(r"/setup/step/2"))


def test_setup_step1_requires_api_key(page, setup_server, clean_setup):
    """Selecting a provider with required key but no key should show error."""
    page.goto(f"{setup_server}/setup")
    _select_gemini(page)
    page.locator("button[type='submit']").click()
    # Should stay on step 1 with error
    expect(page.locator(".alert-error")).to_be_visible()


def test_setup_step2_topics_render(page, setup_server, clean_setup):
    """Step 2 should show topic selection UI."""
    _walk_to_step2(page, setup_server)
    # Topic cards should be visible
    expect(page.locator(".topic-card").first).to_be_visible()
    # Custom topic input
    expect(page.locator("input[name='custom_topics']").first).to_be_visible()


def test_setup_step2_to_step3(page, setup_server, clean_setup):
    """Selecting a topic and submitting should advance to step 3."""
    _walk_to_step2(page, setup_server)
    page.locator("input[name='custom_topics']").first.fill("Test Topic")
    page.locator("button[type='submit']").click()
    page.wait_for_url(re.compile(r"/setup/step/3"))


def test_setup_step3_review(page, setup_server, clean_setup):
    """Step 3 should display summary of chosen provider + topics."""
    _walk_to_step3(page, setup_server)
    expect(page.locator(".review-card")).to_be_visible()
    expect(page.locator(".review-value").first).to_be_visible()


def test_setup_complete_flow(page, setup_server, clean_setup):
    """Full wizard: complete all steps, verify redirect to dashboard."""
    data_dir = clean_setup
    _walk_to_step3(page, setup_server)
    page.locator("button[type='submit']").click()
    page.wait_for_url(re.compile(r"/\?setup=complete"))
    assert (data_dir / "config.yaml").exists()


def test_setup_back_navigation(page, setup_server, clean_setup):
    """Back links should return to previous step."""
    _walk_to_step2(page, setup_server)
    page.locator("a.btn-secondary").click()
    page.wait_for_url(re.compile(r"/setup"))
    expect(page.locator(".preset-card").first).to_be_visible()


def test_post_setup_wizard_redirects_to_settings(page, setup_server, clean_setup):
    """After config exists, /setup GET should redirect to /settings."""
    # Complete setup first
    _walk_to_step3(page, setup_server)
    page.locator("button[type='submit']").click()
    page.wait_for_url(re.compile(r"/\?setup=complete"))

    # Now try to access /setup again — should redirect to /settings
    page.goto(f"{setup_server}/setup")
    page.wait_for_url(re.compile(r"/settings"))
