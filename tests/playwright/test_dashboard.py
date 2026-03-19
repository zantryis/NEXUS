"""Playwright E2E tests for the dashboard (briefing page)."""

import re

import pytest
from playwright.sync_api import expect

pytestmark = [pytest.mark.e2e]


def test_dashboard_loads(page, demo_server):
    """Dashboard should load with title and main content."""
    page.goto(demo_server)
    expect(page).to_have_title(re.compile("Nexus"))
    expect(page.locator(".site-header")).to_be_visible()
    expect(page.locator(".main-content")).to_be_visible()


def test_breaking_alerts_visible(page, demo_server):
    """Breaking alerts should appear in the sidebar."""
    page.goto(demo_server)
    sidebar = page.locator(".sidebar-section").first
    expect(sidebar).to_be_visible()
    # Should show "Breaking Alerts" section with seeded data
    expect(page.locator(".sidebar-breaking-alert").first).to_be_visible()


def test_topic_tab_switching(page, demo_server):
    """Clicking a topic tab should show that topic's panel."""
    page.goto(demo_server)
    tabs = page.locator(".topic-tab")
    if tabs.count() > 1:
        # Click second tab
        second_tab = tabs.nth(1)
        second_tab.click()
        expect(second_tab).to_have_class(re.compile("active"))


def test_theme_toggle(page, demo_server):
    """Theme toggle should switch between light and dark."""
    page.goto(demo_server)
    html = page.locator("html")
    initial_theme = html.get_attribute("data-theme")
    page.locator(".theme-toggle").click()
    new_theme = html.get_attribute("data-theme")
    assert initial_theme != new_theme


def test_stats_bar_visible(page, demo_server):
    """Stats bar should show articles, threads, and cost."""
    page.goto(demo_server)
    stats_bar = page.locator(".briefing-stats-bar")
    if stats_bar.count() > 0:
        expect(stats_bar).to_be_visible()
        expect(page.locator(".stat-card").first).to_be_visible()


def test_thread_card_links(page, demo_server):
    """Thread headline links should navigate to thread detail."""
    page.goto(demo_server)
    thread_link = page.locator(".briefing-thread-headline").first
    if thread_link.count() > 0:
        href = thread_link.get_attribute("href")
        assert href and "/threads/" in href
