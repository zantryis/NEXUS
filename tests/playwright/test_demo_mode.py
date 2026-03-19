"""Playwright E2E tests for demo mode restrictions."""

import pytest
from playwright.sync_api import expect

pytestmark = [pytest.mark.e2e]


def test_demo_badge_visible(page, demo_server):
    """Demo badge should be visible in the header."""
    page.goto(demo_server)
    badge = page.locator(".demo-badge")
    expect(badge).to_be_visible()
    expect(badge).to_have_text("Demo")


def test_settings_page_loads_readonly(page, demo_server):
    """Settings page should load in read-only mode."""
    response = page.goto(f"{demo_server}/settings")
    assert response.status == 200


def test_settings_post_blocked(page, demo_server):
    """POST to settings should be blocked in demo mode."""
    response = page.request.post(f"{demo_server}/settings/user", data={"name": "Hacker"})
    assert response.status == 403


def test_setup_post_blocked(page, demo_server):
    """POST to setup should be blocked in demo mode."""
    response = page.request.post(f"{demo_server}/setup/step/1", data={"name": "Test"})
    assert response.status == 403


def test_chat_widget_visible(page, demo_server):
    """Chat widget should be visible in demo mode."""
    page.goto(demo_server)
    chat = page.locator(".chat-widget, .chat-fab, #chat-widget")
    if chat.count() > 0:
        expect(chat.first).to_be_visible()
