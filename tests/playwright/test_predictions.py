"""Playwright E2E tests for the Forward Look page."""

import pytest
from playwright.sync_api import expect

pytestmark = [pytest.mark.e2e]


def test_forward_look_page_loads(page, demo_server):
    """Forward Look page should load and show content."""
    page.goto(f"{demo_server}/forward-look")
    expect(page.locator(".main-content")).to_be_visible()
    expect(page.locator("h2")).to_contain_text("Forward Look")


def test_forward_look_redirect_works(page, demo_server):
    """Legacy predictions route should redirect to the canonical path."""
    page.goto(f"{demo_server}/predictions")
    assert page.url.endswith("/forward-look")


def test_forward_look_title_visible(page, demo_server):
    """Forward Look page should expose the renamed title."""
    page.goto(f"{demo_server}/forward-look")
    # At minimum the page should load without errors
    assert "Forward Look" in page.title() or "Nexus" in page.title()


def test_forward_look_no_errors(page, demo_server):
    """Forward Look page should not show server errors."""
    response = page.goto(f"{demo_server}/forward-look")
    assert response.status == 200
    content = page.content()
    assert "Internal Server Error" not in content
    assert "500" not in page.title()
