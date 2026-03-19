"""Playwright E2E tests for navigation and page loading."""

import pytest
from playwright.sync_api import expect

pytestmark = [pytest.mark.e2e]


_NAV_ROUTES = [
    ("/", "Briefing"),
    ("/threads/", "Threads"),
    ("/predictions", "Predictions"),
    ("/explore/", "Explore"),
    ("/sources/", "Sources"),
    ("/cost", "Cost"),
    ("/settings", "Settings"),
    ("/changes/", "Changes"),
    ("/entities/", "Entities"),
]


@pytest.mark.parametrize("path,label", _NAV_ROUTES)
def test_nav_route_loads(page, demo_server, path, label):
    """Each nav route should return 200 and render content."""
    response = page.goto(f"{demo_server}{path}")
    assert response.status == 200, f"{path} returned {response.status}"
    expect(page.locator(".site-header")).to_be_visible()


def test_mobile_hamburger_menu(page, demo_server):
    """Hamburger menu should toggle nav visibility on mobile viewport."""
    page.set_viewport_size({"width": 375, "height": 667})
    page.goto(demo_server)

    hamburger = page.locator(".nav-hamburger")
    expect(hamburger).to_be_visible()

    # Nav should be hidden initially
    nav_primary = page.locator(".nav-primary")
    expect(nav_primary).not_to_be_visible()

    # Click hamburger to open
    hamburger.click()
    expect(nav_primary).to_be_visible()

    # Click again to close
    hamburger.click()
    expect(nav_primary).not_to_be_visible()


def test_threads_page_renders(page, demo_server):
    """Threads page should show thread list."""
    page.goto(f"{demo_server}/threads/")
    expect(page.locator(".main-content")).to_be_visible()


def test_explore_page_renders(page, demo_server):
    """Explore page should load."""
    page.goto(f"{demo_server}/explore/")
    expect(page.locator(".main-content")).to_be_visible()


def test_sources_page_renders(page, demo_server):
    """Sources page should show source registries."""
    page.goto(f"{demo_server}/sources/")
    expect(page.locator(".main-content")).to_be_visible()


def test_cost_page_renders(page, demo_server):
    """Cost tracking page should load."""
    page.goto(f"{demo_server}/cost")
    expect(page.locator(".main-content")).to_be_visible()


def test_density_toggle(page, demo_server):
    """Density toggle should switch between comfort and full modes."""
    page.goto(demo_server)

    toggle = page.locator(".density-toggle")
    expect(toggle).to_be_visible()

    # Default should be "full"
    html = page.locator("html")
    assert html.get_attribute("data-density") == "full"

    # Click to switch to comfort
    toggle.click()
    assert html.get_attribute("data-density") == "comfort"

    # Click again to switch back to full
    toggle.click()
    assert html.get_attribute("data-density") == "full"
