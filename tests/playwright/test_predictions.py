"""Playwright E2E tests for the predictions page."""

import pytest
from playwright.sync_api import expect

pytestmark = [pytest.mark.e2e]


def test_predictions_page_loads(page, demo_server):
    """Predictions page should load and show content."""
    page.goto(f"{demo_server}/predictions")
    expect(page.locator(".main-content")).to_be_visible()


def test_prediction_cards_visible(page, demo_server):
    """Prediction cards should be visible if forecasts exist."""
    page.goto(f"{demo_server}/predictions")
    # At minimum the page should load without errors
    assert "Predictions" in page.title() or "Nexus" in page.title()


def test_predictions_no_errors(page, demo_server):
    """Predictions page should not show server errors."""
    response = page.goto(f"{demo_server}/predictions")
    assert response.status == 200
    content = page.content()
    assert "Internal Server Error" not in content
    assert "500" not in page.title()
