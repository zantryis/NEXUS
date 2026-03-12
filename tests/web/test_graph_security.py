"""Tests for SVG graph rendering security."""

from nexus.web.graph import render_entity_network_svg


def test_entity_names_are_escaped():
    """Entity names with HTML/SVG special chars must be escaped."""
    related = [
        {"id": 1, "canonical_name": '<script>alert("xss")</script>', "co_occurrence_count": 5},
    ]
    svg = render_entity_network_svg("Center", related)
    assert "<script>" not in svg
    assert "&lt;script&gt;" in svg


def test_center_name_is_escaped():
    """Center entity name must also be escaped — no raw HTML tags."""
    related = [
        {"id": 1, "canonical_name": "Normal", "co_occurrence_count": 3},
    ]
    svg = render_entity_network_svg('<img src=x>', related)
    # The raw <img> tag must be escaped, not rendered as HTML
    assert "<img" not in svg
    assert "&lt;img" in svg


def test_normal_names_unchanged():
    """Normal names render without escaping artifacts."""
    related = [
        {"id": 1, "canonical_name": "OpenAI", "co_occurrence_count": 5},
        {"id": 2, "canonical_name": "Google DeepMind", "co_occurrence_count": 3},
    ]
    svg = render_entity_network_svg("Anthropic", related)
    assert "OpenAI" in svg
    assert "Anthropic" in svg
    assert "Google DeepMind" in svg
