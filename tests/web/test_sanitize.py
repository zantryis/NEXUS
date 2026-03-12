"""Tests for markdown sanitization."""

from nexus.web.sanitize import safe_markdown


def test_script_tags_stripped():
    """Script tags must be neutralized (escaped or removed)."""
    result = safe_markdown("<script>alert('xss')</script>Hello")
    assert "<script>" not in result
    assert "Hello" in result


def test_onerror_attribute_stripped():
    """Event handler attributes must be removed."""
    result = safe_markdown('<img src=x onerror="alert(1)">')
    assert "onerror" not in result


def test_normal_markdown_preserved():
    """Standard markdown renders correctly."""
    result = safe_markdown("# Hello\n\nThis is **bold** and *italic*.")
    assert "<h1>" in result
    assert "<strong>bold</strong>" in result
    assert "<em>italic</em>" in result


def test_links_preserved():
    """Links are allowed but only href/title attributes."""
    result = safe_markdown('[Click here](https://example.com "Example")')
    assert 'href="https://example.com"' in result
    assert "Click here" in result


def test_tables_preserved():
    """Markdown tables render correctly."""
    md = "| A | B |\n|---|---|\n| 1 | 2 |"
    result = safe_markdown(md)
    assert "<table>" in result
    assert "<td>" in result


def test_iframe_stripped():
    """Iframes must be removed."""
    result = safe_markdown('<iframe src="https://evil.com"></iframe>Text')
    assert "<iframe" not in result
    assert "Text" in result


def test_style_attribute_stripped():
    """Style attributes must be removed."""
    result = safe_markdown('<div style="background:url(javascript:alert(1))">Content</div>')
    assert "style=" not in result
    assert "Content" in result
