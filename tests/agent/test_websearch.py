"""Tests for web search integration in Q&A."""

from unittest.mock import patch, MagicMock

from nexus.agent.websearch import web_search, _is_context_thin


def test_context_thin_empty():
    assert _is_context_thin("No data available in the knowledge store.") is True


def test_context_thin_short():
    assert _is_context_thin("## topic\n- one event") is True


def test_context_not_thin():
    # Lots of events = not thin
    lines = ["## events"] + [f"- [{i}] event {i}" for i in range(20)]
    assert _is_context_thin("\n".join(lines)) is False


@patch("nexus.agent.websearch.DDGS")
async def test_web_search_returns_snippets(mock_ddgs_cls):
    """Web search returns formatted snippets."""
    mock_ddgs = MagicMock()
    mock_ddgs.text.return_value = [
        {"title": "F1 2026 Rules", "body": "New engine regs for 2026.", "href": "https://f1.com/rules"},
        {"title": "Honda Returns", "body": "Honda rejoins as PU supplier.", "href": "https://f1.com/honda"},
    ]
    mock_ddgs_cls.return_value.__enter__ = MagicMock(return_value=mock_ddgs)
    mock_ddgs_cls.return_value.__exit__ = MagicMock(return_value=False)

    results = await web_search("F1 2026 engine regulations", max_results=3)

    assert len(results) == 2
    assert "F1 2026 Rules" in results[0]["title"]
    assert "https://f1.com/rules" in results[0]["url"]


@patch("nexus.agent.websearch.DDGS")
async def test_web_search_handles_failure(mock_ddgs_cls):
    """Web search gracefully handles errors."""
    mock_ddgs_cls.return_value.__enter__ = MagicMock(
        side_effect=Exception("Network error")
    )
    mock_ddgs_cls.return_value.__exit__ = MagicMock(return_value=False)

    results = await web_search("test query")
    assert results == []


@patch("nexus.agent.websearch.DDGS")
async def test_web_search_formats_context(mock_ddgs_cls):
    """Web search results format into context string."""
    from nexus.agent.websearch import format_web_results

    results = [
        {"title": "Article 1", "snippet": "Content 1", "url": "https://a.com"},
        {"title": "Article 2", "snippet": "Content 2", "url": "https://b.com"},
    ]
    context = format_web_results(results)
    assert "Article 1" in context
    assert "https://a.com" in context
    assert "Web search" in context
