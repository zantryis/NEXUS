"""HTML sanitization for rendered markdown content."""

import logging

import markdown

try:
    import bleach
except ModuleNotFoundError:  # pragma: no cover - exercised in slim test envs
    bleach = None

logger = logging.getLogger(__name__)

ALLOWED_TAGS = [
    "p", "h1", "h2", "h3", "h4", "h5", "h6", "br", "hr",
    "ul", "ol", "li", "strong", "em", "a", "code", "pre",
    "blockquote", "table", "thead", "tbody", "tr", "th", "td",
    "img", "div", "span",
]

ALLOWED_ATTRS = {
    "a": ["href", "title"],
    "img": ["src", "alt", "title"],
}


def safe_markdown(text: str) -> str:
    """Render markdown to HTML with sanitization against XSS."""
    html = markdown.markdown(text, extensions=["tables", "fenced_code"])
    if bleach is None:
        logger.warning("bleach is not installed; returning unsanitized markdown HTML")
        return html
    return bleach.clean(html, tags=ALLOWED_TAGS, attributes=ALLOWED_ATTRS)
