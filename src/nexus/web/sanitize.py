"""HTML sanitization for rendered markdown content."""

import bleach
import markdown

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
    return bleach.clean(html, tags=ALLOWED_TAGS, attributes=ALLOWED_ATTRS)
