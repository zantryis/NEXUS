"""Custom Jinja2 template filters for the Nexus web dashboard."""

from datetime import datetime, timezone


def timeago(dt_string: str) -> str:
    """Convert an ISO datetime string to a human-readable relative time.

    Returns strings like "just now", "5m ago", "3h ago", "2 days ago".
    """
    if not dt_string:
        return ""
    try:
        dt = datetime.fromisoformat(dt_string)
    except (ValueError, TypeError):
        return dt_string

    # Treat naive datetimes as UTC
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    now = datetime.now(timezone.utc)
    diff = now - dt

    seconds = int(diff.total_seconds())
    if seconds < 0:
        return "just now"
    if seconds < 60:
        return "just now"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes}m ago"
    hours = minutes // 60
    if hours < 24:
        return f"{hours}h ago"
    days = hours // 24
    if days < 30:
        return f"{days}d ago"
    months = days // 30
    if months < 12:
        return f"{months}mo ago"
    years = days // 365
    return f"{years}y ago"
