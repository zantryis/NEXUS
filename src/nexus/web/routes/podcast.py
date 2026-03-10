"""Podcast RSS feed route — serves audio briefings as a podcast."""

from datetime import datetime
from pathlib import Path
from xml.etree.ElementTree import Element, SubElement, tostring

from fastapi import APIRouter, Request
from fastapi.responses import Response

router = APIRouter()


def _build_rss(audio_dir: Path, base_url: str) -> bytes:
    """Build RSS XML for podcast clients."""
    rss = Element("rss", version="2.0")
    rss.set("xmlns:itunes", "http://www.itunes.com/dtds/podcast-1.0.dtd")
    channel = SubElement(rss, "channel")

    SubElement(channel, "title").text = "Nexus Intelligence Briefing"
    SubElement(channel, "description").text = (
        "Daily multi-perspective news intelligence briefing."
    )
    SubElement(channel, "language").text = "en"
    SubElement(channel, "link").text = base_url

    itunes_author = SubElement(channel, "itunes:author")
    itunes_author.text = "Nexus"

    # List MP3 files newest first
    if audio_dir.exists():
        mp3_files = sorted(audio_dir.glob("*.mp3"), reverse=True)
        for mp3 in mp3_files[:30]:  # Last 30 episodes
            item = SubElement(channel, "item")
            date_str = mp3.stem  # e.g. "2026-03-10"
            SubElement(item, "title").text = f"Briefing — {date_str}"

            enc_url = f"{base_url}/audio/{mp3.name}"
            size = mp3.stat().st_size
            enclosure = SubElement(item, "enclosure")
            enclosure.set("url", enc_url)
            enclosure.set("length", str(size))
            enclosure.set("type", "audio/mpeg")

            try:
                pub_date = datetime.strptime(date_str, "%Y-%m-%d")
                SubElement(item, "pubDate").text = pub_date.strftime(
                    "%a, %d %b %Y 06:00:00 +0000"
                )
            except ValueError:
                pass

    return b'<?xml version="1.0" encoding="UTF-8"?>\n' + tostring(rss, encoding="unicode").encode()


@router.get("/feed.xml")
async def podcast_feed(request: Request):
    """Serve podcast RSS feed."""
    audio_dir = getattr(request.app.state, "audio_dir", None)
    if audio_dir is None:
        audio_dir = Path("data/artifacts/audio")

    base_url = str(request.base_url).rstrip("/")
    xml_bytes = _build_rss(audio_dir, base_url)
    return Response(content=xml_bytes, media_type="application/rss+xml")
