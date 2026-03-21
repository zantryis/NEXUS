"""Entity thumbnail + Wikipedia URL fetching via Wikipedia API and flag CDN."""

import asyncio
import logging

import httpx

logger = logging.getLogger(__name__)

WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"

# ISO 3166-1 alpha-2 codes for known country entities
COUNTRY_ISO2 = {
    "iran": "ir", "united states": "us", "israel": "il", "russia": "ru",
    "china": "cn", "saudi arabia": "sa", "turkey": "tr", "ukraine": "ua",
    "japan": "jp", "south korea": "kr", "north korea": "kp",
    "india": "in", "pakistan": "pk", "france": "fr", "germany": "de",
    "united kingdom": "gb", "italy": "it", "brazil": "br", "canada": "ca",
    "australia": "au", "mexico": "mx", "spain": "es", "egypt": "eg",
    "uae": "ae", "iraq": "iq", "syria": "sy", "qatar": "qa",
    "kuwait": "kw", "bahrain": "bh", "yemen": "ye", "lebanon": "lb",
    "jordan": "jo", "oman": "om", "afghanistan": "af", "philippines": "ph",
}


def get_flag_url(country_name: str) -> str | None:
    """Return flag CDN URL for a country entity."""
    iso = COUNTRY_ISO2.get(country_name.lower())
    return f"https://flagcdn.com/48x36/{iso}.png" if iso else None


def get_wikipedia_url_for_country(country_name: str) -> str:
    """Return a Wikipedia URL for a country."""
    slug = country_name.replace(" ", "_")
    return f"https://en.wikipedia.org/wiki/{slug}"


async def fetch_wikipedia_info(name: str, size: int = 100) -> dict:
    """Fetch entity thumbnail URL and Wikipedia page URL from Wikipedia API.

    Returns: {"thumbnail_url": str|None, "wikipedia_url": str|None}
    """
    params = {
        "action": "query",
        "titles": name,
        "prop": "pageimages|info",
        "pithumbsize": size,
        "inprop": "url",
        "format": "json",
        "redirects": 1,
    }
    result = {"thumbnail_url": None, "wikipedia_url": None}
    try:
        async with httpx.AsyncClient(
            timeout=10,
            headers={"User-Agent": "nexus/1.0 (news intelligence tool)"},
        ) as client:
            resp = await client.get(WIKIPEDIA_API, params=params)
            resp.raise_for_status()
            data = resp.json()
            pages = data.get("query", {}).get("pages", {})
            for page_id, page in pages.items():
                if page_id == "-1":
                    continue
                thumb = page.get("thumbnail", {}).get("source")
                if thumb:
                    result["thumbnail_url"] = thumb
                url = page.get("fullurl")
                if url:
                    result["wikipedia_url"] = url
    except Exception as e:
        logger.debug("Wikipedia info fetch failed for %s: %s", name, e)
    return result


# Keep backwards compat
async def fetch_wikipedia_thumbnail(name: str, size: int = 100) -> str | None:
    """Fetch entity thumbnail URL from Wikipedia API."""
    info = await fetch_wikipedia_info(name, size)
    return info["thumbnail_url"]


async def populate_thumbnails(store) -> dict:
    """Populate thumbnails and Wikipedia URLs for all entities missing them.

    Returns: {fetched: int, skipped: int, failed: int}
    """
    entities = await store.get_all_entities()
    stats = {"fetched": 0, "skipped": 0, "failed": 0}

    for ent in entities:
        if ent.get("thumbnail_url") and ent.get("wikipedia_url"):
            stats["skipped"] += 1
            continue

        etype = ent.get("entity_type", "unknown")
        name = ent["canonical_name"]

        # Country → flag CDN + Wikipedia URL
        if etype == "country":
            thumb = get_flag_url(name) or ent.get("thumbnail_url", "")
            wiki = get_wikipedia_url_for_country(name)
            if thumb or wiki:
                await store.update_entity_media(
                    ent["id"], thumbnail_url=thumb, wikipedia_url=wiki,
                )
                stats["fetched"] += 1
                continue

        # Person, org → Wikipedia API for both thumbnail + URL
        if etype in ("person", "org", "country"):
            info = await fetch_wikipedia_info(name)
            thumb = info["thumbnail_url"] or ent.get("thumbnail_url", "")
            wiki = info["wikipedia_url"] or ent.get("wikipedia_url", "")
            if thumb or wiki:
                await store.update_entity_media(
                    ent["id"], thumbnail_url=thumb, wikipedia_url=wiki,
                )
                stats["fetched"] += 1
            else:
                stats["failed"] += 1
            # Rate limit: 1 req/sec for Wikipedia etiquette
            await asyncio.sleep(1.0)
        else:
            # Concepts, treaties — try Wikipedia too
            info = await fetch_wikipedia_info(name)
            if info["thumbnail_url"] or info["wikipedia_url"]:
                await store.update_entity_media(
                    ent["id"],
                    thumbnail_url=info["thumbnail_url"] or "",
                    wikipedia_url=info["wikipedia_url"] or "",
                )
                stats["fetched"] += 1
            else:
                stats["skipped"] += 1
            await asyncio.sleep(1.0)

    return stats


async def enrich_new_entities(store, entity_ids: list[int]) -> None:
    """Fetch thumbnails + Wikipedia URLs for a batch of newly created entities.

    Called automatically by the pipeline after entity resolution.
    """
    for eid in entity_ids:
        ent = await store.find_entity_by_id(eid)
        if not ent or ent.get("thumbnail_url"):
            continue

        etype = ent.get("entity_type", "unknown")
        name = ent["canonical_name"]

        if etype == "country":
            thumb = get_flag_url(name) or ""
            wiki = get_wikipedia_url_for_country(name)
            await store.update_entity_media(eid, thumbnail_url=thumb, wikipedia_url=wiki)
            continue

        info = await fetch_wikipedia_info(name)
        if info["thumbnail_url"] or info["wikipedia_url"]:
            await store.update_entity_media(
                eid,
                thumbnail_url=info["thumbnail_url"] or "",
                wikipedia_url=info["wikipedia_url"] or "",
            )
        # Rate limit
        await asyncio.sleep(0.5)
