"""Tests for entity resolution — canonicalizing raw entity strings."""

import json
import pytest
from unittest.mock import AsyncMock

from nexus.engine.knowledge.entities import (
    EntityResolution,
    resolve_entities,
)


@pytest.fixture
def mock_llm():
    return AsyncMock()


def _known(name, etype="unknown", aliases=None):
    return {"canonical_name": name, "entity_type": etype, "aliases": aliases or []}


# ── Basic resolution ─────────────────────────────────────────────


async def test_resolve_empty_list(mock_llm):
    result = await resolve_entities(mock_llm, [], [])
    assert result == []
    mock_llm.complete.assert_not_called()


async def test_resolve_new_entities(mock_llm):
    mock_llm.complete.return_value = json.dumps([
        {"raw": "IAEA", "canonical": "IAEA", "type": "org", "is_new": True},
        {"raw": "Iran", "canonical": "Iran", "type": "country", "is_new": True},
    ])

    result = await resolve_entities(mock_llm, ["IAEA", "Iran"], [])
    assert len(result) == 2
    assert result[0] == EntityResolution(raw="IAEA", canonical="IAEA", entity_type="org", is_new=True)
    assert result[1] == EntityResolution(raw="Iran", canonical="Iran", entity_type="country", is_new=True)


async def test_resolve_matches_known_entity(mock_llm):
    known = [_known("IAEA", "org", ["International Atomic Energy Agency"])]
    mock_llm.complete.return_value = json.dumps([
        {"raw": "International Atomic Energy Agency", "canonical": "IAEA", "type": "org", "is_new": False},
    ])

    result = await resolve_entities(
        mock_llm, ["International Atomic Energy Agency"], known
    )
    assert len(result) == 1
    assert result[0].canonical == "IAEA"
    assert result[0].is_new is False


async def test_resolve_canonicalizes_informal_names(mock_llm):
    """'US Treasury' should map to 'US Department of the Treasury'."""
    known = [_known("US Department of the Treasury", "org", ["US Treasury", "Treasury"])]
    mock_llm.complete.return_value = json.dumps([
        {"raw": "US Treasury", "canonical": "US Department of the Treasury", "type": "org", "is_new": False},
    ])

    result = await resolve_entities(mock_llm, ["US Treasury"], known)
    assert result[0].canonical == "US Department of the Treasury"
    assert result[0].is_new is False


async def test_resolve_distinguishes_similar_names(mock_llm):
    """'Iran' (country) and 'Iran Air' (org) should not merge."""
    known = [_known("Iran", "country")]
    mock_llm.complete.return_value = json.dumps([
        {"raw": "Iran", "canonical": "Iran", "type": "country", "is_new": False},
        {"raw": "Iran Air", "canonical": "Iran Air", "type": "org", "is_new": True},
    ])

    result = await resolve_entities(mock_llm, ["Iran", "Iran Air"], known)
    assert len(result) == 2
    iran = next(r for r in result if r.raw == "Iran")
    iran_air = next(r for r in result if r.raw == "Iran Air")
    assert iran.canonical == "Iran"
    assert iran_air.canonical == "Iran Air"
    assert iran.entity_type == "country"
    assert iran_air.entity_type == "org"


# ── Mixed batch ──────────────────────────────────────────────────


async def test_resolve_mixed_new_and_known(mock_llm):
    known = [
        _known("IAEA", "org"),
        _known("Iran", "country"),
    ]
    mock_llm.complete.return_value = json.dumps([
        {"raw": "IAEA", "canonical": "IAEA", "type": "org", "is_new": False},
        {"raw": "Ali Khamenei", "canonical": "Ali Khamenei", "type": "person", "is_new": True},
    ])

    result = await resolve_entities(mock_llm, ["IAEA", "Ali Khamenei"], known)
    assert len(result) == 2
    iaea = next(r for r in result if r.raw == "IAEA")
    khamenei = next(r for r in result if r.raw == "Ali Khamenei")
    assert iaea.is_new is False
    assert khamenei.is_new is True
    assert khamenei.entity_type == "person"


# ── LLM prompt construction ─────────────────────────────────────


async def test_prompt_includes_known_entities(mock_llm):
    mock_llm.complete.return_value = json.dumps([
        {"raw": "test", "canonical": "test", "type": "unknown", "is_new": True},
    ])
    known = [
        _known("IAEA", "org", ["International Atomic Energy Agency"]),
        _known("Iran", "country"),
    ]

    await resolve_entities(mock_llm, ["test"], known)

    call_args = mock_llm.complete.call_args
    user_prompt = call_args.kwargs["user_prompt"]
    assert "IAEA [org]" in user_prompt
    assert "International Atomic Energy Agency" in user_prompt
    assert "Iran [country]" in user_prompt


async def test_uses_knowledge_summary_config_key(mock_llm):
    mock_llm.complete.return_value = json.dumps([
        {"raw": "x", "canonical": "x", "type": "unknown", "is_new": True},
    ])

    await resolve_entities(mock_llm, ["x"], [])

    call_args = mock_llm.complete.call_args
    assert call_args.kwargs["config_key"] == "knowledge_summary"
    assert call_args.kwargs["json_response"] is True


async def test_known_entities_capped_at_200(mock_llm):
    """Prompt shouldn't overflow with huge entity lists."""
    mock_llm.complete.return_value = json.dumps([
        {"raw": "x", "canonical": "x", "type": "unknown", "is_new": True},
    ])
    known = [_known(f"Entity-{i}", "unknown") for i in range(500)]

    await resolve_entities(mock_llm, ["x"], known)

    user_prompt = mock_llm.complete.call_args.kwargs["user_prompt"]
    # Should have at most 200 entity lines
    entity_lines = [l for l in user_prompt.split("\n") if l.startswith("- Entity-")]
    assert len(entity_lines) == 200


# ── Error handling / fallback ────────────────────────────────────


async def test_fallback_on_json_decode_error(mock_llm):
    mock_llm.complete.return_value = "not valid json {{"

    result = await resolve_entities(mock_llm, ["IAEA", "Iran"], [])
    assert len(result) == 2
    assert result[0].raw == "IAEA"
    assert result[0].canonical == "IAEA"
    assert result[0].entity_type == "unknown"
    assert result[0].is_new is True


async def test_fallback_on_missing_raw_field(mock_llm):
    mock_llm.complete.return_value = json.dumps([
        {"canonical": "IAEA", "type": "org"},  # Missing "raw" key
    ])

    result = await resolve_entities(mock_llm, ["IAEA", "Iran"], [])
    # Should fallback to raw-as-canonical for all
    assert len(result) == 2
    assert all(r.canonical == r.raw for r in result)


async def test_fallback_on_type_error(mock_llm):
    mock_llm.complete.return_value = json.dumps("not a list or dict")

    result = await resolve_entities(mock_llm, ["IAEA"], [])
    assert len(result) == 1
    assert result[0].canonical == "IAEA"


# ── Edge cases ───────────────────────────────────────────────────


async def test_resolve_single_entity(mock_llm):
    mock_llm.complete.return_value = json.dumps([
        {"raw": "OPEC", "canonical": "OPEC", "type": "org", "is_new": True},
    ])

    result = await resolve_entities(mock_llm, ["OPEC"], [])
    assert len(result) == 1
    assert result[0].entity_type == "org"


async def test_resolve_handles_missing_optional_fields(mock_llm):
    """LLM may omit 'canonical', 'type', or 'is_new'."""
    mock_llm.complete.return_value = json.dumps([
        {"raw": "IAEA"},  # Only raw field
    ])

    result = await resolve_entities(mock_llm, ["IAEA"], [])
    assert len(result) == 1
    assert result[0].canonical == "IAEA"  # Falls back to raw
    assert result[0].entity_type == "unknown"  # Default
    assert result[0].is_new is True  # Default


async def test_resolve_wraps_single_object_response(mock_llm):
    """LLM might return a single object instead of array."""
    mock_llm.complete.return_value = json.dumps(
        {"raw": "IAEA", "canonical": "IAEA", "type": "org", "is_new": True}
    )

    result = await resolve_entities(mock_llm, ["IAEA"], [])
    assert len(result) == 1
    assert result[0].canonical == "IAEA"
