"""Tests for knowledge synthesis — TopicSynthesis builder."""

import json
import pytest
from datetime import date
from unittest.mock import AsyncMock
from nexus.config.models import TopicConfig
from nexus.engine.knowledge.events import Event
from nexus.engine.sources.polling import ContentItem
from nexus.engine.synthesis.knowledge import (
    TopicSynthesis, NarrativeThread, _persist_threads, synthesize_topic,
)
from nexus.engine.synthesis.threads import ThreadMatch


@pytest.fixture
def topic():
    return TopicConfig(name="Iran-US Relations", subtopics=["sanctions", "nuclear"])


@pytest.fixture
def events():
    return [
        Event(
            date=date(2026, 3, 9),
            summary="US announces new sanctions on Iran",
            entities=["US", "Iran", "Treasury Dept"],
            sources=[{"url": "https://nyt.com/1", "outlet": "nyt", "affiliation": "private", "country": "US"}],
            significance=8,
        ),
        Event(
            date=date(2026, 3, 9),
            summary="Iran condemns new US sanctions",
            entities=["Iran", "US", "Foreign Ministry"],
            sources=[{"url": "https://tass.com/1", "outlet": "tass", "affiliation": "state", "country": "RU"}],
            significance=7,
        ),
    ]


@pytest.fixture
def articles():
    return [
        ContentItem(
            title="Sanctions article", url="https://nyt.com/1", source_id="nyt",
            source_affiliation="private", detected_language="en",
        ),
        ContentItem(
            title="Iran response", url="https://tass.com/1", source_id="tass",
            source_affiliation="state", detected_language="en",
        ),
    ]


@pytest.mark.asyncio
async def test_synthesize_topic_produces_threads(topic, events, articles):
    mock_llm = AsyncMock()
    mock_llm.complete.return_value = json.dumps({
        "threads": [
            {
                "headline": "US-Iran sanctions escalation",
                "event_indices": [0, 1],
                "convergence": ["New sanctions were announced"],
                "divergence": [{
                    "claim": "Sanctions impact",
                    "source_a": "nyt",
                    "framing_a": "Targeted response to nuclear program",
                    "source_b": "tass",
                    "framing_b": "Unjustified economic warfare",
                }],
                "key_entities": ["US", "Iran", "Treasury Dept"],
                "significance": 8,
            }
        ]
    })

    result = await synthesize_topic(mock_llm, topic, events, articles, [], [])

    assert isinstance(result, TopicSynthesis)
    assert result.topic_name == "Iran-US Relations"
    assert len(result.threads) == 1
    assert result.threads[0].headline == "US-Iran sanctions escalation"
    assert len(result.threads[0].convergence) == 1
    assert len(result.threads[0].divergence) == 1
    assert result.source_balance == {"private": 1, "state": 1}
    assert "en" in result.languages_represented


@pytest.mark.asyncio
async def test_synthesize_topic_empty_events(topic, articles):
    mock_llm = AsyncMock()
    result = await synthesize_topic(mock_llm, topic, [], articles, [], [])

    assert result.topic_name == "Iran-US Relations"
    assert result.threads == []
    assert result.metadata["event_count"] == 0
    mock_llm.complete.assert_not_called()


@pytest.mark.asyncio
async def test_synthesize_topic_fallback_on_bad_json(topic, events, articles):
    mock_llm = AsyncMock()
    mock_llm.complete.return_value = "not valid json"

    result = await synthesize_topic(mock_llm, topic, events, articles, [], [])

    # Fallback: one thread per event
    assert len(result.threads) == 2
    assert result.metadata.get("fallback") is True


@pytest.mark.asyncio
async def test_synthesize_topic_tolerates_missing_thread_headline(topic, events, articles):
    mock_llm = AsyncMock()
    mock_llm.complete.return_value = json.dumps(
        {
            "threads": [
                {
                    "event_indices": [0, 1],
                    "convergence": [],
                    "divergence": [],
                    "key_entities": ["US", "Iran"],
                    "significance": 8,
                }
            ]
        }
    )

    result = await synthesize_topic(mock_llm, topic, events, articles, [], [])

    assert len(result.threads) == 1
    assert result.metadata.get("fallback") is None
    assert result.threads[0].headline == "US and Iran developments"


def test_topic_synthesis_model():
    syn = TopicSynthesis(
        topic_name="Test",
        threads=[NarrativeThread(headline="Thread 1", significance=7)],
        source_balance={"private": 3, "state": 1},
        languages_represented=["en", "fa"],
    )
    assert syn.topic_name == "Test"
    assert len(syn.threads) == 1
    assert syn.threads[0].significance == 7


@pytest.mark.asyncio
async def test_synthesize_new_convergence_format(topic, events, articles):
    """New convergence format with fact + confirmed_by is parsed correctly."""
    mock_llm = AsyncMock()
    mock_llm.complete.return_value = json.dumps({
        "threads": [{
            "headline": "Sanctions thread",
            "event_indices": [0, 1],
            "convergence": [
                {"fact": "New sanctions were announced", "confirmed_by": ["nyt", "tass"]}
            ],
            "divergence": [{
                "shared_event": "US sanctions announcement",
                "source_a": "nyt", "framing_a": "Targeted response",
                "source_b": "tass", "framing_b": "Economic warfare",
            }],
            "key_entities": ["US", "Iran"],
            "significance": 8,
        }]
    })

    result = await synthesize_topic(mock_llm, topic, events, articles, [], [])
    thread = result.threads[0]
    assert len(thread.convergence) == 1
    assert isinstance(thread.convergence[0], dict)
    assert thread.convergence[0]["fact"] == "New sanctions were announced"
    assert "nyt" in thread.convergence[0]["confirmed_by"]
    assert thread.divergence[0]["shared_event"] == "US sanctions announcement"


@pytest.mark.asyncio
async def test_synthesize_single_source_empty_convergence(topic, articles):
    """When LLM correctly returns empty convergence for single-source thread."""
    single_source_events = [
        Event(
            date=date(2026, 3, 9),
            summary="BBC reports on sanctions",
            entities=["US", "Iran"],
            sources=[{"url": "https://bbc.com/1", "outlet": "bbc", "affiliation": "public", "country": "GB"}],
            significance=7,
        ),
    ]
    mock_llm = AsyncMock()
    mock_llm.complete.return_value = json.dumps({
        "threads": [{
            "headline": "Single source thread",
            "event_indices": [0],
            "convergence": [],
            "divergence": [],
            "key_entities": ["US", "Iran"],
            "significance": 7,
        }]
    })

    result = await synthesize_topic(mock_llm, topic, single_source_events, articles, [], [])
    assert result.threads[0].convergence == []
    assert result.threads[0].divergence == []


def test_build_synthesis_prompt_narrow():
    from nexus.engine.synthesis.knowledge import _build_synthesis_prompt
    topic = TopicConfig(name="Iran-US", scope="narrow", subtopics=["sanctions"])
    prompt = _build_synthesis_prompt(topic)
    assert "FOCUSED" in prompt
    assert "causal chains" in prompt


def test_build_synthesis_prompt_broad():
    from nexus.engine.synthesis.knowledge import _build_synthesis_prompt
    topic = TopicConfig(name="AI/ML", scope="broad", subtopics=["agents", "reasoning"])
    prompt = _build_synthesis_prompt(topic)
    assert "BROAD" in prompt
    assert "agents, reasoning" in prompt
    assert "Do NOT merge unrelated subfields" in prompt


def test_build_synthesis_prompt_medium():
    from nexus.engine.synthesis.knowledge import _build_synthesis_prompt
    topic = TopicConfig(name="Energy", scope="medium")
    prompt = _build_synthesis_prompt(topic)
    # Medium scope should not have the scope-specific instructions
    assert "BROAD" not in prompt
    assert "FOCUSED" not in prompt
    # But should still have the base content
    assert "knowledge synthesis engine" in prompt


# ── Divergence Prompt Variants ───────────────────────────────────────────────

def test_build_synthesis_prompt_default_divergence():
    """Default prompt contains the structured divergence wording."""
    from nexus.engine.synthesis.knowledge import _build_synthesis_prompt
    topic = TopicConfig(name="Test", subtopics=["a"])
    prompt = _build_synthesis_prompt(topic)
    assert "TONE CONTRAST" in prompt
    assert "ACTOR FRAMING" in prompt
    assert "'killed' vs 'dead'" in prompt


def test_build_synthesis_prompt_custom_divergence():
    """Custom divergence instructions substitute correctly."""
    from nexus.engine.synthesis.knowledge import _build_synthesis_prompt
    topic = TopicConfig(name="Test", subtopics=["a"])
    custom = "3. Look for framing differences between outlets.\n"
    prompt = _build_synthesis_prompt(
        topic,
        divergence_instructions=custom,
        divergence_output_qualifier="Actively look for divergence. ",
    )
    assert "Look for framing differences" in prompt
    assert "Actively look for divergence" in prompt
    assert "Only flag when outlets disagree" not in prompt


@pytest.mark.asyncio
async def test_synthesize_topic_passes_divergence_instructions(topic, events, articles):
    """Verify divergence_instructions reaches the LLM system prompt."""
    mock_llm = AsyncMock()
    mock_llm.complete.return_value = json.dumps({
        "threads": [{
            "headline": "Test thread",
            "event_indices": [0],
            "convergence": [],
            "divergence": [],
            "key_entities": ["US"],
            "significance": 5,
        }]
    })

    custom_text = "CUSTOM_DIVERGENCE_MARKER_FOR_TEST"
    await synthesize_topic(
        mock_llm, topic, events, articles, [], [],
        divergence_instructions=custom_text,
    )

    # Verify the system_prompt kwarg contains our custom text
    call_kwargs = mock_llm.complete.call_args
    system_prompt = call_kwargs.kwargs.get("system_prompt", call_kwargs.args[1] if len(call_kwargs.args) > 1 else "")
    assert custom_text in system_prompt


# ── Framing in Event Formatting ──────────────────────────────────────────


def test_event_formatting_includes_framing():
    """Events with framing in source dicts produce per-source framing lines in synthesis prompt."""
    events = [Event(
        date=date(2026, 3, 9),
        summary="Sanctions announced",
        entities=["US", "Iran"],
        sources=[
            {"url": "https://nyt.com/1", "outlet": "nyt", "affiliation": "private",
             "country": "US", "framing": "[neutral] Reports policy details; US as enforcer"},
            {"url": "https://tass.com/1", "outlet": "tass", "affiliation": "state",
             "country": "RU", "framing": "[critical] Emphasizes economic damage; US as aggressor"},
        ],
        significance=8,
    )]

    # We need to call synthesize_topic's event formatting, but that's inline.
    # Instead, test via the formatted user_prompt by calling the function.
    # The event formatting is in synthesize_topic, so test the output format directly.
    from nexus.engine.synthesis.knowledge import _format_events
    formatted = _format_events(events)
    assert "nyt (private/US): [neutral]" in formatted
    assert "tass (state/RU): [critical]" in formatted
    assert "US as aggressor" in formatted


def test_event_formatting_no_framing_fallback():
    """Events without framing still format correctly (no framing line)."""
    from nexus.engine.synthesis.knowledge import _format_events
    events = [Event(
        date=date(2026, 3, 9),
        summary="Routine event",
        entities=["US"],
        sources=[
            {"url": "https://nyt.com/1", "outlet": "nyt", "affiliation": "private", "country": "US"},
        ],
        significance=5,
    )]
    formatted = _format_events(events)
    assert "nyt (private/US)" in formatted
    # No ": [" framing marker since no framing was provided
    assert ": [" not in formatted


def test_build_article_snippets_multi_source():
    """Multi-source events get article text snippets."""
    from nexus.engine.synthesis.knowledge import _build_article_snippets
    events = [Event(
        date=date(2026, 3, 9),
        summary="Sanctions event",
        entities=["US"],
        sources=[
            {"url": "https://nyt.com/1", "outlet": "nyt"},
            {"url": "https://tass.com/1", "outlet": "tass"},
        ],
        significance=8,
    )]
    articles = [
        ContentItem(
            title="NYT Article", url="https://nyt.com/1", source_id="nyt",
            full_text="The United States announced new sanctions targeting Iran's oil sector...",
        ),
        ContentItem(
            title="TASS Article", url="https://tass.com/1", source_id="tass",
            full_text="Russia condemned the latest round of American sanctions as economic warfare...",
        ),
    ]
    snippets = _build_article_snippets(events, articles)
    assert "nyt" in snippets
    assert "tass" in snippets
    assert "NYT Article" in snippets
    assert "TASS Article" in snippets


def test_build_article_snippets_single_source_skipped():
    """Single-source events produce no snippets."""
    from nexus.engine.synthesis.knowledge import _build_article_snippets
    events = [Event(
        date=date(2026, 3, 9),
        summary="Solo event",
        entities=["US"],
        sources=[{"url": "https://nyt.com/1", "outlet": "nyt"}],
        significance=5,
    )]
    articles = [
        ContentItem(
            title="NYT Article", url="https://nyt.com/1", source_id="nyt",
            full_text="Some text here...",
        ),
    ]
    snippets = _build_article_snippets(events, articles)
    assert snippets == ""


def test_divergence_variants_dict_structure():
    """DIVERGENCE_VARIANTS has expected keys with instructions + output_qualifier."""
    from nexus.engine.synthesis.knowledge import DIVERGENCE_VARIANTS
    expected_keys = {"baseline", "broadened", "structured", "encouraged"}
    assert set(DIVERGENCE_VARIANTS.keys()) == expected_keys
    for name, variant in DIVERGENCE_VARIANTS.items():
        assert "instructions" in variant, f"{name} missing 'instructions'"
        assert "output_qualifier" in variant, f"{name} missing 'output_qualifier'"
        assert isinstance(variant["instructions"], str)
        assert isinstance(variant["output_qualifier"], str)


# ── Convergence Validation ────────────────────────────────────


def test_validate_convergence_strips_non_independent():
    """Convergence confirmed by 2 same-affiliation/country outlets → stripped."""
    from nexus.engine.synthesis.knowledge import _validate_convergence

    thread = NarrativeThread(
        headline="Test thread",
        events=[Event(
            date=date(2026, 3, 9), summary="Test", significance=5,
            sources=[
                {"outlet": "cgtn", "affiliation": "state", "country": "CN"},
                {"outlet": "xinhua", "affiliation": "state", "country": "CN"},
            ],
        )],
        convergence=[{"fact": "Agreed fact", "confirmed_by": ["cgtn", "xinhua"]}],
    )
    _validate_convergence([thread])
    assert len(thread.convergence) == 0


def test_validate_convergence_keeps_independent():
    """Convergence confirmed by independent outlets → preserved."""
    from nexus.engine.synthesis.knowledge import _validate_convergence

    thread = NarrativeThread(
        headline="Test thread",
        events=[Event(
            date=date(2026, 3, 9), summary="Test", significance=5,
            sources=[
                {"outlet": "nyt", "affiliation": "private", "country": "US"},
                {"outlet": "tass", "affiliation": "state", "country": "RU"},
            ],
        )],
        convergence=[{"fact": "Agreed fact", "confirmed_by": ["nyt", "tass"]}],
    )
    _validate_convergence([thread])
    assert len(thread.convergence) == 1


def test_validate_convergence_empty_noop():
    """Thread with no convergence → no error."""
    from nexus.engine.synthesis.knowledge import _validate_convergence

    thread = NarrativeThread(
        headline="Empty thread",
        convergence=[],
    )
    _validate_convergence([thread])
    assert thread.convergence == []


# ── Thread Overlap Detection ─────────────────────────────────


def test_check_thread_overlaps_detects_high_overlap():
    """Two threads with 4/5 shared entities → detected as overlap."""
    from nexus.engine.synthesis.knowledge import _check_thread_overlaps

    threads = [
        NarrativeThread(
            headline="Thread A",
            key_entities=["Iran", "US", "IAEA", "Sanctions", "EU"],
        ),
        NarrativeThread(
            headline="Thread B",
            key_entities=["Iran", "US", "IAEA", "Sanctions", "Russia"],
        ),
    ]
    overlaps = _check_thread_overlaps(threads)
    assert len(overlaps) == 1
    assert overlaps[0][0] == 0
    assert overlaps[0][1] == 1
    assert overlaps[0][2] > 0.5


def test_check_thread_overlaps_ignores_low_overlap():
    """Two threads with 1/6 shared entities → no overlap flagged."""
    from nexus.engine.synthesis.knowledge import _check_thread_overlaps

    threads = [
        NarrativeThread(
            headline="Thread A",
            key_entities=["Iran", "US", "IAEA"],
        ),
        NarrativeThread(
            headline="Thread B",
            key_entities=["China", "Taiwan", "TSMC", "US"],
        ),
    ]
    overlaps = _check_thread_overlaps(threads)
    assert len(overlaps) == 0


def test_check_thread_overlaps_empty_entities():
    """Threads with no entities → no crash, no overlaps."""
    from nexus.engine.synthesis.knowledge import _check_thread_overlaps

    threads = [
        NarrativeThread(headline="Thread A", key_entities=[]),
        NarrativeThread(headline="Thread B", key_entities=["Iran"]),
    ]
    overlaps = _check_thread_overlaps(threads)
    assert len(overlaps) == 0


async def test_consolidate_threads_merges_overlapping(tmp_path):
    """_consolidate_threads should merge overlapping threads after persistence."""
    from nexus.engine.knowledge.store import KnowledgeStore
    from nexus.engine.synthesis.knowledge import _consolidate_threads

    store = KnowledgeStore(tmp_path / "test.db")
    await store.initialize()

    # Create two overlapping active threads
    tid_a = await store.upsert_thread("iran-sanctions-a", "Iran Sanctions Push", 8, "active")
    tid_b = await store.upsert_thread("iran-sanctions-b", "Iran Sanctions Drive", 5, "active")
    await store.link_thread_topic(tid_a, "iran-us")
    await store.link_thread_topic(tid_b, "iran-us")

    # Manually set key_entities via event linking + entity resolution
    # Simpler: use the get_active_threads override
    # Actually, get_active_threads fetches key_entities from event_entities.
    # For this test, we'll add events with matching entities.
    ev_a = Event(
        date=date(2026, 3, 10), summary="Iran sanctions tightened",
        entities=["Iran", "IAEA", "US"], sources=[],
    )
    ev_b = Event(
        date=date(2026, 3, 11), summary="Iran nuclear deal stalls",
        entities=["Iran", "IAEA", "EU"], sources=[],
    )
    ids_a = await store.add_events([ev_a], "iran-us")
    ids_b = await store.add_events([ev_b], "iran-us")
    await store.link_thread_events(tid_a, ids_a)
    await store.link_thread_events(tid_b, ids_b)

    # Resolve entities so key_entities show up
    for name in ["Iran", "IAEA", "US", "EU"]:
        eid = await store.upsert_entity(name, "country")
        # Link entities to events
        for ev_id in ids_a + ids_b:
            # Check if this entity name is in the event's entities
            ev_row = await store.db.execute("SELECT id FROM events WHERE id = ?", (ev_id,))
            if await ev_row.fetchone():
                await store.db.execute(
                    "INSERT OR IGNORE INTO event_entities (event_id, entity_id) VALUES (?, ?)",
                    (ev_id, eid),
                )
    await store.db.commit()

    llm = AsyncMock()
    merged = await _consolidate_threads(store, llm, "iran-us")

    # Should have merged the two threads
    assert len(merged) >= 1
    # Higher significance thread (tid_a, sig=8) should be the keeper
    keep_ids = {p[0] for p in merged}
    absorb_ids = {p[1] for p in merged}
    assert tid_a in keep_ids
    assert tid_b in absorb_ids

    # Verify absorbed thread is marked merged
    cursor = await store.db.execute("SELECT status FROM threads WHERE id = ?", (tid_b,))
    assert (await cursor.fetchone())[0] == "merged"

    await store.close()


async def test_persist_threads_reuses_existing_slug_when_headline_changes(tmp_path):
    """Event matches should map synthesis threads back to existing slugs."""
    from nexus.engine.knowledge.store import KnowledgeStore
    import nexus.engine.synthesis.knowledge as knowledge_module

    store = KnowledgeStore(tmp_path / "test.db")
    await store.initialize()

    event = Event(
        date=date(2026, 3, 10),
        summary="Sanctions announced",
        entities=["US", "Iran"],
        sources=[],
        significance=8,
    )
    event_id = (await store.add_events([event], "iran-us"))[0]
    existing_id = await store.upsert_thread("sanctions-escalation", "Sanctions Escalation", 8, "active")
    await store.link_thread_topic(existing_id, "iran-us")
    await store.link_thread_events(existing_id, [event_id])

    synthesis = TopicSynthesis(
        topic_name="Iran-US Relations",
        threads=[
            NarrativeThread(
                headline="US expands sanctions pressure",
                events=[event],
                key_entities=["US", "Iran"],
                significance=8,
            ),
        ],
    )

    original = knowledge_module.match_events_to_threads
    knowledge_module.match_events_to_threads = AsyncMock(return_value=[
        ThreadMatch(event_index=0, thread_slug="sanctions-escalation", is_new_thread=False),
    ])
    try:
        await _persist_threads(store, AsyncMock(), synthesis, [event], "iran-us")
    finally:
        knowledge_module.match_events_to_threads = original

    threads = await store.get_all_threads()
    assert len(threads) == 1
    assert threads[0]["id"] == existing_id
    assert threads[0]["slug"] == "sanctions-escalation"
    assert threads[0]["headline"] == "US expands sanctions pressure"

    await store.close()
