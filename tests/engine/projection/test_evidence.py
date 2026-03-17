"""Tests for evidence assembly layer."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from unittest.mock import AsyncMock


from nexus.engine.projection.evidence import (
    EvidencePackage,
    _word_match,
    assemble_evidence_package,
    format_evidence_section,
)


@dataclass
class FakeEvent:
    event_id: int
    date: date
    summary: str
    significance: int = 5
    entities: list = None
    sources: list = None
    relation_to_prior: str = ""

    def __post_init__(self):
        if self.entities is None:
            self.entities = []
        if self.sources is None:
            self.sources = []


def _make_store(
    entities=None,
    entity_events=None,
    threads=None,
    thread_snapshot=None,
    convergence=None,
    divergence=None,
    causal_links=None,
    relationships=None,
    relationship_timeline=None,
    cross_topic_signals=None,
):
    """Build a mock KnowledgeStore with configurable returns."""
    store = AsyncMock()
    store.get_all_entities = AsyncMock(return_value=entities or [])
    store.get_events_for_entity = AsyncMock(return_value=entity_events or [])
    store.get_threads_for_entity = AsyncMock(return_value=threads or [])
    store.get_thread_snapshot_as_of = AsyncMock(return_value=thread_snapshot)
    store.get_convergence_for_thread = AsyncMock(return_value=convergence or [])
    store.get_divergence_for_thread = AsyncMock(return_value=divergence or [])
    store.get_causal_links_for_thread = AsyncMock(return_value=causal_links or [])
    store.get_active_relationships_for_entity = AsyncMock(
        return_value=relationships or []
    )
    store.get_relationship_timeline = AsyncMock(
        return_value=relationship_timeline or []
    )
    store.get_cross_topic_signals_as_of = AsyncMock(
        return_value=cross_topic_signals or []
    )
    return store


class TestAssembleEvidencePackage:
    async def test_returns_evidence_package(self):
        """Should return an EvidencePackage with correct question and date."""
        store = _make_store()
        pkg = await assemble_evidence_package(store, "Will X happen?", as_of=date(2026, 3, 15))
        assert isinstance(pkg, EvidencePackage)
        assert pkg.question == "Will X happen?"
        assert pkg.as_of == date(2026, 3, 15)

    async def test_resolves_entities_from_question(self):
        """Should find entities whose canonical names appear in the question."""
        store = _make_store(entities=[
            {"id": 1, "canonical_name": "Iran", "entity_type": "country"},
            {"id": 2, "canonical_name": "OpenAI", "entity_type": "org"},
            {"id": 3, "canonical_name": "Brazil", "entity_type": "country"},
        ])
        pkg = await assemble_evidence_package(
            store, "Will Iran develop nuclear weapons?", as_of=date(2026, 3, 15),
        )
        entity_names = [e["name"] for e in pkg.entities]
        assert "Iran" in entity_names
        assert "OpenAI" not in entity_names

    async def test_filters_events_by_as_of_date(self):
        """Events after as_of must be excluded to prevent leakage."""
        cutoff = date(2026, 3, 10)
        future_event = FakeEvent(event_id=1, date=date(2026, 3, 12), summary="Future event")
        past_event = FakeEvent(event_id=2, date=date(2026, 3, 8), summary="Past event")

        store = _make_store(
            entities=[{"id": 1, "canonical_name": "Iran", "entity_type": "country"}],
            entity_events=[future_event, past_event],
        )
        pkg = await assemble_evidence_package(
            store, "Will Iran do X?", as_of=cutoff,
        )
        event_summaries = [e.get("summary", e.summary if hasattr(e, "summary") else "") for e in pkg.recent_events]
        assert "Past event" in event_summaries
        assert "Future event" not in event_summaries

    async def test_calls_leakage_safe_methods(self):
        """Store methods that accept as_of/cutoff must be called with the right date."""
        cutoff = date(2026, 3, 10)
        store = _make_store(
            entities=[{"id": 1, "canonical_name": "Iran", "entity_type": "country"}],
            threads=[{"id": 10, "slug": "iran-thread", "headline": "Iran tensions"}],
        )
        await assemble_evidence_package(store, "Will Iran do X?", as_of=cutoff)

        # Relationships must use as_of
        store.get_active_relationships_for_entity.assert_called()
        call_kwargs = store.get_active_relationships_for_entity.call_args
        assert call_kwargs.kwargs.get("as_of") == cutoff or call_kwargs[1].get("as_of") == cutoff

        # Thread snapshot must use cutoff
        store.get_thread_snapshot_as_of.assert_called()
        snapshot_args = store.get_thread_snapshot_as_of.call_args
        assert cutoff in snapshot_args.args or cutoff in snapshot_args.kwargs.values()

    async def test_respects_caps(self):
        """Evidence should be capped to prevent prompt overflow."""
        # Create 20 events — should be capped to 15
        many_events = [
            FakeEvent(event_id=i, date=date(2026, 3, 1), summary=f"Event {i}")
            for i in range(20)
        ]
        store = _make_store(
            entities=[{"id": 1, "canonical_name": "Iran", "entity_type": "country"}],
            entity_events=many_events,
        )
        pkg = await assemble_evidence_package(store, "Will Iran do X?", as_of=date(2026, 3, 15))
        assert len(pkg.recent_events) <= 15

    async def test_empty_store_returns_empty_coverage(self):
        """With no entities found, coverage stats should reflect emptiness."""
        store = _make_store()
        pkg = await assemble_evidence_package(store, "Will X happen?", as_of=date(2026, 3, 15))
        assert pkg.coverage["entities_found"] == 0
        assert pkg.coverage["threads_found"] == 0
        assert len(pkg.recent_events) == 0

    async def test_coverage_stats_populated(self):
        """Coverage dict should track what evidence was found."""
        store = _make_store(
            entities=[
                {"id": 1, "canonical_name": "OpenAI", "entity_type": "org"},
            ],
            entity_events=[
                FakeEvent(event_id=1, date=date(2026, 3, 5), summary="Event 1"),
            ],
            threads=[{"id": 10, "slug": "ai-thread", "headline": "AI developments"}],
            convergence=[{"fact_text": "AI is advancing", "confirmed_by": ["NYT", "BBC"]}],
        )
        pkg = await assemble_evidence_package(
            store, "Will OpenAI release GPT-5?", as_of=date(2026, 3, 15),
        )
        assert pkg.coverage["entities_found"] == 1
        assert pkg.coverage["threads_found"] == 1
        assert pkg.coverage["events_found"] >= 1
        assert pkg.coverage["convergence_found"] >= 1


class TestFormatEvidenceSection:
    def test_formats_empty_package(self):
        """Empty evidence should produce readable 'no evidence' sections."""
        pkg = EvidencePackage(
            question="Will X?",
            as_of=date(2026, 3, 15),
            entities=[],
            threads=[],
            convergence=[],
            divergence=[],
            causal_chains=[],
            relationships=[],
            relationship_changes=[],
            cross_topic_signals=[],
            recent_events=[],
            coverage={"entities_found": 0, "threads_found": 0},
        )
        sections = format_evidence_section(pkg)
        assert isinstance(sections, str)
        assert "No monitored intelligence" in sections or "no evidence" in sections.lower()

    def test_formats_with_evidence(self):
        """Evidence sections should include thread trajectories, events, etc."""
        pkg = EvidencePackage(
            question="Will Iran do X?",
            as_of=date(2026, 3, 15),
            entities=[{"name": "Iran", "entity_id": 1}],
            threads=[{
                "headline": "Iran nuclear talks",
                "trajectory_label": "accelerating",
                "momentum_score": 45.0,
                "velocity_7d": 3.0,
            }],
            convergence=[{
                "fact_text": "Talks resumed last week",
                "confirmed_by": ["NYT", "BBC"],
            }],
            divergence=[],
            causal_chains=[],
            relationships=[{
                "source_entity_name": "Iran",
                "target_entity_name": "US",
                "relation_type": "diplomatic_tension",
            }],
            relationship_changes=[],
            cross_topic_signals=[],
            recent_events=[{
                "date": "2026-03-10",
                "summary": "Iran resumed enrichment",
                "significance": 8,
            }],
            coverage={"entities_found": 1, "threads_found": 1,
                      "events_found": 1, "convergence_found": 1},
        )
        sections = format_evidence_section(pkg)
        assert "accelerating" in sections
        assert "Iran nuclear talks" in sections
        assert "Talks resumed" in sections
        assert "Iran resumed enrichment" in sections

    def test_temporal_markers_in_events(self):
        """Recent events should include '5d ago' style markers."""
        pkg = EvidencePackage(
            question="Will Iran do X?",
            as_of=date(2026, 3, 15),
            entities=[], threads=[], convergence=[], divergence=[],
            causal_chains=[], relationships=[], relationship_changes=[],
            cross_topic_signals=[],
            recent_events=[
                {"date": "2026-03-15", "summary": "Today event", "significance": 8},
                {"date": "2026-03-10", "summary": "Older event", "significance": 6},
            ],
            coverage={"entities_found": 0, "threads_found": 0, "events_found": 2},
        )
        sections = format_evidence_section(pkg)
        assert "today" in sections.lower()
        assert "5d ago" in sections


class TestWordMatch:
    def test_exact_word(self):
        assert _word_match("Iran", "Will Iran develop nuclear weapons?")

    def test_no_substring_false_positive(self):
        assert not _word_match("US", "We must discuss this issue")
        assert not _word_match("EU", "The queue is long")

    def test_short_entity_at_boundary(self):
        assert _word_match("US", "The US imposed sanctions")
        assert _word_match("EU", "EU trade policy is changing")

    def test_case_insensitive(self):
        assert _word_match("iran", "Will IRAN do X?")
        assert _word_match("IRAN", "iran is important")

    def test_multi_word_entity(self):
        assert _word_match("United States", "The United States of America")
        assert not _word_match("United States", "United Kingdom states")
