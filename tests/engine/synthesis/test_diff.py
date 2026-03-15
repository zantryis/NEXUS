"""Tests for TopicSynthesis diff computation."""

from datetime import date

from nexus.engine.knowledge.events import Event
from nexus.engine.synthesis.diff import diff_syntheses, is_empty_diff
from nexus.engine.synthesis.knowledge import NarrativeThread, TopicSynthesis


def _make_event(summary="Event A", significance=5, entities=None):
    return Event(
        date=date(2026, 3, 10),
        summary=summary,
        sources=[{"outlet": "test"}],
        entities=entities or [],
        relation_to_prior="",
        significance=significance,
    )


def _make_thread(headline, events=None, convergence=None, divergence=None,
                 key_entities=None):
    return NarrativeThread(
        headline=headline,
        events=events or [],
        convergence=convergence or [],
        divergence=divergence or [],
        key_entities=key_entities or [],
    )


def _make_synthesis(threads=None, source_balance=None):
    return TopicSynthesis(
        topic_name="Test",
        threads=threads or [],
        source_balance=source_balance or {},
    )


def test_new_threads():
    prev = _make_synthesis(threads=[_make_thread("Old Thread")])
    curr = _make_synthesis(threads=[
        _make_thread("Old Thread"),
        _make_thread("Brand New Thread"),
    ])

    diff = diff_syntheses(curr, prev)
    assert len(diff["new_threads"]) == 1
    assert diff["new_threads"][0].headline == "Brand New Thread"
    assert diff["updated_threads"] == []
    assert diff["resolved_threads"] == []


def test_updated_threads_new_events():
    ev1 = _make_event("First event")
    ev2 = _make_event("Second event")

    prev = _make_synthesis(threads=[_make_thread("Thread A", events=[ev1])])
    curr = _make_synthesis(threads=[_make_thread("Thread A", events=[ev1, ev2])])

    diff = diff_syntheses(curr, prev)
    assert len(diff["updated_threads"]) == 1
    update = diff["updated_threads"][0]
    assert update["thread"].headline == "Thread A"
    assert len(update["new_events"]) == 1
    assert update["new_events"][0].summary == "Second event"
    assert update["prev_event_count"] == 1


def test_updated_threads_no_new_events():
    """Same events in both — thread should NOT appear in updated_threads."""
    ev1 = _make_event("Same event")
    prev = _make_synthesis(threads=[_make_thread("Thread A", events=[ev1])])
    curr = _make_synthesis(threads=[_make_thread("Thread A", events=[ev1])])

    diff = diff_syntheses(curr, prev)
    assert diff["updated_threads"] == []


def test_resolved_threads():
    prev = _make_synthesis(threads=[
        _make_thread("Active Thread"),
        _make_thread("Gone Thread"),
    ])
    curr = _make_synthesis(threads=[_make_thread("Active Thread")])

    diff = diff_syntheses(curr, prev)
    assert "Gone Thread" in diff["resolved_threads"]
    assert len(diff["resolved_threads"]) == 1


def test_new_convergence_in_existing_thread():
    conv1 = {"fact": "Both agree on X", "confirmed_by": ["A", "B"]}
    conv2 = {"fact": "New agreement on Y", "confirmed_by": ["C", "D"]}

    prev = _make_synthesis(threads=[_make_thread("Thread", convergence=[conv1])])
    curr = _make_synthesis(threads=[_make_thread("Thread", convergence=[conv1, conv2])])

    diff = diff_syntheses(curr, prev)
    assert len(diff["new_convergence"]) == 1
    assert diff["new_convergence"][0]["thread_headline"] == "Thread"
    assert len(diff["new_convergence"][0]["facts"]) == 1
    assert diff["new_convergence"][0]["facts"][0]["fact"] == "New agreement on Y"


def test_new_convergence_in_new_thread():
    conv = {"fact": "Fresh fact", "confirmed_by": ["A", "B"]}
    prev = _make_synthesis()
    curr = _make_synthesis(threads=[_make_thread("New Thread", convergence=[conv])])

    diff = diff_syntheses(curr, prev)
    assert len(diff["new_convergence"]) == 1
    assert diff["new_convergence"][0]["thread_headline"] == "New Thread"


def test_new_divergence_in_existing_thread():
    div1 = {"shared_event": "Event X", "source_a": "A", "framing_a": "fa",
            "source_b": "B", "framing_b": "fb"}
    div2 = {"shared_event": "Event Y", "source_a": "C", "framing_a": "fc",
            "source_b": "D", "framing_b": "fd"}

    prev = _make_synthesis(threads=[_make_thread("Thread", divergence=[div1])])
    curr = _make_synthesis(threads=[_make_thread("Thread", divergence=[div1, div2])])

    diff = diff_syntheses(curr, prev)
    assert len(diff["new_divergence"]) == 1
    assert len(diff["new_divergence"][0]["items"]) == 1
    assert diff["new_divergence"][0]["items"][0]["shared_event"] == "Event Y"


def test_new_divergence_in_new_thread():
    div = {"shared_event": "Event Z", "source_a": "A", "framing_a": "fa",
           "source_b": "B", "framing_b": "fb"}
    prev = _make_synthesis()
    curr = _make_synthesis(threads=[_make_thread("New Thread", divergence=[div])])

    diff = diff_syntheses(curr, prev)
    assert len(diff["new_divergence"]) == 1
    assert diff["new_divergence"][0]["thread_headline"] == "New Thread"


def test_new_entities():
    prev = _make_synthesis(threads=[
        _make_thread("Thread", key_entities=["Alice", "Bob"]),
    ])
    curr = _make_synthesis(threads=[
        _make_thread("Thread", key_entities=["Alice", "Bob", "Charlie"]),
    ])

    diff = diff_syntheses(curr, prev)
    assert diff["new_entities"] == ["Charlie"]


def test_new_entities_from_new_thread():
    prev = _make_synthesis()
    curr = _make_synthesis(threads=[
        _make_thread("New", key_entities=["Zara", "Adam"]),
    ])

    diff = diff_syntheses(curr, prev)
    assert sorted(diff["new_entities"]) == ["Adam", "Zara"]


def test_source_balance_shift():
    prev = _make_synthesis(source_balance={"state": 5, "private": 3})
    curr = _make_synthesis(source_balance={"state": 3, "private": 5, "public": 2})

    diff = diff_syntheses(curr, prev)
    assert diff["source_balance_shift"]["state"] == -2
    assert diff["source_balance_shift"]["private"] == 2
    assert diff["source_balance_shift"]["public"] == 2


def test_source_balance_shift_removed_affiliation():
    prev = _make_synthesis(source_balance={"state": 4})
    curr = _make_synthesis(source_balance={})

    diff = diff_syntheses(curr, prev)
    assert diff["source_balance_shift"]["state"] == -4


def test_is_empty_diff_true():
    diff = diff_syntheses(_make_synthesis(), _make_synthesis())
    assert is_empty_diff(diff) is True


def test_is_empty_diff_false_new_threads():
    prev = _make_synthesis()
    curr = _make_synthesis(threads=[_make_thread("New")])
    diff = diff_syntheses(curr, prev)
    assert is_empty_diff(diff) is False


def test_is_empty_diff_false_resolved_threads():
    prev = _make_synthesis(threads=[_make_thread("Gone")])
    curr = _make_synthesis()
    diff = diff_syntheses(curr, prev)
    assert is_empty_diff(diff) is False


def test_is_empty_diff_ignores_source_balance():
    """Source balance shift alone does not make the diff non-empty."""
    prev = _make_synthesis(source_balance={"state": 5})
    curr = _make_synthesis(source_balance={"state": 10})
    diff = diff_syntheses(curr, prev)
    assert is_empty_diff(diff) is True
    assert diff["source_balance_shift"] == {"state": 5}


def test_empty_syntheses():
    diff = diff_syntheses(_make_synthesis(), _make_synthesis())
    assert diff["new_threads"] == []
    assert diff["updated_threads"] == []
    assert diff["resolved_threads"] == []
    assert diff["new_convergence"] == []
    assert diff["new_divergence"] == []
    assert diff["new_entities"] == []
    assert diff["source_balance_shift"] == {}


def test_headline_normalization():
    """Headline matching is case-insensitive and whitespace-stripped."""
    prev = _make_synthesis(threads=[_make_thread("  Some Thread  ")])
    curr = _make_synthesis(threads=[_make_thread("some thread")])

    diff = diff_syntheses(curr, prev)
    assert diff["new_threads"] == []
    assert diff["resolved_threads"] == []
