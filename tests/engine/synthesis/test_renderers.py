"""Tests for briefing renderers (synthesis → prompt → LLM → markdown)."""

import pytest
from datetime import date
from unittest.mock import AsyncMock

from nexus.config.models import NexusConfig, UserConfig, BriefingConfig
from nexus.engine.knowledge.compression import Summary
from nexus.engine.knowledge.events import Event
from nexus.engine.projection.models import (
    CrossTopicSignal, TopicProjection, ProjectionItem,
)
from nexus.engine.synthesis.knowledge import TopicSynthesis, NarrativeThread
from nexus.engine.synthesis.renderers import (
    _build_synthesis_context,
    render_text_briefing,
    EDITORIAL_STANCE,
)


# ── Fixtures ──


def _make_thread(
    headline: str = "Test Thread",
    significance: int = 7,
    events: list[Event] | None = None,
    convergence: list | None = None,
    divergence: list | None = None,
    key_entities: list[str] | None = None,
) -> NarrativeThread:
    return NarrativeThread(
        headline=headline,
        significance=significance,
        events=events or [],
        convergence=convergence or [],
        divergence=divergence or [],
        key_entities=key_entities or [],
    )


def _make_synthesis(
    topic_name: str = "AI Research",
    threads: list[NarrativeThread] | None = None,
    background: list[Summary] | None = None,
    source_balance: dict | None = None,
    languages: list[str] | None = None,
    cross_topic_signals: list[CrossTopicSignal] | None = None,
    projection: TopicProjection | None = None,
) -> TopicSynthesis:
    return TopicSynthesis(
        topic_name=topic_name,
        threads=threads or [],
        background=background or [],
        source_balance=source_balance or {},
        languages_represented=languages or [],
        cross_topic_signals=cross_topic_signals or [],
        projection=projection,
    )


def _make_config(style: str = "analytical", depth: str = "detailed", lang: str = "en") -> NexusConfig:
    return NexusConfig(
        user=UserConfig(name="Tristan", timezone="America/Denver", output_language=lang),
        briefing=BriefingConfig(style=style, depth=depth),
    )


# ── _build_synthesis_context ──


def test_build_context_basic_topic():
    syn = _make_synthesis(topic_name="Iran-US Relations")
    ctx = _build_synthesis_context([syn])
    assert "### Topic: Iran-US Relations" in ctx


def test_build_context_with_background():
    syn = _make_synthesis(
        background=[
            Summary(
                period_start=date(2026, 3, 1),
                period_end=date(2026, 3, 7),
                text="Tensions escalated over sanctions.",
                event_count=5,
            ),
        ],
    )
    ctx = _build_synthesis_context([syn])
    assert "Background context" in ctx
    assert "Tensions escalated" in ctx
    assert "2026-03-01" in ctx


def test_build_context_limits_background_to_3():
    """Only the last 3 background summaries should appear."""
    summaries = [
        Summary(period_start=date(2026, 2, i), period_end=date(2026, 2, i + 6),
                text=f"Week {i}", event_count=1)
        for i in range(1, 8, 7)  # 1 summary
    ]
    # Add 5 summaries
    summaries = [
        Summary(period_start=date(2026, i, 1), period_end=date(2026, i, 7),
                text=f"Month {i} summary", event_count=i)
        for i in range(1, 6)
    ]
    syn = _make_synthesis(background=summaries)
    ctx = _build_synthesis_context([syn])
    # Only last 3 should appear
    assert "Month 3 summary" in ctx
    assert "Month 4 summary" in ctx
    assert "Month 5 summary" in ctx
    assert "Month 1 summary" not in ctx
    assert "Month 2 summary" not in ctx


def test_build_context_with_threads_and_events():
    thread = _make_thread(
        headline="AI Safety Summit",
        significance=9,
        events=[
            Event(
                date=date(2026, 3, 10),
                summary="Global AI safety summit convened",
                sources=[
                    {"outlet": "Reuters", "affiliation": "private", "country": "GB"},
                    {"outlet": "Xinhua", "affiliation": "state", "country": "CN"},
                ],
                entities=["UN", "OpenAI"],
                significance=9,
            ),
        ],
        key_entities=["UN", "OpenAI", "EU"],
    )
    syn = _make_synthesis(threads=[thread])
    ctx = _build_synthesis_context([syn])
    assert "AI Safety Summit" in ctx
    assert "significance: 9" in ctx
    assert "Global AI safety summit convened" in ctx
    assert "Reuters (private/GB)" in ctx
    assert "Xinhua (state/CN)" in ctx
    assert "UN, OpenAI, EU" in ctx


def test_build_context_with_convergence():
    thread = _make_thread(
        convergence=[
            {"fact": "Summit date confirmed", "confirmed_by": ["Reuters", "AP"]},
            "Simple string convergence",
        ],
    )
    syn = _make_synthesis(threads=[thread])
    ctx = _build_synthesis_context([syn])
    assert "Summit date confirmed" in ctx
    assert "Reuters, AP" in ctx
    assert "Simple string convergence" in ctx


def test_build_context_with_divergence():
    thread = _make_thread(
        divergence=[
            {
                "shared_event": "Military exercise",
                "source_a": "CNN",
                "framing_a": "Defensive measure",
                "source_b": "RT",
                "framing_b": "Provocative escalation",
            },
        ],
    )
    syn = _make_synthesis(threads=[thread])
    ctx = _build_synthesis_context([syn])
    assert "Military exercise" in ctx
    assert 'CNN says "Defensive measure"' in ctx
    assert 'RT says "Provocative escalation"' in ctx


def test_build_context_with_source_balance_and_languages():
    syn = _make_synthesis(
        source_balance={"state": 3, "public": 5, "private": 8},
        languages=["en", "fa", "zh"],
    )
    ctx = _build_synthesis_context([syn])
    assert "Source balance:" in ctx
    assert "Languages: en, fa, zh" in ctx


def test_build_context_with_cross_topic_signals():
    signal = CrossTopicSignal(
        topic_slug="iran-us",
        related_topic_slug="energy-transition",
        shared_entity="OPEC",
        observed_at=date(2026, 3, 10),
    )
    syn = _make_synthesis(cross_topic_signals=[signal])
    ctx = _build_synthesis_context([syn])
    assert "OPEC" in ctx
    assert "energy-transition" in ctx


def test_build_context_with_projection():
    proj = TopicProjection(
        topic_slug="iran-us",
        topic_name="Iran-US Relations",
        generated_for=date(2026, 3, 10),
        status="ready",
        items=[
            ProjectionItem(
                claim="Sanctions likely to tighten",
                confidence="high",
                horizon_days=7,
                signpost="UN vote scheduled",
                review_after=date(2026, 3, 17),
            ),
        ],
    )
    syn = _make_synthesis(projection=proj)
    ctx = _build_synthesis_context([syn])
    assert "Forward Look status: ready" in ctx
    assert "Sanctions likely to tighten" in ctx
    assert "confidence: high" in ctx
    assert "horizon: 7d" in ctx


def test_build_context_multiple_topics():
    syn1 = _make_synthesis(topic_name="AI Research")
    syn2 = _make_synthesis(topic_name="Iran-US Relations")
    ctx = _build_synthesis_context([syn1, syn2])
    assert "AI Research" in ctx
    assert "Iran-US Relations" in ctx
    assert "---" in ctx  # separator


def test_build_context_empty_synthesis():
    """Empty synthesis should still produce a topic header."""
    syn = _make_synthesis(topic_name="Empty Topic")
    ctx = _build_synthesis_context([syn])
    assert "### Topic: Empty Topic" in ctx


# ── render_text_briefing ──


@pytest.mark.asyncio
async def test_render_briefing_calls_llm_with_correct_config():
    mock_llm = AsyncMock()
    mock_llm.complete.return_value = "# Briefing\n\nContent here."

    config = _make_config(style="analytical", depth="detailed", lang="en")
    syn = _make_synthesis(topic_name="AI Research")

    result = await render_text_briefing(mock_llm, config, [syn])

    assert result == "# Briefing\n\nContent here."
    mock_llm.complete.assert_called_once()

    call_kwargs = mock_llm.complete.call_args.kwargs
    assert call_kwargs["config_key"] == "synthesis"
    assert "en" in call_kwargs["system_prompt"]
    assert "analytical" in call_kwargs["system_prompt"]
    assert "detailed" in call_kwargs["system_prompt"]
    assert "AI Research" in call_kwargs["user_prompt"]


@pytest.mark.asyncio
async def test_render_briefing_editorial_appends_stance():
    mock_llm = AsyncMock()
    mock_llm.complete.return_value = "# Editorial Briefing"

    config = _make_config(style="editorial")
    syn = _make_synthesis()

    await render_text_briefing(mock_llm, config, [syn])

    system_prompt = mock_llm.complete.call_args.kwargs["system_prompt"]
    assert "opinionated analyst" in system_prompt
    assert "international law" in system_prompt.lower()


@pytest.mark.asyncio
async def test_render_briefing_non_editorial_no_stance():
    mock_llm = AsyncMock()
    mock_llm.complete.return_value = "# Briefing"

    config = _make_config(style="analytical")
    syn = _make_synthesis()

    await render_text_briefing(mock_llm, config, [syn])

    system_prompt = mock_llm.complete.call_args.kwargs["system_prompt"]
    assert "opinionated analyst" not in system_prompt


@pytest.mark.asyncio
async def test_render_briefing_respects_output_language():
    mock_llm = AsyncMock()
    mock_llm.complete.return_value = "# Briefing en Español"

    config = _make_config(lang="es")
    syn = _make_synthesis()

    await render_text_briefing(mock_llm, config, [syn])

    system_prompt = mock_llm.complete.call_args.kwargs["system_prompt"]
    assert "es" in system_prompt


@pytest.mark.asyncio
async def test_render_briefing_with_market_divergences():
    mock_llm = AsyncMock()
    mock_llm.complete.return_value = "# Briefing with divergences"

    config = _make_config()
    syn = _make_synthesis()

    divergences = [{"market": "Iran sanctions", "engine_prob": 0.7, "market_prob": 0.3}]

    from unittest.mock import patch
    with patch("nexus.engine.projection.kalshi_matcher.render_divergence_section") as mock_div:
        mock_div.return_value = "## Market Divergences\n\nIran sanctions: engine 0.7 vs market 0.3"

        await render_text_briefing(mock_llm, config, [syn], market_divergences=divergences)

    user_prompt = mock_llm.complete.call_args.kwargs["user_prompt"]
    assert "Market Divergences" in user_prompt


@pytest.mark.asyncio
async def test_render_briefing_no_divergences_when_none():
    mock_llm = AsyncMock()
    mock_llm.complete.return_value = "# Clean Briefing"

    config = _make_config()
    syn = _make_synthesis()

    await render_text_briefing(mock_llm, config, [syn], market_divergences=None)

    user_prompt = mock_llm.complete.call_args.kwargs["user_prompt"]
    assert "Market Divergence" not in user_prompt
