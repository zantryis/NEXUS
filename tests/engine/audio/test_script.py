"""Tests for audio dialogue script generation."""

import json
from unittest.mock import AsyncMock
from datetime import date

from nexus.config.models import NexusConfig, UserConfig, AudioConfig
from nexus.engine.audio.script import (
    DialogueTurn,
    DialogueScript,
    generate_dialogue_script,
)
from nexus.engine.synthesis.knowledge import TopicSynthesis, NarrativeThread
from nexus.engine.knowledge.events import Event


def _make_config(**audio_kw) -> NexusConfig:
    return NexusConfig(
        user=UserConfig(name="Tristan", output_language="en"),
        audio=AudioConfig(**audio_kw),
    )


def _make_synthesis() -> list[TopicSynthesis]:
    return [
        TopicSynthesis(
            topic_name="Iran-US Relations",
            threads=[
                NarrativeThread(
                    headline="Sanctions Escalation",
                    events=[
                        Event(
                            date=date(2026, 3, 10),
                            summary="US imposes new sanctions on Iran",
                            significance=8,
                            entities=["US", "Iran"],
                            sources=[{"outlet": "reuters", "affiliation": "private",
                                      "country": "US", "language": "en"}],
                        ),
                    ],
                    convergence=[{"fact": "Sanctions target oil", "confirmed_by": ["reuters", "bbc"]}],
                    divergence=[{
                        "shared_event": "Impact",
                        "source_a": "reuters", "framing_a": "Devastating",
                        "source_b": "tass", "framing_b": "Minimal",
                    }],
                    key_entities=["US", "Iran", "Treasury"],
                    significance=8,
                ),
            ],
            source_balance={"private": 2, "state": 1},
            languages_represented=["en", "fa"],
        ),
    ]


def test_dialogue_turn_model():
    turn = DialogueTurn(speaker="A", text="Hello there!")
    assert turn.speaker == "A"
    assert turn.text == "Hello there!"


def test_dialogue_script_model():
    script = DialogueScript(turns=[
        DialogueTurn(speaker="A", text="Welcome."),
        DialogueTurn(speaker="B", text="Thanks."),
    ])
    assert len(script.turns) == 2
    assert script.turns[0].speaker == "A"


async def test_generate_dialogue_script():
    """Test LLM call returns parsed DialogueScript."""
    mock_response = json.dumps({
        "turns": [
            {"speaker": "A", "text": "Welcome to today's briefing."},
            {"speaker": "B", "text": "Let's dive into Iran sanctions."},
            {"speaker": "A", "text": "Reuters reports devastating impact."},
            {"speaker": "B", "text": "But TASS says minimal effect, right?"},
        ]
    })

    llm = AsyncMock()
    llm.complete = AsyncMock(return_value=mock_response)

    config = _make_config()
    syntheses = _make_synthesis()

    script = await generate_dialogue_script(llm, config, syntheses, report_date=date(2026, 3, 10))

    assert isinstance(script, DialogueScript)
    assert len(script.turns) == 4
    assert script.turns[0].speaker == "A"
    assert script.turns[3].speaker == "B"
    llm.complete.assert_called_once()
    call_kwargs = llm.complete.call_args
    assert call_kwargs.kwargs["config_key"] == "dialogue_script"
    assert call_kwargs.kwargs["json_response"] is True
    # Verify date and host names are in the system prompt
    system_prompt = call_kwargs.kwargs["system_prompt"]
    assert "March 10, 2026" in system_prompt
    assert "Nova" in system_prompt
    assert "Atlas" in system_prompt


async def test_generate_dialogue_script_includes_context():
    """Verify the user prompt contains synthesis context."""
    mock_response = json.dumps({"turns": [{"speaker": "A", "text": "Hi."}]})
    llm = AsyncMock()
    llm.complete = AsyncMock(return_value=mock_response)

    config = _make_config()
    syntheses = _make_synthesis()

    await generate_dialogue_script(llm, config, syntheses)

    user_prompt = llm.complete.call_args.kwargs["user_prompt"]
    assert "Iran-US Relations" in user_prompt
    assert "Sanctions Escalation" in user_prompt


async def test_generate_dialogue_script_fallback_on_bad_json():
    """Bad JSON from LLM should produce a fallback single-turn script."""
    llm = AsyncMock()
    llm.complete = AsyncMock(return_value="not valid json {{{")

    config = _make_config()
    syntheses = _make_synthesis()

    script = await generate_dialogue_script(llm, config, syntheses)
    assert isinstance(script, DialogueScript)
    assert len(script.turns) >= 1
