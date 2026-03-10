"""Tests for LLM-as-judge evaluation."""

import json
import pytest
from unittest.mock import AsyncMock
from nexus.engine.evaluation.judge import judge_synthesis, _format_synthesis_for_judge
from nexus.engine.synthesis.knowledge import TopicSynthesis, NarrativeThread


@pytest.fixture
def synthesis():
    return TopicSynthesis(
        topic_name="Iran-US Relations",
        threads=[
            NarrativeThread(
                headline="Sanctions escalation",
                convergence=["New sanctions announced"],
                divergence=[{
                    "claim": "Impact assessment",
                    "source_a": "nyt", "framing_a": "Targeted response",
                    "source_b": "tass", "framing_b": "Economic warfare",
                }],
                key_entities=["US", "Iran"],
                significance=8,
            ),
        ],
        source_balance={"private": 3, "state": 2},
        languages_represented=["en", "fa"],
    )


def test_format_synthesis(synthesis):
    text = _format_synthesis_for_judge(synthesis)
    assert "Iran-US Relations" in text
    assert "Sanctions escalation" in text
    assert "private" in text
    assert "nyt" in text


@pytest.mark.asyncio
async def test_judge_synthesis(synthesis):
    mock_llm = AsyncMock()
    mock_llm.complete.return_value = json.dumps({
        "completeness": 7,
        "source_balance": 8,
        "convergence_accuracy": 6,
        "divergence_detection": 7,
        "entity_coverage": 5,
        "overall": 6.6,
        "strengths": ["Good source diversity"],
        "weaknesses": ["Entity coverage could be better"],
        "suggestions": ["Add more entities"],
    })

    scores = await judge_synthesis(mock_llm, synthesis)
    assert scores["completeness"] == 7
    assert scores["overall"] == 6.6
    assert len(scores["strengths"]) == 1


@pytest.mark.asyncio
async def test_judge_synthesis_bad_json(synthesis):
    mock_llm = AsyncMock()
    mock_llm.complete.return_value = "not json"

    scores = await judge_synthesis(mock_llm, synthesis)
    assert "error" in scores
