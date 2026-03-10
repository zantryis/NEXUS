"""Dialogue script generation — LLM produces a two-host podcast script."""

import json
import logging
from pydantic import BaseModel, Field

from nexus.config.models import NexusConfig
from nexus.engine.synthesis.knowledge import TopicSynthesis
from nexus.engine.synthesis.renderers import _build_synthesis_context
from nexus.llm.client import LLMClient

logger = logging.getLogger(__name__)


class DialogueTurn(BaseModel):
    speaker: str  # "A" or "B"
    text: str


class DialogueScript(BaseModel):
    turns: list[DialogueTurn] = Field(default_factory=list)


DIALOGUE_SYSTEM_PROMPT = (
    "You are a podcast script writer producing a two-host news dialogue. "
    "Output language: {output_language}.\n\n"
    "HOSTS:\n"
    "- Host A: The lead anchor. Introduces topics, provides context, drives the narrative.\n"
    "- Host B: The analyst. Asks clarifying questions, offers alternative perspectives, "
    "occasionally pushes back or expresses surprise.\n\n"
    "REQUIREMENTS:\n"
    "- Write natural, conversational dialogue — not a script reading\n"
    "- Every factual claim MUST verbally attribute its source "
    "(e.g., 'According to Reuters...', 'TASS is reporting that...')\n"
    "- When sources disagree, have the hosts discuss BOTH framings naturally\n"
    "- Include smooth transitions between topics\n"
    "- Host B should ask genuine clarifying questions that add value\n"
    "- Keep each turn to 1-3 sentences for natural pacing\n"
    "- Start with a brief welcome and end with a sign-off\n"
    "- Cover ALL topics provided, ordered by significance\n\n"
    "OUTPUT FORMAT: JSON object with a 'turns' array:\n"
    '{{"turns": [{{"speaker": "A", "text": "..."}}, {{"speaker": "B", "text": "..."}}]}}\n'
)


async def generate_dialogue_script(
    llm: LLMClient,
    config: NexusConfig,
    syntheses: list[TopicSynthesis],
) -> DialogueScript:
    """Generate a two-host dialogue script from TopicSynthesis objects."""
    context = _build_synthesis_context(syntheses)
    system_prompt = DIALOGUE_SYSTEM_PROMPT.format(
        output_language=config.user.output_language,
    )

    try:
        response = await llm.complete(
            config_key="dialogue_script",
            system_prompt=system_prompt,
            user_prompt=context,
            json_response=True,
        )
        data = json.loads(response)

        # Handle both {"turns": [...]} and bare [...]
        if isinstance(data, list):
            turns_data = data
        else:
            turns_data = data.get("turns", [])

        turns = [DialogueTurn(**t) for t in turns_data]
        return DialogueScript(turns=turns)

    except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
        logger.warning(f"Dialogue script generation failed: {e}")
        # Fallback: produce a minimal script from synthesis context
        fallback_text = "Here's a summary of today's developments. " + " ".join(
            s.topic_name for s in syntheses
        )
        return DialogueScript(turns=[
            DialogueTurn(speaker="A", text=fallback_text),
        ])
