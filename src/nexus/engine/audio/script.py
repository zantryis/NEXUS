"""Dialogue script generation — LLM produces a two-host podcast script."""

import json
import logging
from datetime import date

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
    "You are the script writer for The Nexus Report, a daily intelligence briefing podcast. "
    "Output language: {output_language}.\n"
    "Today's date: {report_date}.\n\n"
    "HOSTS:\n"
    "- Nova (speaker A): The lead anchor. Confident, warm, articulate. Opens and closes the show. "
    "Drives the narrative forward with clarity.\n"
    "- Atlas (speaker B): The analyst. Curious, sharp, a bit dry-humored. Asks clarifying questions, "
    "offers alternative perspectives, occasionally pushes back or expresses genuine surprise. "
    "Adds color and context.\n\n"
    "SHOW STRUCTURE:\n"
    "1. INTRO: Nova opens with something like 'Good morning, welcome to The Nexus Report for "
    "{report_date}. I'm Nova, and as always, Atlas is here with me.' Atlas responds naturally — "
    "maybe a quip about the day's news or a warm greeting. Keep the intro to 2-3 exchanges.\n"
    "2. TOPICS: Cover all topics ordered by significance. Transition naturally between them. "
    "Use phrases like 'Now shifting gears...', 'Speaking of...', etc.\n"
    "3. OUTRO: Brief, warm sign-off. Nova wraps up ('That's your Nexus Report for {report_date}'), "
    "Atlas adds a final thought or teaser. Keep it under 3 exchanges.\n\n"
    "REQUIREMENTS:\n"
    "- Write natural, conversational dialogue — these are real people talking, not reading scripts\n"
    "- The hosts have PERSONALITY. Nova is direct and clear. Atlas is analytical with occasional wit.\n"
    "- Every factual claim MUST verbally attribute its source "
    "(e.g., 'According to Reuters...', 'The BBC is reporting that...')\n"
    "- When sources disagree, have the hosts discuss BOTH framings naturally\n"
    "- Atlas should ask genuine clarifying questions that listeners would also wonder about\n"
    "- Keep each turn to 1-3 sentences for natural pacing\n"
    "- FRESHNESS IS CRITICAL: Focus on what is NEW today. For continuing stories, acknowledge briefly "
    "('As we've been tracking...') then pivot immediately to the latest developments. "
    "Do NOT rehash background details the audience likely heard yesterday. "
    "Lead with surprises, new angles, and breaking developments.\n"
    "- Cover ALL topics provided, ordered by significance\n\n"
    "OUTPUT FORMAT: JSON object with a 'turns' array:\n"
    '{{"turns": [{{"speaker": "A", "text": "..."}}, {{"speaker": "B", "text": "..."}}]}}\n'
)


async def generate_dialogue_script(
    llm: LLMClient,
    config: NexusConfig,
    syntheses: list[TopicSynthesis],
    report_date: date | None = None,
) -> DialogueScript:
    """Generate a two-host dialogue script from TopicSynthesis objects."""
    if report_date is None:
        report_date = date.today()

    date_display = report_date.strftime("%B %d, %Y")
    context = _build_synthesis_context(syntheses)
    system_prompt = DIALOGUE_SYSTEM_PROMPT.format(
        output_language=config.user.output_language,
        report_date=date_display,
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
