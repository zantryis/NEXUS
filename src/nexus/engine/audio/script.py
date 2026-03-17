"""Dialogue script generation — LLM produces a two-host podcast script."""

import json
import logging
import re
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


PODCAST_STYLE_ADDENDUMS = {
    "conversational": (
        "\n\nSTYLE: Keep the dialogue loose and natural — like two colleagues chatting "
        "over coffee. Occasional humor, relaxed pacing, genuine reactions."
    ),
    "analytical": (
        "\n\nSTYLE: Prioritize depth and precision. Both hosts should cite data, "
        "draw comparisons, and explain mechanisms. Less banter, more substance."
    ),
    "energetic": (
        "\n\nSTYLE: High energy, punchy delivery. Short sentences, strong hooks, "
        "dramatic pauses. Open with the most surprising development. "
        "Atlas should express genuine excitement or concern."
    ),
    "formal": (
        "\n\nSTYLE: Professional broadcast tone. Measured pacing, precise language, "
        "minimal humor. Think BBC World Service or NPR — authoritative and polished."
    ),
}


EDITORIAL_DIALOGUE_ADDENDUM = (
    "\n\nEDITORIAL VOICE:\n"
    "Nova and Atlas have strong editorial opinions grounded in international law "
    "and scientific consensus. When covering stories where international norms are "
    "violated or public health is at stake, they don't 'both-sides' it — they state "
    "which position has legal/scientific backing. Atlas in particular should push back "
    "on weak framings. Keep it sharp but never preachy."
)

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


def _strip_json_fences(text: str) -> str:
    """Remove common markdown fences around JSON payloads."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _fallback_dialogue_script(syntheses: list[TopicSynthesis], report_date: date) -> DialogueScript:
    """Create a minimal but usable two-host fallback script."""
    turns: list[DialogueTurn] = [
        DialogueTurn(
            speaker="A",
            text=(
                f"Good morning, welcome to The Nexus Report for {report_date.strftime('%B %d, %Y')}. "
                "I'm Nova, and Atlas is here with me."
            ),
        ),
        DialogueTurn(
            speaker="B",
            text="Morning, Nova. Let's move quickly through the biggest developments shaping today's briefing.",
        ),
    ]

    for syn in syntheses:
        if not syn.threads:
            continue
        thread = syn.threads[0]
        first_event = thread.events[0] if thread.events else None
        source_name = "our source set"
        summary = thread.headline
        if first_event:
            summary = first_event.summary or thread.headline
            if first_event.sources:
                source_name = first_event.sources[0].get("outlet", source_name)
        turns.append(
            DialogueTurn(
                speaker="A",
                text=f"On {syn.topic_name}, according to {source_name}, {summary}.",
            )
        )
        turns.append(
            DialogueTurn(
                speaker="B",
                text=f"That's the key line we're watching in {syn.topic_name.lower()} today.",
            )
        )

    turns.append(
        DialogueTurn(
            speaker="A",
            text=f"That's your Nexus Report for {report_date.strftime('%B %d, %Y')}. Thanks for listening.",
        )
    )
    return DialogueScript(turns=turns)


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
    if config.briefing.style == "editorial":
        system_prompt += EDITORIAL_DIALOGUE_ADDENDUM

    # Apply podcast style addendum
    podcast_style = getattr(config.audio, "podcast_style", "conversational")
    if podcast_style in PODCAST_STYLE_ADDENDUMS:
        system_prompt += PODCAST_STYLE_ADDENDUMS[podcast_style]

    try:
        response = await llm.complete(
            config_key="dialogue_script",
            system_prompt=system_prompt,
            user_prompt=context,
            json_response=True,
        )
        data = json.loads(_strip_json_fences(response))

        # Handle both {"turns": [...]} and bare [...]
        if isinstance(data, list):
            turns_data = data
        else:
            turns_data = data.get("turns", [])

        turns = [DialogueTurn(**t) for t in turns_data]
        if not turns:
            logger.warning("Dialogue script model returned zero turns; using fallback script.")
            return _fallback_dialogue_script(syntheses, report_date)
        return DialogueScript(turns=turns)

    except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
        logger.warning(f"Dialogue script generation failed: {e}")
        return _fallback_dialogue_script(syntheses, report_date)
