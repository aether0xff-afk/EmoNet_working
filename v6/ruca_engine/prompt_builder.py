from __future__ import annotations

from .models import CharacterProfile, EmotionState, InnerVoiceCandidate, MemoryItem, SpontaneousReactionDecision
from .context import TurnContext


def build_response_prompt(
    *,
    user_text: str,
    ruca: CharacterProfile,
    emotion_state: EmotionState,
    turn_context: TurnContext,
    memories: tuple[MemoryItem, ...],
    voices: tuple[InnerVoiceCandidate, ...],
    spontaneous: SpontaneousReactionDecision,
) -> str:
    memory_lines = "\n".join(f"- {item.memory_type}: {item.summary}" for item in memories) or "- 없음"
    voice_lines = "\n".join(
        f"- {voice.source_character}: {voice.content} / action={voice.recommended_action}"
        for voice in voices
    )
    return f"""[ROLE]
You are composing the final external response for {ruca.name}.
Return only {ruca.name}'s user-facing Korean utterance. Do not expose internal labels, JSON, trace names, or module names.

[CHARACTER]
role: {ruca.role}
tone_style: {ruca.tone_style}
relationship_state: {ruca.relationship_state}

[USER_INPUT]
{user_text}

[TURN_CONTEXT]
event_type: {turn_context.event_type}
user_position: {turn_context.user_position}
rookie_question: {turn_context.rookie_question}
unresolved_need: {turn_context.unresolved_need}

[EMOTION_STATE]
valence={emotion_state.valence:.3f}
arousal={emotion_state.arousal:.3f}
affinity={emotion_state.affinity:.3f}
stability={emotion_state.stability:.3f}
protective_tension={emotion_state.protective_tension:.3f}
curiosity={emotion_state.curiosity:.3f}

[RELEVANT_MEMORY]
{memory_lines}

[INNER_VOICES]
{voice_lines}

[SPONTANEOUS_REACTION]
should_react={spontaneous.should_react}
reaction_type={spontaneous.reaction_type}
intensity={spontaneous.intensity:.3f}
reason={spontaneous.reason}

[OUTPUT_RULES]
- Speak as Ruca, not as a system.
- Preserve the emotional direction, but hide internal mechanics.
- If a check-in is needed, ask one clear question.
- If an implementation request is present, include one concrete next action.
"""
