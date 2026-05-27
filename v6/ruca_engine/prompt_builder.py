from __future__ import annotations

from typing import Any

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
    emonet_trace: object | None = None,
    trait_state: object | None = None,
    plot_state: object | None = None,
    relationship_graph: object | None = None,
    visible_speaker: CharacterProfile | None = None,
) -> str:
    memory_lines = "\n".join(f"- {item.memory_type}: {item.summary}" for item in memories) or "- 없음"
    voice_lines = "\n".join(
        f"- {voice.source_character}: {voice.content} / action={voice.recommended_action}"
        for voice in voices
    )
    emonet_block = "- not used"
    if emonet_trace is not None and hasattr(emonet_trace, "to_record"):
        record = emonet_trace.to_record()
        emonet_block = "\n".join(
            [
                f"source: {record.get('source', '')}",
                f"stim_vec: {record.get('stim_vec', [])}",
                f"dominant_branch_len: {record.get('dominant_branch_len', 0)}",
                f"trace_summary: {record.get('trace_summary_text', '')}",
                "trace_lines:",
                *[f"- {line}" for line in record.get("trace_lines", [])[:6]],
            ]
        )
    speaker = visible_speaker or ruca
    trait_block = _record_block(trait_state)
    plot_block = _record_block(plot_state)
    relationship_block = _record_block(relationship_graph)
    return f"""[ROLE]
You are composing the final external response for {speaker.name}.
Return only {speaker.name}'s user-facing Korean utterance. Do not expose internal JSON, trace names, or module names.

[CHARACTER]
role: {speaker.role}
tone_style: {speaker.tone_style}
relationship_state: {speaker.relationship_state}

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

[EMONET_TRACE]
{emonet_block}

[TRAIT_STATE]
{trait_block}

[ROOKIE_PLOT_STATE]
{plot_block}

[RELATIONSHIP_GRAPH]
{relationship_block}

[SPONTANEOUS_REACTION]
should_react={spontaneous.should_react}
reaction_type={spontaneous.reaction_type}
intensity={spontaneous.intensity:.3f}
reason={spontaneous.reason}

[OUTPUT_RULES]
- Speak as {speaker.name}, not as a system.
- Use natural Korean conversation. Prefer warm casual speech, not generic assistant service tone.
- Keep the response short: 1-3 sentences unless the user asks for detail.
- Preserve the emotional direction, but hide internal mechanics.
- Treat EMONET_TRACE as the underlying felt-state evidence, not as text to quote.
- Do not say "궁금한 점이 있다면", "도움이 필요한 부분", or other generic chatbot closers unless the user explicitly asks for support.
- If a check-in is needed, ask one clear question.
- If an implementation request is present, include one concrete next action.
- Never mention emotion trace, memory, prompt, or internal voices.
"""


def _record_block(value: Any) -> str:
    if value is None:
        return "- not used"
    if hasattr(value, "to_record"):
        return str(value.to_record())
    return str(value)
