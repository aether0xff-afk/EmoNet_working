from __future__ import annotations

from .models import EmotionState, InnerVoiceCandidate, SpontaneousReactionDecision


def compose_response(
    *,
    user_text: str,
    emotion_state: EmotionState,
    voices: tuple[InnerVoiceCandidate, ...],
    spontaneous: SpontaneousReactionDecision,
) -> str:
    primary = _select_primary_voice(voices)
    opening = _opening_for_state(emotion_state, spontaneous)
    action = _action_line(primary, spontaneous)
    if spontaneous.should_react:
        return f"{opening} {action} { _spontaneous_line(spontaneous) }".strip()
    return f"{opening} {action}".strip()


def _select_primary_voice(voices: tuple[InnerVoiceCandidate, ...]) -> InnerVoiceCandidate:
    if not voices:
        raise ValueError("at least one inner voice candidate is required")
    return max(voices, key=lambda voice: (voice.urgency * 0.65 + voice.confidence * 0.35))


def _opening_for_state(emotion_state: EmotionState, spontaneous: SpontaneousReactionDecision) -> str:
    if spontaneous.reaction_type == "check_in" or emotion_state.protective_tension >= 0.62:
        return "잠깐, 이건 그냥 넘기고 싶지 않아."
    if emotion_state.valence >= 0.18 and emotion_state.affinity >= 0.55:
        return "좋아, 지금 온도는 꽤 괜찮아."
    if spontaneous.reaction_type == "initiative":
        return "좋아. 말만 정리하지 말고 바로 작게 잡자."
    return "응, 지금 흐름은 이렇게 잡을게."


def _action_line(primary: InnerVoiceCandidate, spontaneous: SpontaneousReactionDecision) -> str:
    if primary.source_character.lower() == "rocky" or spontaneous.reaction_type == "initiative":
        return "먼저 가장 작은 실행 단위부터 만들고, 막히는 지점은 바로 기록해서 다시 고치면 돼."
    if primary.source_character.lower() == "ricky":
        return "상황을 한 번 정리한 다음, 필요한 선택지만 남겨서 움직이는 게 좋아."
    return "네가 던진 신호를 먼저 받아 두고, 너무 멀리 돌아가지 않게 다음 반응으로 이어갈게."


def _spontaneous_line(spontaneous: SpontaneousReactionDecision) -> str:
    if spontaneous.reaction_type == "check_in":
        return "그리고 이건 확인하고 싶어. 지금 네가 버티는 쪽인지, 바로 도와야 하는 쪽인지 알려줘."
    if spontaneous.reaction_type == "warm_reciprocity":
        return "그 온도는 나도 놓치지 않을게."
    if spontaneous.reaction_type == "initiative":
        return "내 쪽에서 먼저 다음 단계까지 끌고 갈게."
    return ""
