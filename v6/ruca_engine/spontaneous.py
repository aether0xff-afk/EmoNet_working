from __future__ import annotations

from .emotion import InputSignals
from .models import EmotionState, MemoryItem, SpontaneousReactionDecision


def decide_spontaneous_reaction(
    *,
    emotion_state: EmotionState,
    signals: InputSignals,
    memories: tuple[MemoryItem, ...] = (),
    event_type: str = "user_message",
    elapsed_minutes: float = 0.0,
) -> SpontaneousReactionDecision:
    repeated_alarm = sum(1 for item in memories if item.emotion_snapshot.get("protective_tension", 0.0) >= 0.55)
    if event_type == "silence_tick":
        return SpontaneousReactionDecision(
            should_react=False,
            reaction_type="internal_only",
            intensity=0.0,
            reason="짧은 침묵은 관계를 밀어붙이지 않고 내부 상태만 갱신한다.",
        )
    if event_type == "long_silence" and (elapsed_minutes >= 45.0 or emotion_state.protective_tension >= 0.45 or repeated_alarm >= 1):
        return SpontaneousReactionDecision(
            should_react=True,
            reaction_type="quiet_check_in",
            intensity=round(float(min(1.0, 0.35 + elapsed_minutes / 180.0 + repeated_alarm * 0.1)), 3),
            reason="침묵이 길어졌고 최근 긴장 또는 관계 기억이 남아 있어 짧은 확인이 적절하다.",
        )
    if signals.alarm >= 0.60 or emotion_state.protective_tension >= 0.70 or repeated_alarm >= 2:
        intensity = max(signals.alarm, emotion_state.protective_tension, min(1.0, repeated_alarm / 3.0))
        return SpontaneousReactionDecision(
            should_react=True,
            reaction_type="check_in",
            intensity=round(float(intensity), 3),
            reason="불안 또는 보호 긴장이 높아 Ruca가 그냥 지나치기 어렵다.",
        )
    if signals.warmth >= 0.55 and emotion_state.affinity >= 0.55:
        return SpontaneousReactionDecision(
            should_react=True,
            reaction_type="warm_reciprocity",
            intensity=round(float(min(1.0, signals.warmth + emotion_state.affinity * 0.25)), 3),
            reason="관계적 온도가 충분히 높아 짧은 상호 반응을 덧붙일 수 있다.",
        )
    if signals.action_pressure >= 0.60 and emotion_state.stability >= 0.35:
        return SpontaneousReactionDecision(
            should_react=True,
            reaction_type="initiative",
            intensity=round(float(signals.action_pressure), 3),
            reason="실행 압력이 높아 Ruca가 먼저 다음 행동을 잡아 주는 편이 좋다.",
        )
    return SpontaneousReactionDecision(
        should_react=False,
        reaction_type="none",
        intensity=0.0,
        reason="현재 상태에서는 사용자의 주도권을 유지하는 편이 좋다.",
    )
