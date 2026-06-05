from __future__ import annotations

from .emotion import InputSignals
from .context import TurnContext
from .event_scheduler import RucaEvent
from .models import CharacterProfile, EmotionState, InnerVoiceCandidate, MemoryItem


def generate_inner_voices(
    *,
    profiles: dict[str, CharacterProfile],
    user_text: str,
    emotion_state: EmotionState,
    memories: tuple[MemoryItem, ...] = (),
    signals: InputSignals,
    context: TurnContext | None = None,
    event: RucaEvent | None = None,
) -> tuple[InnerVoiceCandidate, ...]:
    memory_hint = _memory_hint(memories)
    rookie_hint = f" Rookie 관점: {context.rookie_question}" if context else ""
    ruca_content = _ruca_content(emotion_state, signals, memory_hint, rookie_hint, event)
    ricky_content = _ricky_content(signals, memory_hint, context, event)
    rocky_content = _rocky_content(emotion_state, signals, event)
    return (
        InnerVoiceCandidate(
            voice_id="voice-ruca",
            source_character=profiles["ruca"].name,
            content=ruca_content,
            emotion_bias={"affinity": emotion_state.affinity, "valence": emotion_state.valence},
            urgency=round(max(signals.alarm, signals.intensity) * 0.55, 3),
            confidence=0.74,
            recommended_action="관계를 해치지 않게 먼저 감정의 결을 받아 준다.",
        ),
        InnerVoiceCandidate(
            voice_id="voice-ricky",
            source_character=profiles["ricky"].name,
            content=ricky_content,
            emotion_bias={"stability": emotion_state.stability, "curiosity": emotion_state.curiosity},
            urgency=round(max(signals.curiosity, 1.0 - emotion_state.stability) * 0.50, 3),
            confidence=0.82,
            recommended_action="상황을 정리하고 필요한 확인 질문 또는 다음 단계를 제시한다.",
        ),
        InnerVoiceCandidate(
            voice_id="voice-rocky",
            source_character=profiles["rocky"].name,
            content=rocky_content,
            emotion_bias={"protective_tension": emotion_state.protective_tension, "arousal": emotion_state.arousal},
            urgency=round(max(emotion_state.protective_tension, signals.action_pressure) * 0.86, 3),
            confidence=0.68,
            recommended_action="위험하거나 막힌 지점이 있으면 바로 행동으로 옮기게 만든다.",
        ),
    )


def _memory_hint(memories: tuple[MemoryItem, ...]) -> str:
    if not memories:
        return "아직 강하게 붙잡을 관계 기억은 없다."
    top = memories[0]
    return f"가장 가까운 기억은 '{top.summary}'이다."


def _is_silence_event(event: RucaEvent | None) -> bool:
    return bool(event and event.event_type in {"no_reply", "silence_tick", "long_silence"})


def _ruca_content(
    emotion_state: EmotionState,
    signals: InputSignals,
    memory_hint: str,
    rookie_hint: str,
    event: RucaEvent | None,
) -> str:
    if _is_silence_event(event):
        return f"말이 없는 시간을 먼저 존중한다. 상태는 갱신하되, 필요가 크지 않으면 바깥으로 밀고 나가지 않는다. {memory_hint}{rookie_hint}"
    if signals.alarm >= 0.55 or emotion_state.protective_tension >= 0.58:
        return f"사용자가 흔들리는 신호를 보인다. 먼저 붙잡고, 너무 멀리 분석하지 않는다. {memory_hint}{rookie_hint}"
    if signals.warmth >= 0.50:
        return f"관계 신호가 따뜻하다. 가볍게 넘기지 말고 Ruca 쪽에서도 온도를 돌려준다. {memory_hint}{rookie_hint}"
    return f"차분하게 받아 주되, 필요한 만큼만 앞으로 민다. {memory_hint}{rookie_hint}"


def _ricky_content(
    signals: InputSignals,
    memory_hint: str,
    context: TurnContext | None,
    event: RucaEvent | None,
) -> str:
    need = f" 남은 필요: {context.unresolved_need}." if context else ""
    if _is_silence_event(event):
        return f"침묵 이벤트다. 최근 기억 압력과 보호 긴장이 충분하지 않으면 내부 기록만 남긴다.{need} {memory_hint}"
    if signals.curiosity >= 0.45:
        return f"질문 의도가 있다. 개념과 다음 행동을 분리해서 정리한다.{need} {memory_hint}"
    if signals.action_pressure >= 0.50:
        return f"실행 요청이 강하다. 범위를 작게 나누고 검증 기준을 붙인다.{need} {memory_hint}"
    return f"상황을 과장하지 않고 현재 맥락을 유지한다.{need} {memory_hint}"


def _rocky_content(emotion_state: EmotionState, signals: InputSignals, event: RucaEvent | None) -> str:
    if event and event.event_type == "silence_tick":
        return "지금은 먼저 치고 나가지 않는다. 대기하면서 다음 신호를 받을 준비만 한다."
    if event and event.event_type in {"no_reply", "long_silence"}:
        return "침묵이 길어질 수 있다. 압박하지 않는 짧은 확인이 필요한지 내부 압력을 본다."
    if signals.action_pressure >= 0.50:
        return "멈추지 말고 바로 작게 시작한다. 막히면 원인을 찾아 다시 붙인다."
    if emotion_state.protective_tension >= 0.55:
        return "사용자를 방치하면 안 된다. 짧고 단단하게 안전한 다음 행동을 준다."
    return "에너지는 낮게 유지하되, 필요하면 바로 앞으로 밀 준비를 한다."
