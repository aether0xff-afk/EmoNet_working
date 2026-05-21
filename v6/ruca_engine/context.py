from __future__ import annotations

from dataclasses import asdict, dataclass

from .emotion import InputSignals
from .models import CharacterProfile, MemoryItem


@dataclass(frozen=True)
class TurnContext:
    event_type: str
    user_position: str
    rookie_question: str
    unresolved_need: str
    memory_pressure: float

    def to_record(self) -> dict[str, float | str]:
        return asdict(self)


def analyze_turn_context(
    *,
    user_text: str,
    scheduled_event_type: str = "user_message",
    rookie: CharacterProfile,
    signals: InputSignals,
    memories: tuple[MemoryItem, ...],
) -> TurnContext:
    memory_pressure = min(1.0, sum(item.importance for item in memories[:3]) / 2.0)
    if scheduled_event_type in {"silence_tick", "long_silence"}:
        event_type = scheduled_event_type
        user_position = "사용자는 지금 직접 말하지 않고 있고, Ruca는 관계를 밀어붙이지 않는 위치에 있다."
        rookie_question = "침묵을 깨야 할 만큼 중요한 신호가 있는가?"
        unresolved_need = "내부 상태 갱신과 발화 억제 균형"
    elif signals.alarm >= 0.55:
        event_type = "distress"
        user_position = "사용자는 지금 설명보다 안정과 확인을 먼저 필요로 하는 위치에 있다."
        rookie_question = "내가 지금 뭘 몰라서 더 불안해하는 걸까?"
        unresolved_need = "안전한 다음 행동과 짧은 확인"
    elif signals.action_pressure >= 0.50:
        event_type = "implementation_request"
        user_position = "사용자는 관찰자가 아니라 실행을 맡기려는 위치에 있다."
        rookie_question = "첫 단추를 어디에 끼워야 실제로 움직일까?"
        unresolved_need = "작은 실행 단위와 검증 기준"
    elif signals.curiosity >= 0.45:
        event_type = "question"
        user_position = "사용자는 구조를 이해하려는 초입자 위치에 있다."
        rookie_question = "어떤 선택지가 남아 있고 무엇부터 이해해야 할까?"
        unresolved_need = "개념 정리와 선택지 압축"
    elif signals.warmth >= 0.50:
        event_type = "relationship_signal"
        user_position = "사용자는 관계적 온도를 건네는 위치에 있다."
        rookie_question = "이 온도를 어떻게 어색하지 않게 돌려줄 수 있을까?"
        unresolved_need = "짧은 상호성"
    else:
        event_type = "ordinary_turn"
        user_position = "사용자는 흐름을 이어 가는 위치에 있다."
        rookie_question = "지금 대화에서 놓치면 안 되는 작은 신호가 있을까?"
        unresolved_need = "맥락 유지"
    has_relationship_memory = any(item.memory_type == "relationship" for item in memories)
    if memory_pressure >= 0.45 or has_relationship_memory:
        unresolved_need = f"{unresolved_need}, 이전 관계 기억 반영"
    return TurnContext(
        event_type=event_type,
        user_position=user_position,
        rookie_question=f"{rookie.name}: {rookie_question}",
        unresolved_need=unresolved_need,
        memory_pressure=round(memory_pressure, 3),
    )
