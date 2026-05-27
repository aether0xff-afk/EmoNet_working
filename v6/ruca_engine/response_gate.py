from __future__ import annotations

from .models import EmotionState, ResponseDecision, SpontaneousReactionDecision


def decide_response_action(
    *,
    event_type: str,
    emotion_state: EmotionState,
    spontaneous: SpontaneousReactionDecision,
    elapsed_minutes: float = 0.0,
) -> ResponseDecision:
    if event_type == "no_reply":
        if spontaneous.should_react:
            return ResponseDecision(
                action="send_message",
                intensity=spontaneous.intensity,
                reason="무입력 사건이 충분히 길고 내부 긴장이 높아 자발 메시지를 보낸다.",
            )
        if elapsed_minutes > 0:
            return ResponseDecision(
                action="update_internal_only",
                intensity=spontaneous.intensity,
                reason="침묵을 감정 상태에 반영하되 아직 사용자에게 보이지 않는다.",
            )
        return ResponseDecision(action="stay_silent", intensity=0.0, reason="처리할 시간 경과가 없다.")

    if spontaneous.should_react:
        return ResponseDecision(
            action="send_message",
            intensity=spontaneous.intensity,
            reason="현재 입력에 대한 Ruca의 표면 반응을 생성한다.",
        )
    if emotion_state.arousal <= 0.08:
        return ResponseDecision(action="stay_silent", intensity=0.0, reason="표면 반응이 필요할 만큼 상태가 움직이지 않았다.")
    return ResponseDecision(action="send_message", intensity=max(0.1, spontaneous.intensity), reason="일반 사용자 입력에 응답한다.")
