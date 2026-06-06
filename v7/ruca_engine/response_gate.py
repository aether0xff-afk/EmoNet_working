from __future__ import annotations

from .models import EmotionState, ResponseDecision, SpontaneousReactionDecision


SILENCE_EVENT_TYPES = {"typing", "answering", "processing", "no_reply", "silence_tick", "long_silence"}


def decide_response_action(
    *,
    event_type: str,
    emotion_state: EmotionState,
    spontaneous: SpontaneousReactionDecision,
    elapsed_minutes: float = 0.0,
) -> ResponseDecision:
    if event_type in SILENCE_EVENT_TYPES:
        if event_type in {"typing", "answering", "processing"}:
            return ResponseDecision(
                action="update_internal_only",
                intensity=spontaneous.intensity,
                reason="환경 상태 tick은 내부 상태만 갱신하고 외부 메시지를 보내지 않는다.",
            )
        if event_type == "silence_tick":
            if elapsed_minutes > 0:
                return ResponseDecision(
                    action="update_internal_only",
                    intensity=spontaneous.intensity,
                    reason="짧은 침묵은 내부 상태만 갱신하고 외부 메시지를 보내지 않는다.",
                )
            return ResponseDecision(action="stay_silent", intensity=0.0, reason="처리할 시간 경과가 없다.")
        if spontaneous.should_react:
            return ResponseDecision(
                action="send_message",
                intensity=spontaneous.intensity,
                reason="무입력 시간이 충분히 길고 내부 압력이 높아 낮은 강도의 자발 메시지를 보낸다.",
            )
        if elapsed_minutes > 0:
            return ResponseDecision(
                action="update_internal_only",
                intensity=spontaneous.intensity,
                reason="침묵을 감정 상태에 반영하되 아직 사용자에게 보이지 않는다.",
            )
        return ResponseDecision(action="stay_silent", intensity=0.0, reason="처리할 시간 경과가 없다.")

    if event_type in {"delayed_speech", "delayed_reply"}:
        return ResponseDecision(
            action="send_message",
            intensity=max(0.2, spontaneous.intensity),
            reason="신경망이 이전 입력을 잠시 품은 뒤 말이 밖으로 나오는 타이밍에 도달했다.",
        )

    if spontaneous.should_react:
        return ResponseDecision(
            action="send_message",
            intensity=spontaneous.intensity,
            reason="현재 입력에 대한 Ruca의 표면 반응을 생성한다.",
        )
    if emotion_state.arousal <= 0.08:
        return ResponseDecision(action="stay_silent", intensity=0.0, reason="표면 반응이 필요할 만큼 상태가 움직이지 않았다.")
    return ResponseDecision(action="send_message", intensity=max(0.1, spontaneous.intensity), reason="일반 사용자 입력 뒤 말이 밖으로 나온다.")
