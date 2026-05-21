from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class RucaEvent:
    event_type: str
    user_text: str
    elapsed_minutes: float
    should_speak: bool
    reason: str

    def to_record(self) -> dict[str, bool | float | str]:
        return asdict(self)


def schedule_event(
    user_text: str,
    *,
    elapsed_minutes: float = 0.0,
    force_silence: bool = False,
) -> RucaEvent:
    clean = (user_text or "").strip()
    elapsed = max(0.0, float(elapsed_minutes))
    if clean:
        return RucaEvent(
            event_type="user_message",
            user_text=clean,
            elapsed_minutes=elapsed,
            should_speak=True,
            reason="user supplied an explicit message",
        )
    if force_silence:
        return RucaEvent(
            event_type="silence_tick",
            user_text="",
            elapsed_minutes=elapsed,
            should_speak=False,
            reason="forced internal-only silence tick",
        )
    if elapsed >= 45.0:
        return RucaEvent(
            event_type="long_silence",
            user_text="",
            elapsed_minutes=elapsed,
            should_speak=True,
            reason="long silence may justify a gentle check-in",
        )
    return RucaEvent(
        event_type="silence_tick",
        user_text="",
        elapsed_minutes=elapsed,
        should_speak=False,
        reason="short silence should update internal state without speaking",
    )


def text_for_emotion(event: RucaEvent) -> str:
    if event.user_text:
        return event.user_text
    if event.event_type == "long_silence":
        return "quiet pause after a long silence"
    return "quiet pause"
