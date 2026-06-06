from __future__ import annotations

from dataclasses import asdict, dataclass


VALID_EVENT_TYPES = {
    "user_message",
    "delayed_speech",
    "delayed_reply",
    "typing",
    "answering",
    "processing",
    "no_reply",
    "silence_tick",
    "long_silence",
}


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
    """Normalize legacy user-message/silence calls into a runtime event.

    New autonomous callers should use ``make_event(event_type="no_reply")``.
    ``silence_tick`` and ``long_silence`` remain supported for compatibility
    with the earlier v6 CLI and tests.
    """
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
        return make_event(event_type="silence_tick", elapsed_minutes=elapsed)
    if elapsed >= 45.0:
        return make_event(event_type="long_silence", elapsed_minutes=elapsed)
    return make_event(event_type="silence_tick", elapsed_minutes=elapsed)


def make_event(
    *,
    event_type: str,
    text: str = "",
    elapsed_minutes: float = 0.0,
) -> RucaEvent:
    """Create an explicit runtime event without replaying old user text."""
    clean_type = str(event_type or "").strip().lower()
    if clean_type not in VALID_EVENT_TYPES:
        allowed = ", ".join(sorted(VALID_EVENT_TYPES))
        raise ValueError(f"unsupported event_type: {event_type!r}; expected one of {allowed}")

    clean_text = (text or "").strip()
    elapsed = max(0.0, float(elapsed_minutes))
    if clean_type == "user_message":
        if not clean_text:
            raise ValueError("user_message events require non-empty text")
        return RucaEvent(
            event_type="user_message",
            user_text=clean_text,
            elapsed_minutes=elapsed,
            should_speak=True,
            reason="user supplied an explicit message",
        )
    if clean_type in {"delayed_speech", "delayed_reply"}:
        if not clean_text:
            raise ValueError(f"{clean_type} events require reference text")
        return RucaEvent(
            event_type="delayed_speech",
            user_text="",
            elapsed_minutes=elapsed,
            should_speak=True,
            reason="previous user message becomes speech after neural timing delay",
        )
    if clean_type == "no_reply":
        return RucaEvent(
            event_type="no_reply",
            user_text="",
            elapsed_minutes=elapsed,
            should_speak=False,
            reason="elapsed time without a new user message is processed as an autonomous event",
        )
    if clean_type == "typing":
        return RucaEvent(
            event_type="typing",
            user_text="",
            elapsed_minutes=elapsed,
            should_speak=False,
            reason="user typing is processed as environmental stimulation without a visible reply",
        )
    if clean_type == "answering":
        return RucaEvent(
            event_type="answering",
            user_text="",
            elapsed_minutes=elapsed,
            should_speak=False,
            reason="assistant answer generation is processed as environmental stimulation",
        )
    if clean_type == "processing":
        return RucaEvent(
            event_type="processing",
            user_text="",
            elapsed_minutes=elapsed,
            should_speak=False,
            reason="internal processing time is processed as environmental stimulation",
        )
    if clean_type == "long_silence":
        return RucaEvent(
            event_type="long_silence",
            user_text="",
            elapsed_minutes=elapsed,
            should_speak=True,
            reason="legacy long-silence compatibility event; response gate still decides whether to speak",
        )
    return RucaEvent(
        event_type="silence_tick",
        user_text="",
        elapsed_minutes=elapsed,
        should_speak=False,
        reason="legacy short-silence compatibility event for internal-only state updates",
    )


def text_for_emotion(event: RucaEvent) -> str:
    if event.user_text:
        return event.user_text
    if event.event_type in {"no_reply", "long_silence"}:
        return "quiet pause after a long silence"
    return "quiet pause"
