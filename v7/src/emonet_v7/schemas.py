"""Shared event schemas for EmoNet v7."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

EventKind = Literal[
    "user_message",
    "internal_thought",
    "module_message",
    "elapsed_time",
]

EVENT_KIND_TO_ID: dict[str, int] = {
    "user_message": 0,
    "internal_thought": 1,
    "module_message": 2,
    "elapsed_time": 3,
}


@dataclass(frozen=True)
class Event:
    """One text or time event observed by an EmoNet module."""

    event_id: str
    kind: EventKind
    text: str
    speaker_id: str
    elapsed_seconds: float = 0.0
