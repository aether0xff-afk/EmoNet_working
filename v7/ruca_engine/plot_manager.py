from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .context import TurnContext
from .emotion import InputSignals
from .models import clamp


@dataclass(frozen=True)
class RookiePlotState:
    scene_id: str = "scene-0001"
    scene_pressure: float = 0.0
    unresolved_threads: tuple[dict[str, Any], ...] = ()
    next_scene_hint: str = "현재 장면을 유지한다."

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> "RookiePlotState":
        if not isinstance(payload, Mapping):
            return cls()
        threads = payload.get("unresolved_threads", ())
        return cls(
            scene_id=str(payload.get("scene_id", "scene-0001") or "scene-0001"),
            scene_pressure=clamp(float(payload.get("scene_pressure", 0.0)), 0.0, 1.0),
            unresolved_threads=tuple(dict(item) for item in threads if isinstance(item, Mapping)),
            next_scene_hint=str(payload.get("next_scene_hint", "") or "현재 장면을 유지한다."),
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "scene_pressure": self.scene_pressure,
            "unresolved_threads": [dict(item) for item in self.unresolved_threads],
            "next_scene_hint": self.next_scene_hint,
        }


def update_plot_state(
    previous: RookiePlotState,
    *,
    event_type: str,
    user_text: str,
    signals: InputSignals,
    context: TurnContext,
    elapsed_minutes: float = 0.0,
    max_threads: int = 8,
) -> RookiePlotState:
    pressure_delta = signals.action_pressure * 0.20 + signals.alarm * 0.18 + signals.curiosity * 0.08
    if event_type == "no_reply":
        pressure_delta += min(0.35, max(0.0, elapsed_minutes) / 360.0)
    scene_pressure = round(clamp(previous.scene_pressure * 0.86 + pressure_delta, 0.0, 1.0), 4)

    threads = list(previous.unresolved_threads)
    thread_type = _thread_type(context, signals, event_type)
    if thread_type:
        next_thread = {
            "thread_id": f"thread-{len(threads) + 1:04d}",
            "thread_type": thread_type,
            "summary": _thread_summary(thread_type, user_text, event_type),
            "pressure": round(max(scene_pressure, signals.intensity), 3),
        }
        if thread_type == "silence_followup":
            threads = _upsert_silence_thread(threads, next_thread)
        else:
            threads.append(next_thread)
    threads = threads[-max(1, int(max_threads)) :]
    return RookiePlotState(
        scene_id=previous.scene_id,
        scene_pressure=scene_pressure,
        unresolved_threads=tuple(threads),
        next_scene_hint=_next_scene_hint(scene_pressure, thread_type, event_type),
    )


def _thread_type(context: TurnContext, signals: InputSignals, event_type: str) -> str:
    if event_type == "no_reply":
        return "silence_followup"
    if context.event_type == "implementation_request" or signals.action_pressure >= 0.45:
        return "implementation"
    if signals.alarm >= 0.55:
        return "distress"
    if signals.warmth >= 0.50:
        return "relationship_warmth"
    return ""


def _thread_summary(thread_type: str, user_text: str, event_type: str) -> str:
    if event_type == "no_reply":
        return "무입력 시간이 장면 압력을 올림"
    text = user_text.strip()
    return f"{thread_type}: {text[:80]}" if text else thread_type


def _next_scene_hint(scene_pressure: float, thread_type: str, event_type: str) -> str:
    if event_type == "no_reply" and scene_pressure >= 0.45:
        return "사용자를 재촉하지 않는 낮은 강도의 확인 장면을 준비한다."
    if thread_type == "implementation":
        return "실행 단위를 하나씩 닫는 장면으로 유지한다."
    if scene_pressure >= 0.70:
        return "긴장을 낮추는 짧은 장면 전환이 필요하다."
    return "현재 장면을 유지한다."


def _upsert_silence_thread(threads: list[dict[str, Any]], next_thread: dict[str, Any]) -> list[dict[str, Any]]:
    for index, thread in enumerate(threads):
        if thread.get("thread_type") == "silence_followup":
            updated = dict(thread)
            updated["summary"] = next_thread["summary"]
            updated["pressure"] = max(float(updated.get("pressure", 0.0)), float(next_thread["pressure"]))
            return [*threads[:index], updated, *threads[index + 1 :]]
    return [*threads, next_thread]
