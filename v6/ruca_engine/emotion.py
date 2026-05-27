from __future__ import annotations

import re
from dataclasses import dataclass

from .models import EmotionState, clamp


ALARM_PATTERNS = re.compile(r"불안|무서|위험|힘들|죽|아파|망했|싫어|혼란|두려|panic|anxious", re.IGNORECASE)
WARM_PATTERNS = re.compile(r"고마|좋아|괜찮|믿|편해|사랑|기뻐|thank|thanks|love", re.IGNORECASE)
ACTION_PATTERNS = re.compile(r"해야|하자|시작|끝내|고쳐|만들|구현|해봐|가자|start|build|fix", re.IGNORECASE)
QUESTION_PATTERNS = re.compile(r"\?|어떻게|왜|뭐|궁금|알려|설명|how|why|what", re.IGNORECASE)


@dataclass(frozen=True)
class InputSignals:
    alarm: float
    warmth: float
    action_pressure: float
    curiosity: float
    intensity: float

    def to_record(self) -> dict[str, float]:
        return {
            "alarm": self.alarm,
            "warmth": self.warmth,
            "action_pressure": self.action_pressure,
            "curiosity": self.curiosity,
            "intensity": self.intensity,
        }


def analyze_input(text: str) -> InputSignals:
    clean = (text or "").strip()
    length_factor = min(1.0, len(clean) / 240.0)
    exclaim_factor = min(0.35, clean.count("!") * 0.08)
    alarm = 0.65 if ALARM_PATTERNS.search(clean) else 0.0
    warmth = 0.55 if WARM_PATTERNS.search(clean) else 0.0
    action = 0.55 if ACTION_PATTERNS.search(clean) else 0.0
    curiosity = 0.50 if QUESTION_PATTERNS.search(clean) else 0.0
    intensity = clamp(length_factor + exclaim_factor + max(alarm, action) * 0.35, 0.0, 1.0)
    return InputSignals(
        alarm=clamp(alarm + exclaim_factor, 0.0, 1.0),
        warmth=clamp(warmth, 0.0, 1.0),
        action_pressure=clamp(action + exclaim_factor, 0.0, 1.0),
        curiosity=clamp(curiosity + length_factor * 0.25, 0.0, 1.0),
        intensity=intensity,
    )


def update_emotion_state(previous: EmotionState, text: str) -> tuple[EmotionState, InputSignals]:
    signals = analyze_input(text)
    valence_delta = signals.warmth * 0.18 - signals.alarm * 0.22
    arousal_delta = max(signals.alarm, signals.action_pressure, signals.intensity) * 0.28
    affinity_delta = signals.warmth * 0.12 - signals.alarm * 0.03
    stability_delta = signals.warmth * 0.08 - max(signals.alarm, signals.intensity) * 0.18
    protective_delta = signals.alarm * 0.34 + signals.action_pressure * 0.10
    curiosity_delta = signals.curiosity * 0.18 + signals.warmth * 0.05

    next_state = EmotionState(
        valence=clamp(previous.valence * 0.72 + valence_delta),
        arousal=clamp(previous.arousal * 0.70 + 0.12 + arousal_delta, 0.0, 1.0),
        affinity=clamp(previous.affinity * 0.88 + affinity_delta, 0.0, 1.0),
        stability=clamp(previous.stability * 0.84 + 0.08 + stability_delta, 0.0, 1.0),
        protective_tension=clamp(previous.protective_tension * 0.76 + protective_delta, 0.0, 1.0),
        curiosity=clamp(previous.curiosity * 0.80 + 0.05 + curiosity_delta, 0.0, 1.0),
    )
    return next_state, signals


def update_emotion_for_event(
    previous: EmotionState,
    *,
    event_type: str,
    text: str = "",
    elapsed_minutes: float = 0.0,
) -> tuple[EmotionState, InputSignals]:
    if event_type != "no_reply":
        return update_emotion_state(previous, text)

    elapsed = max(0.0, float(elapsed_minutes))
    time_pressure = clamp(elapsed / 180.0, 0.0, 1.0)
    recent_warmth = 0.18 if "고마" in text or "좋" in text or "따뜻" in text else 0.0
    signals = InputSignals(
        alarm=clamp(time_pressure * 0.38, 0.0, 1.0),
        warmth=recent_warmth,
        action_pressure=0.0,
        curiosity=clamp(0.18 + time_pressure * 0.28, 0.0, 1.0),
        intensity=clamp(0.20 + time_pressure * 0.58, 0.0, 1.0),
    )
    next_state = EmotionState(
        valence=clamp(previous.valence * 0.88 - time_pressure * 0.06 + recent_warmth * 0.04),
        arousal=clamp(previous.arousal * 0.88 + 0.05 + time_pressure * 0.18, 0.0, 1.0),
        affinity=clamp(previous.affinity * 0.94 + recent_warmth * 0.03, 0.0, 1.0),
        stability=clamp(previous.stability * 0.90 + 0.04 - time_pressure * 0.12, 0.0, 1.0),
        protective_tension=clamp(previous.protective_tension * 0.90 + time_pressure * 0.16, 0.0, 1.0),
        curiosity=clamp(previous.curiosity * 0.90 + 0.05 + time_pressure * 0.10, 0.0, 1.0),
    )
    return next_state, signals
