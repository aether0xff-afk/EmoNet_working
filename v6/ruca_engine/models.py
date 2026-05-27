from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def clamp(value: float, lower: float = -1.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, float(value)))


@dataclass(frozen=True)
class CharacterProfile:
    character_id: str
    name: str
    role: str
    traits: dict[str, float] = field(default_factory=dict)
    tone_style: str = ""
    relationship_state: str = ""
    visibility: str = "internal"

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "CharacterProfile":
        required = ("character_id", "name", "role")
        missing = [key for key in required if not str(payload.get(key, "")).strip()]
        if missing:
            raise ValueError(f"character profile missing required fields: {', '.join(missing)}")
        traits = payload.get("traits", {})
        if not isinstance(traits, Mapping):
            raise ValueError("character profile traits must be an object")
        return cls(
            character_id=str(payload["character_id"]).strip(),
            name=str(payload["name"]).strip(),
            role=str(payload["role"]).strip(),
            traits={str(key): float(value) for key, value in traits.items()},
            tone_style=str(payload.get("tone_style", "") or "").strip(),
            relationship_state=str(payload.get("relationship_state", "") or "").strip(),
            visibility=str(payload.get("visibility", "internal") or "internal").strip(),
        )

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EmotionState:
    valence: float = 0.0
    arousal: float = 0.25
    affinity: float = 0.45
    stability: float = 0.60
    protective_tension: float = 0.20
    curiosity: float = 0.40

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> "EmotionState":
        if not isinstance(payload, Mapping):
            return cls()
        return cls(
            valence=clamp(float(payload.get("valence", 0.0))),
            arousal=clamp(float(payload.get("arousal", 0.25)), 0.0, 1.0),
            affinity=clamp(float(payload.get("affinity", 0.45)), 0.0, 1.0),
            stability=clamp(float(payload.get("stability", 0.60)), 0.0, 1.0),
            protective_tension=clamp(float(payload.get("protective_tension", 0.20)), 0.0, 1.0),
            curiosity=clamp(float(payload.get("curiosity", 0.40)), 0.0, 1.0),
        )

    def to_record(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class MemoryItem:
    memory_id: str
    memory_type: str
    summary: str
    source_event: str
    emotion_snapshot: dict[str, float]
    importance: float
    ruca_interpretation: str = ""
    emotion_delta: dict[str, float] = field(default_factory=dict)
    relationship_effect: dict[str, float] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now_iso)
    last_accessed_at: str = field(default_factory=utc_now_iso)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "MemoryItem":
        snapshot = payload.get("emotion_snapshot", {})
        return cls(
            memory_id=str(payload.get("memory_id", "")).strip(),
            memory_type=str(payload.get("memory_type", "short_term")).strip(),
            summary=str(payload.get("summary", "")).strip(),
            source_event=str(payload.get("source_event", "")).strip(),
            emotion_snapshot={str(k): float(v) for k, v in dict(snapshot).items()},
            importance=clamp(float(payload.get("importance", 0.0)), 0.0, 1.0),
            ruca_interpretation=str(payload.get("ruca_interpretation", "") or "").strip(),
            emotion_delta={str(k): float(v) for k, v in dict(payload.get("emotion_delta", {})).items()},
            relationship_effect={str(k): float(v) for k, v in dict(payload.get("relationship_effect", {})).items()},
            created_at=str(payload.get("created_at", "") or utc_now_iso()),
            last_accessed_at=str(payload.get("last_accessed_at", "") or utc_now_iso()),
        )

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class InnerVoiceCandidate:
    voice_id: str
    source_character: str
    content: str
    emotion_bias: dict[str, float]
    urgency: float
    confidence: float
    recommended_action: str

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SpontaneousReactionDecision:
    should_react: bool
    reaction_type: str
    intensity: float
    reason: str

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ResponseDecision:
    action: str
    intensity: float
    reason: str

    def to_record(self) -> dict[str, Any]:
        return asdict(self)
