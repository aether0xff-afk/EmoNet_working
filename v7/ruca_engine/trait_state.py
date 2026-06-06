from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .emotion import InputSignals
from .models import CharacterProfile, clamp


DEFAULT_TRAITS = {
    "warmth": 0.40,
    "protectiveness": 0.30,
    "analysis": 0.35,
    "initiative": 0.30,
    "curiosity": 0.35,
}


@dataclass(frozen=True)
class CharacterTraitState:
    characters: dict[str, dict[str, float]] = field(default_factory=dict)

    @classmethod
    def from_profiles(cls, profiles: Mapping[str, CharacterProfile]) -> "CharacterTraitState":
        characters: dict[str, dict[str, float]] = {}
        for character_id, profile in profiles.items():
            traits = dict(DEFAULT_TRAITS)
            for name, value in profile.traits.items():
                traits[str(name)] = clamp(float(value), 0.0, 1.0)
            characters[str(character_id)] = traits
        return cls(characters=characters)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> "CharacterTraitState":
        if not isinstance(payload, Mapping):
            return cls()
        source = payload.get("characters", payload)
        if not isinstance(source, Mapping):
            return cls()
        characters: dict[str, dict[str, float]] = {}
        for character_id, traits in source.items():
            if isinstance(traits, Mapping):
                characters[str(character_id)] = {str(name): clamp(float(value), 0.0, 1.0) for name, value in traits.items()}
        return cls(characters=characters)

    def to_record(self) -> dict[str, Any]:
        return {"characters": {key: dict(value) for key, value in self.characters.items()}}


def update_trait_state(
    previous: CharacterTraitState,
    profiles: Mapping[str, CharacterProfile],
    signals: InputSignals,
    *,
    event_type: str,
) -> CharacterTraitState:
    base = previous if previous.characters else CharacterTraitState.from_profiles(profiles)
    characters = {character_id: dict(traits) for character_id, traits in base.characters.items()}
    silence_pressure = 0.08 if event_type == "no_reply" else 0.0

    _move(characters, "ruca", "warmth", signals.warmth * 0.28 + signals.alarm * 0.10, 0.18)
    _move(characters, "ruca", "protectiveness", signals.alarm * 0.30 + silence_pressure, 0.18)
    _move(characters, "ricky", "analysis", max(signals.curiosity, signals.action_pressure) * 0.26, 0.16)
    _move(characters, "rocky", "protectiveness", signals.alarm * 0.38 + signals.action_pressure * 0.18 + silence_pressure, 0.22)
    _move(characters, "rocky", "initiative", signals.action_pressure * 0.34, 0.16)
    _move(characters, "rookie", "curiosity", signals.curiosity * 0.28 + silence_pressure, 0.16)

    return CharacterTraitState(characters=characters)


def _move(characters: dict[str, dict[str, float]], character_id: str, trait: str, delta: float, alpha: float) -> None:
    traits = characters.setdefault(character_id, dict(DEFAULT_TRAITS))
    current = clamp(float(traits.get(trait, DEFAULT_TRAITS.get(trait, 0.35))), 0.0, 1.0)
    target = clamp(current + float(delta), 0.0, 1.0)
    traits[trait] = round(clamp(current * (1.0 - alpha) + target * alpha, 0.0, 1.0), 4)
