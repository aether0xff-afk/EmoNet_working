from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .models import EmotionState, utc_now_iso
from .plot_manager import RookiePlotState
from .relationship_graph import RelationshipGraph
from .trait_state import CharacterTraitState


@dataclass(frozen=True)
class RucaSessionState:
    schema_version: int = 1
    emotion_state: EmotionState = field(default_factory=EmotionState)
    trait_state: CharacterTraitState = field(default_factory=CharacterTraitState)
    plot_state: RookiePlotState = field(default_factory=RookiePlotState)
    relationship_graph: RelationshipGraph = field(default_factory=RelationshipGraph)
    turn_index: int = 0
    recent_history: tuple[dict[str, Any], ...] = ()
    updated_at: str = field(default_factory=utc_now_iso)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> "RucaSessionState":
        if not isinstance(payload, Mapping):
            return cls()
        history = payload.get("recent_history", ())
        return cls(
            schema_version=int(payload.get("schema_version", 1)),
            emotion_state=EmotionState.from_mapping(payload.get("emotion_state")),
            trait_state=CharacterTraitState.from_mapping(payload.get("trait_state")),
            plot_state=RookiePlotState.from_mapping(payload.get("plot_state")),
            relationship_graph=RelationshipGraph.from_mapping(payload.get("relationship_graph")),
            turn_index=int(payload.get("turn_index", 0)),
            recent_history=tuple(dict(item) for item in history if isinstance(item, Mapping)),
            updated_at=str(payload.get("updated_at", "") or utc_now_iso()),
        )

    def next_turn(
        self,
        *,
        user_text: str,
        assistant_text: str,
        emotion_state: EmotionState,
        trait_state: CharacterTraitState | None = None,
        plot_state: RookiePlotState | None = None,
        relationship_graph: RelationshipGraph | None = None,
        debug_summary: Mapping[str, Any],
        max_history: int = 12,
    ) -> "RucaSessionState":
        entry = {
            "turn_index": self.turn_index + 1,
            "user_text": user_text,
            "assistant_text": assistant_text,
            "spoke": bool(assistant_text.strip()),
            "event_type": str(debug_summary.get("event_type", "")),
            "spontaneous_reaction": dict(debug_summary.get("spontaneous_reaction", {})),
            "created_at": utc_now_iso(),
        }
        history = (*self.recent_history, entry)[-max(1, int(max_history)) :]
        return RucaSessionState(
            schema_version=self.schema_version,
            emotion_state=emotion_state,
            trait_state=trait_state or self.trait_state,
            plot_state=plot_state or self.plot_state,
            relationship_graph=relationship_graph or self.relationship_graph,
            turn_index=self.turn_index + 1,
            recent_history=tuple(history),
            updated_at=utc_now_iso(),
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "emotion_state": self.emotion_state.to_record(),
            "trait_state": self.trait_state.to_record(),
            "plot_state": self.plot_state.to_record(),
            "relationship_graph": self.relationship_graph.to_record(),
            "turn_index": self.turn_index,
            "recent_history": [dict(item) for item in self.recent_history],
            "updated_at": self.updated_at,
        }


class SessionStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> RucaSessionState:
        if not self.path.exists():
            return RucaSessionState()
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError(f"session file must contain an object: {self.path}")
        return RucaSessionState.from_mapping(payload)

    def save(self, state: RucaSessionState) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(state.to_record(), ensure_ascii=False, indent=2), encoding="utf-8")
