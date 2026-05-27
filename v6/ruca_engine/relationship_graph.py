from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .emotion import InputSignals
from .models import EmotionState, clamp


DEFAULT_METRICS = {
    "trust": 0.0,
    "need_for_reassurance": 0.0,
    "protective_tension": 0.0,
    "alignment": 0.0,
}


@dataclass(frozen=True)
class RelationshipEdge:
    source: str
    target: str
    metrics: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "RelationshipEdge":
        metrics = payload.get("metrics", {})
        return cls(
            source=str(payload.get("source", "")).strip(),
            target=str(payload.get("target", "")).strip(),
            metrics={str(name): clamp(float(value), -1.0, 1.0) for name, value in dict(metrics).items()},
        )

    def to_record(self) -> dict[str, Any]:
        return {"source": self.source, "target": self.target, "metrics": dict(self.metrics)}


@dataclass(frozen=True)
class RelationshipGraph:
    edges: tuple[RelationshipEdge, ...] = ()

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> "RelationshipGraph":
        if not isinstance(payload, Mapping):
            return cls()
        source = payload.get("edges", ())
        return cls(edges=tuple(RelationshipEdge.from_mapping(item) for item in source if isinstance(item, Mapping)))

    def edge(self, source: str, target: str) -> RelationshipEdge:
        for item in self.edges:
            if item.source == source and item.target == target:
                return item
        return RelationshipEdge(source=source, target=target, metrics=dict(DEFAULT_METRICS))

    def to_record(self) -> dict[str, Any]:
        return {"edges": [edge.to_record() for edge in self.edges]}


def update_relationship_graph(
    previous: RelationshipGraph,
    *,
    signals: InputSignals,
    emotion_state: EmotionState,
    event_type: str,
) -> RelationshipGraph:
    edges = {(edge.source, edge.target): dict(edge.metrics) for edge in previous.edges}
    silence = 0.08 if event_type == "no_reply" else 0.0

    _add(edges, "user", "ruca", "trust", signals.warmth * 0.08 - signals.alarm * 0.025)
    _add(edges, "user", "ruca", "need_for_reassurance", signals.alarm * 0.07 + silence)
    _add(edges, "ruca", "rocky", "protective_tension", emotion_state.protective_tension * 0.05 + signals.alarm * 0.06 + silence)
    _add(edges, "ruca", "ricky", "alignment", signals.curiosity * 0.04 + signals.action_pressure * 0.03)
    _add(edges, "rookie", "ruca", "alignment", signals.warmth * 0.04 + signals.curiosity * 0.03)

    return RelationshipGraph(
        edges=tuple(
            RelationshipEdge(source=source, target=target, metrics=metrics)
            for (source, target), metrics in sorted(edges.items())
        )
    )


def _add(edges: dict[tuple[str, str], dict[str, float]], source: str, target: str, metric: str, delta: float) -> None:
    metrics = edges.setdefault((source, target), dict(DEFAULT_METRICS))
    metrics[metric] = round(clamp(float(metrics.get(metric, 0.0)) + float(delta), -1.0, 1.0), 4)
