from __future__ import annotations

from dataclasses import dataclass, field
from math import tanh


STIM_VECTOR_DIM = 8
STIM_VECTOR_NAMES = (
    "valence",
    "arousal",
    "threat",
    "novelty",
    "agency",
    "social_pressure",
    "fatigue",
    "coherence",
)


@dataclass(frozen=True)
class RuntimeEvent:
    kind: str
    text: str = ""
    elapsed_seconds: float = 0.0


@dataclass
class NeuronCell:
    neuron_id: int
    cluster_id: int
    activation: float = 0.0
    potential: float = 0.0
    memory_k: float = 0.0
    local_memory: list[dict] = field(default_factory=list)


@dataclass(frozen=True)
class NeuronSnapshot:
    neuron_id: int
    cluster_id: int
    activation: float
    potential: float
    memory_k: float
    local_memory: tuple[dict, ...]


@dataclass(frozen=True)
class ClusterSnapshot:
    cluster_id: int
    size: int
    mean_activity: float
    peak_activity: float
    memory_load: float


@dataclass(frozen=True)
class RuntimeSnapshot:
    tick_index: int
    event_kind: str
    stim_vec: tuple[float, ...]
    neurons: tuple[NeuronSnapshot, ...]
    clusters: tuple[ClusterSnapshot, ...]
    dominant_cluster_id: int | None


class AlwaysOnEmoNetRuntime:
    """Small persistent neural runtime for v7/v8 structure experiments."""

    def __init__(
        self,
        neuron_count: int = 64,
        cluster_count: int = 4,
        remember_threshold: float = 0.6,
        decay: float = 0.86,
    ) -> None:
        if neuron_count <= 0:
            raise ValueError("neuron_count must be positive")
        if cluster_count <= 0:
            raise ValueError("cluster_count must be positive")
        self.tick_index = 0
        self.cluster_count = min(cluster_count, neuron_count)
        self.remember_threshold = remember_threshold
        self.decay = decay
        self.neurons = [
            NeuronCell(neuron_id=index, cluster_id=index % self.cluster_count)
            for index in range(neuron_count)
        ]

    def tick(self, event: RuntimeEvent) -> RuntimeSnapshot:
        stim_vec = self._encode_event(event)
        stim_energy = sum(abs(value) for value in stim_vec) / STIM_VECTOR_DIM

        for neuron in self.neurons:
            direct_drive = stim_vec[neuron.neuron_id % STIM_VECTOR_DIM]
            neighbor_drive = self._neighbor_activity(neuron.neuron_id)
            neuron.potential = (neuron.potential * self.decay) + direct_drive + (neighbor_drive * 0.18)
            neuron.activation = tanh(neuron.potential)
            neuron.memory_k = (neuron.memory_k * 0.94) + (abs(neuron.activation) * stim_energy)
            if neuron.memory_k >= self.remember_threshold:
                self._store_local_memory(neuron, event, stim_vec)
                neuron.memory_k *= 0.5

        clusters = self._cluster_snapshots()
        dominant = max(clusters, key=lambda cluster: cluster.mean_activity).cluster_id if clusters else None
        snapshot = RuntimeSnapshot(
            tick_index=self.tick_index,
            event_kind=event.kind,
            stim_vec=tuple(stim_vec),
            neurons=tuple(self._neuron_snapshot(neuron) for neuron in self.neurons),
            clusters=tuple(clusters),
            dominant_cluster_id=dominant,
        )
        self.tick_index += 1
        return snapshot

    def _encode_event(self, event: RuntimeEvent) -> list[float]:
        kind = event.kind.lower().strip()
        text = event.text or ""
        text_len = min(len(text) / 120.0, 1.0)
        elapsed = min(max(event.elapsed_seconds, 0.0) / 10.0, 1.0)
        base = [0.0] * STIM_VECTOR_DIM

        if kind == "user_message":
            base = [0.15, 0.55, 0.18, 0.42, 0.36, 0.28, 0.08, 0.34]
            base[1] += text_len * 0.25
            base[3] += self._lexical_variance(text) * 0.2
        elif kind == "typing":
            base = [0.04, 0.22, 0.05, 0.18, 0.08, 0.16, 0.03, 0.12]
            base[1] += elapsed * 0.08
            base[6] += elapsed * 0.04
        elif kind == "answering":
            base = [0.08, 0.34, 0.04, 0.22, 0.38, 0.24, 0.10, 0.26]
        elif kind == "processing":
            base = [0.02, 0.28, 0.06, 0.32, 0.22, 0.08, 0.06, 0.24]
        else:
            base = [0.0, 0.08, 0.02, 0.03, 0.02, 0.02, 0.02 + elapsed * 0.05, 0.10]

        return [max(-1.0, min(1.0, value)) for value in base]

    def _neighbor_activity(self, neuron_id: int) -> float:
        left = self.neurons[(neuron_id - 1) % len(self.neurons)].activation
        right = self.neurons[(neuron_id + 1) % len(self.neurons)].activation
        return (left + right) / 2.0

    def _store_local_memory(self, neuron: NeuronCell, event: RuntimeEvent, stim_vec: list[float]) -> None:
        neuron.local_memory.append(
            {
                "tick_index": self.tick_index,
                "event_kind": event.kind,
                "text_signature": event.text[:80],
                "stim_vec": list(stim_vec),
                "activation": neuron.activation,
            }
        )
        if len(neuron.local_memory) > 8:
            del neuron.local_memory[0]

    def _cluster_snapshots(self) -> list[ClusterSnapshot]:
        snapshots: list[ClusterSnapshot] = []
        for cluster_id in range(self.cluster_count):
            members = [neuron for neuron in self.neurons if neuron.cluster_id == cluster_id]
            if not members:
                continue
            activities = [abs(neuron.activation) for neuron in members]
            snapshots.append(
                ClusterSnapshot(
                    cluster_id=cluster_id,
                    size=len(members),
                    mean_activity=sum(activities) / len(activities),
                    peak_activity=max(activities),
                    memory_load=sum(len(neuron.local_memory) for neuron in members),
                )
            )
        return snapshots

    def _neuron_snapshot(self, neuron: NeuronCell) -> NeuronSnapshot:
        return NeuronSnapshot(
            neuron_id=neuron.neuron_id,
            cluster_id=neuron.cluster_id,
            activation=neuron.activation,
            potential=neuron.potential,
            memory_k=neuron.memory_k,
            local_memory=tuple(dict(item) for item in neuron.local_memory),
        )

    def _lexical_variance(self, text: str) -> float:
        if not text:
            return 0.0
        return min(len(set(text)) / max(len(text), 1), 1.0)
