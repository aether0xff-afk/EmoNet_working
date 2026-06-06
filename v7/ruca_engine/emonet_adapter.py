from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models import EmotionState, clamp


ROOT = Path(__file__).resolve().parents[2]
V7_ROOT = ROOT / "v7"
if str(V7_ROOT) not in sys.path:
    sys.path.insert(0, str(V7_ROOT))

from runtime import AlwaysOnEmoNetRuntime, RuntimeEvent, RuntimeSnapshot


@dataclass(frozen=True)
class EmoNetTraceResult:
    emotion_state: EmotionState
    snapshot: RuntimeSnapshot
    source: str = "emonet_v7_always_on_runtime"

    def to_record(self) -> dict[str, Any]:
        trace_profile = {
            "tick_index": self.snapshot.tick_index,
            "neuron_count": len(self.snapshot.neurons),
            "cluster_count": len(self.snapshot.clusters),
            "mean_active_nodes": _mean_active_nodes(self.snapshot),
            "active_window_ticks": self.snapshot.tick_index + 1,
            "ticks_run": self.snapshot.tick_index + 1,
            "dominant_cluster_id": self.snapshot.dominant_cluster_id,
        }
        return {
            "source": self.source,
            "event_kind": self.snapshot.event_kind,
            "emotion_state": self.emotion_state.to_record(),
            "stim_vec": list(self.snapshot.stim_vec),
            "stim_vec_dim": len(self.snapshot.stim_vec),
            "stim_vec_names": _stim_vec_names(),
            "trace_summary_text": _trace_summary_text(self.snapshot),
            "trace_lines": _trace_lines(self.snapshot),
            "trace_profile": trace_profile,
            "cluster_profile": [cluster.__dict__.copy() for cluster in self.snapshot.clusters],
            "neuron_memory": _neuron_memory_summary(self.snapshot),
        }


def create_emonet_runtime() -> AlwaysOnEmoNetRuntime:
    return AlwaysOnEmoNetRuntime(neuron_count=64, cluster_count=4)


def infer_emonet_trace(
    text: str,
    *,
    event_type: str = "user_message",
    elapsed_minutes: float = 0.0,
    runtime: AlwaysOnEmoNetRuntime | None = None,
) -> EmoNetTraceResult:
    active_runtime = runtime or create_emonet_runtime()
    event = RuntimeEvent(
        kind=_runtime_event_kind(event_type),
        text=str(text or ""),
        elapsed_seconds=max(0.0, float(elapsed_minutes)) * 60.0,
    )
    snapshot = active_runtime.tick(event)
    return EmoNetTraceResult(emotion_state=_snapshot_to_emotion_state(snapshot), snapshot=snapshot)


def _snapshot_to_emotion_state(snapshot: RuntimeSnapshot) -> EmotionState:
    stim = list(snapshot.stim_vec)
    valence, arousal, threat, novelty, agency, social_pressure, fatigue, coherence = stim
    cluster_pressure = clamp(_mean_active_nodes(snapshot) / max(len(snapshot.neurons), 1), 0.0, 1.0)
    memory_pressure = clamp(_total_memory_count(snapshot) / max(len(snapshot.neurons), 1), 0.0, 1.0)
    return EmotionState(
        valence=clamp(0.5 + valence * 0.35 - threat * 0.25 - fatigue * 0.10, 0.0, 1.0),
        arousal=clamp(arousal * 0.70 + novelty * 0.20 + cluster_pressure * 0.35, 0.0, 1.0),
        affinity=clamp(0.35 + social_pressure * 0.25 + coherence * 0.25 - threat * 0.15, 0.0, 1.0),
        stability=clamp(0.30 + coherence * 0.50 - arousal * 0.18 - fatigue * 0.18, 0.0, 1.0),
        protective_tension=clamp(threat * 0.55 + arousal * 0.20 + memory_pressure * 0.20, 0.0, 1.0),
        curiosity=clamp(novelty * 0.55 + agency * 0.30 + cluster_pressure * 0.20, 0.0, 1.0),
    )


def _runtime_event_kind(event_type: str) -> str:
    clean = str(event_type or "").strip().lower()
    if clean in {"typing", "answering", "processing", "user_message", "delayed_speech", "delayed_reply"}:
        return clean
    if clean in {"no_reply", "silence_tick", "long_silence"}:
        return "idle"
    return "idle"


def _trace_summary_text(snapshot: RuntimeSnapshot) -> str:
    dominant = snapshot.dominant_cluster_id
    active = _mean_active_nodes(snapshot)
    memory_count = _total_memory_count(snapshot)
    return (
        f"event={snapshot.event_kind}, tick={snapshot.tick_index}, "
        f"dominant_cluster={dominant}, mean_active_nodes={active:.2f}, "
        f"stored_neuron_memories={memory_count}"
    )


def _trace_lines(snapshot: RuntimeSnapshot) -> list[str]:
    lines = [
        (
            f"tick={snapshot.tick_index} event={snapshot.event_kind} "
            f"dominant_cluster={snapshot.dominant_cluster_id} stim_dim={len(snapshot.stim_vec)}"
        )
    ]
    for cluster in snapshot.clusters:
        lines.append(
            f"cluster={cluster.cluster_id} size={cluster.size} "
            f"mean_activity={cluster.mean_activity:.3f} peak_activity={cluster.peak_activity:.3f} "
            f"memory_load={cluster.memory_load:.0f}"
        )
    return lines


def _mean_active_nodes(snapshot: RuntimeSnapshot) -> float:
    return sum(abs(neuron.activation) for neuron in snapshot.neurons)


def _total_memory_count(snapshot: RuntimeSnapshot) -> int:
    return sum(len(neuron.local_memory) for neuron in snapshot.neurons)


def _neuron_memory_summary(snapshot: RuntimeSnapshot) -> dict[str, Any]:
    remembered = [neuron for neuron in snapshot.neurons if neuron.local_memory]
    return {
        "remembering_neuron_count": len(remembered),
        "stored_memory_count": _total_memory_count(snapshot),
        "remembering_neuron_ids": [neuron.neuron_id for neuron in remembered[:16]],
    }


def _stim_vec_names() -> list[str]:
    from runtime import STIM_VECTOR_NAMES

    return list(STIM_VECTOR_NAMES)
