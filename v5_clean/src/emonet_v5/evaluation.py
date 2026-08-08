from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from .model import EmoNetV5Clean
from .trace import NeuralTrace, temporal_shuffle, wrong_sample_controls


@dataclass(frozen=True)
class ContextProbeResult:
    name: str
    history_distance: float
    reset_distance: float
    history_to_reset_ratio: float
    trace_a_fingerprint: str
    trace_b_fingerprint: str
    reset_a_fingerprint: str
    reset_b_fingerprint: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def normalized_l2(left: np.ndarray, right: np.ndarray, eps: float = 1e-8) -> float:
    a = np.asarray(left, dtype=np.float32).reshape(-1)
    b = np.asarray(right, dtype=np.float32).reshape(-1)
    if a.shape != b.shape:
        raise ValueError("distance inputs must have the same shape")
    denom = float(np.sqrt(np.mean(a * a)) + np.sqrt(np.mean(b * b)) + eps)
    return float(np.sqrt(np.mean((a - b) ** 2)) / denom)


def trace_distance(left: NeuralTrace, right: NeuralTrace) -> float:
    return normalized_l2(left.summary_features(), right.summary_features())


def run_context_probe(
    model: EmoNetV5Clean,
    *,
    name: str,
    context_a: list[str],
    context_b: list[str],
    final_text: str,
) -> ContextProbeResult:
    """Compare the same final event under two different histories.

    The model is rebuilt from the same seed before each arm so topology and input
    projection are controlled. A second pair repeats the experiment but resets
    recurrent state immediately before the final event.
    """

    model.reset_all()
    model.consume_sequence(context_a)
    trace_a = model.consume_event(final_text)

    model.reset_all()
    model.consume_sequence(context_b)
    trace_b = model.consume_event(final_text)

    model.reset_all()
    model.consume_sequence(context_a)
    model.reset_episode()
    reset_a = model.consume_event(final_text)

    model.reset_all()
    model.consume_sequence(context_b)
    model.reset_episode()
    reset_b = model.consume_event(final_text)

    history_distance = trace_distance(trace_a, trace_b)
    reset_distance = trace_distance(reset_a, reset_b)
    ratio = history_distance / max(reset_distance, 1e-12)

    return ContextProbeResult(
        name=name,
        history_distance=history_distance,
        reset_distance=reset_distance,
        history_to_reset_ratio=ratio,
        trace_a_fingerprint=trace_a.fingerprint(),
        trace_b_fingerprint=trace_b.fingerprint(),
        reset_a_fingerprint=reset_a.fingerprint(),
        reset_b_fingerprint=reset_b.fingerprint(),
    )


def build_controls(traces: list[NeuralTrace], seed: int) -> dict[str, list[NeuralTrace]]:
    """Create canonical trace controls for downstream experiments."""

    if not traces:
        raise ValueError("at least one trace is required")
    shuffled = [temporal_shuffle(trace, seed + idx) for idx, trace in enumerate(traces)]
    controls: dict[str, list[NeuralTrace]] = {
        "real": [NeuralTrace(trace.states.copy(), trace.event_index) for trace in traces],
        "temporal_shuffle": shuffled,
    }
    if len(traces) >= 2:
        controls["wrong_sample"] = wrong_sample_controls(traces)
    return controls
