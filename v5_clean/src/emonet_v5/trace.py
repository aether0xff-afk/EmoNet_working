from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

import numpy as np


@dataclass(frozen=True)
class NeuralTrace:
    """Raw recurrent state trajectory for exactly one input event."""

    states: np.ndarray
    event_index: int

    def __post_init__(self) -> None:
        states = np.asarray(self.states, dtype=np.float32)
        if states.ndim != 2:
            raise ValueError("trace states must have shape [ticks, neurons]")
        object.__setattr__(self, "states", states)

    @property
    def ticks(self) -> int:
        return int(self.states.shape[0])

    @property
    def num_neurons(self) -> int:
        return int(self.states.shape[1])

    @property
    def final_state(self) -> np.ndarray:
        if self.ticks == 0:
            return np.zeros(self.num_neurons, dtype=np.float32)
        return self.states[-1].copy()

    def fingerprint(self) -> str:
        return sha256(self.states.tobytes(order="C")).hexdigest()

    def summary_features(self) -> np.ndarray:
        """Deterministic non-learned summary used only for sanity probes."""

        if self.ticks == 0:
            return np.zeros(self.num_neurons * 6, dtype=np.float32)
        first = self.states[0]
        last = self.states[-1]
        return np.concatenate(
            [
                self.states.mean(axis=0),
                self.states.std(axis=0),
                self.states.min(axis=0),
                self.states.max(axis=0),
                last,
                last - first,
            ],
            axis=0,
        ).astype(np.float32, copy=False)


def temporal_shuffle(trace: NeuralTrace, seed: int) -> NeuralTrace:
    rng = np.random.default_rng(seed)
    order = rng.permutation(trace.ticks)
    return NeuralTrace(states=trace.states[order].copy(), event_index=trace.event_index)


def wrong_sample_controls(traces: list[NeuralTrace]) -> list[NeuralTrace]:
    """Rotate traces by one sample while preserving shape expectations."""

    if len(traces) < 2:
        raise ValueError("at least two traces are required for wrong-sample controls")
    rotated = traces[1:] + traces[:1]
    controls: list[NeuralTrace] = []
    for original, replacement in zip(traces, rotated, strict=True):
        if original.states.shape != replacement.states.shape:
            raise ValueError("wrong-sample controls require equal trace shapes")
        controls.append(
            NeuralTrace(states=replacement.states.copy(), event_index=original.event_index)
        )
    return controls
