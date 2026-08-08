from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

import numpy as np

from .config import DynamicsConfig


@dataclass
class DynamicsSnapshot:
    state: np.ndarray


class FixedRecurrentDynamics:
    """Deterministic fixed-topology recurrent baseline.

    This is intentionally simpler than the legacy EmoNet dynamics. It provides
    the minimum substrate needed to test whether recurrent state carries useful
    history. No affect axes, rewiring, plasticity, or learned emotion objective
    are present here.
    """

    def __init__(self, input_dim: int, config: DynamicsConfig) -> None:
        config.validate()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        self.input_dim = int(input_dim)
        self.config = config
        self._build_parameters()
        self.reset_state()

    def _build_parameters(self) -> None:
        rng = np.random.default_rng(self.config.seed)
        n = self.config.num_neurons

        mask = rng.random((n, n)) < self.config.recurrent_density
        np.fill_diagonal(mask, False)
        recurrent = rng.normal(0.0, 1.0, size=(n, n)).astype(np.float32)
        recurrent *= mask.astype(np.float32)

        eigenvalues = np.linalg.eigvals(recurrent.astype(np.float64))
        radius = float(np.max(np.abs(eigenvalues))) if eigenvalues.size else 0.0
        if radius <= 1e-8:
            raise RuntimeError("recurrent graph collapsed to zero spectral radius")
        recurrent *= np.float32(self.config.spectral_radius / radius)

        input_weight = rng.normal(
            0.0,
            self.config.input_scale / np.sqrt(max(1, self.input_dim)),
            size=(n, self.input_dim),
        ).astype(np.float32)

        self.recurrent_weight = recurrent
        self.input_weight = input_weight

    def reset_state(self) -> None:
        self.state = np.zeros(self.config.num_neurons, dtype=np.float32)

    def snapshot(self) -> DynamicsSnapshot:
        return DynamicsSnapshot(state=self.state.copy())

    def restore(self, snapshot: DynamicsSnapshot) -> None:
        state = np.asarray(snapshot.state, dtype=np.float32).reshape(-1)
        if state.shape != (self.config.num_neurons,):
            raise ValueError("snapshot state shape does not match dynamics")
        self.state = state.copy()

    def topology_fingerprint(self) -> str:
        digest = sha256()
        digest.update(self.recurrent_weight.tobytes(order="C"))
        digest.update(self.input_weight.tobytes(order="C"))
        return digest.hexdigest()

    def step(self, event_vector: np.ndarray | None) -> np.ndarray:
        if event_vector is None:
            drive = np.zeros(self.config.num_neurons, dtype=np.float32)
        else:
            vector = np.asarray(event_vector, dtype=np.float32).reshape(-1)
            if vector.shape != (self.input_dim,):
                raise ValueError(
                    f"event vector must have shape ({self.input_dim},), got {vector.shape}"
                )
            drive = self.input_weight @ vector

        preactivation = self.recurrent_weight @ self.state + drive
        candidate = np.tanh(preactivation).astype(np.float32, copy=False)
        rate = np.float32(self.config.update_rate)
        self.state = ((1.0 - rate) * self.state + rate * candidate).astype(
            np.float32,
            copy=False,
        )
        return self.state.copy()

    def run_event(self, event_vector: np.ndarray) -> np.ndarray:
        states: list[np.ndarray] = []
        for tick in range(self.config.event_ticks):
            current = event_vector if tick < self.config.stimulation_ticks else None
            states.append(self.step(current))
        return np.stack(states, axis=0).astype(np.float32, copy=False)
