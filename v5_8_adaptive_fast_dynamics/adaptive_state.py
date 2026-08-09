from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from emonet_v5 import DynamicsConfig, NeuralTrace, TextEncoder
from emonet_v5.dynamics import FixedRecurrentDynamics
from v5_6_dual_timescale_state.dual_state import SlowEMAMemory


@dataclass(frozen=True)
class AdaptiveObservation:
    fast_trace: NeuralTrace
    slow_state: np.ndarray
    residual_input: np.ndarray
    adaptation_state: np.ndarray


class AdaptiveFastDynamics:
    """v5.0 fixed recurrence plus activity-dependent neural adaptation.

    The seeded matrices are copied from FixedRecurrentDynamics so beta=0 can
    serve as an exact mechanistic ablation of v5.8 back to the frozen v5.7
    residual-driven fast dynamics.
    """

    def __init__(
        self,
        input_dim: int,
        config: DynamicsConfig,
        *,
        adaptation_strength: float,
        adaptation_decay: float,
        use_recurrence: bool = True,
    ) -> None:
        if adaptation_strength < 0.0:
            raise ValueError("adaptation_strength must be nonnegative")
        if not 0.0 <= adaptation_decay < 1.0:
            raise ValueError("adaptation_decay must be in [0, 1)")
        base = FixedRecurrentDynamics(input_dim=input_dim, config=config)
        self.input_dim = int(input_dim)
        self.config = config
        self.adaptation_strength = float(adaptation_strength)
        self.adaptation_decay = float(adaptation_decay)
        self.use_recurrence = bool(use_recurrence)
        self.input_weight = base.input_weight.copy()
        self.recurrent_weight = base.recurrent_weight.copy()
        if not self.use_recurrence:
            self.recurrent_weight.fill(0.0)
        self.reset_state()

    def reset_state(self) -> None:
        self.state = np.zeros(self.config.num_neurons, dtype=np.float32)
        self.adaptation = np.zeros(self.config.num_neurons, dtype=np.float32)

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

        preactivation = (
            self.recurrent_weight @ self.state
            + drive
            - np.float32(self.adaptation_strength) * self.adaptation
        )
        candidate = np.tanh(preactivation).astype(np.float32, copy=False)
        rate = np.float32(self.config.update_rate)
        self.state = ((1.0 - rate) * self.state + rate * candidate).astype(
            np.float32, copy=False
        )
        decay = np.float32(self.adaptation_decay)
        self.adaptation = (
            decay * self.adaptation
            + (1.0 - decay) * np.abs(self.state)
        ).astype(np.float32, copy=False)
        return self.state.copy()

    def run_event(self, event_vector: np.ndarray) -> np.ndarray:
        states: list[np.ndarray] = []
        for tick in range(self.config.event_ticks):
            current = event_vector if tick < self.config.stimulation_ticks else None
            states.append(self.step(current))
        return np.stack(states, axis=0).astype(np.float32, copy=False)


class AdaptiveResidualState:
    """Frozen slow EMA context plus residual-driven adaptive fast dynamics."""

    def __init__(
        self,
        encoder: TextEncoder,
        *,
        seed: int,
        adaptation_strength: float,
        adaptation_decay: float,
        slow_decay: float = 0.80,
        use_recurrence: bool = True,
        dynamics_config: DynamicsConfig | None = None,
    ) -> None:
        self.encoder = encoder
        config = dynamics_config or DynamicsConfig(seed=seed)
        if config.seed != seed:
            config = DynamicsConfig(
                num_neurons=config.num_neurons,
                recurrent_density=config.recurrent_density,
                spectral_radius=config.spectral_radius,
                update_rate=config.update_rate,
                input_scale=config.input_scale,
                event_ticks=config.event_ticks,
                stimulation_ticks=config.stimulation_ticks,
                seed=seed,
            )
        self.config = config
        self.seed = int(seed)
        self.adaptation_strength = float(adaptation_strength)
        self.adaptation_decay = float(adaptation_decay)
        self.use_recurrence = bool(use_recurrence)
        self.slow = SlowEMAMemory(encoder.output_dim, decay=slow_decay)
        self._build_fast()
        self._event_index = 0

    def _build_fast(self) -> None:
        self.fast = AdaptiveFastDynamics(
            input_dim=self.encoder.output_dim,
            config=self.config,
            adaptation_strength=self.adaptation_strength,
            adaptation_decay=self.adaptation_decay,
            use_recurrence=self.use_recurrence,
        )

    def consume_event(self, text: str) -> AdaptiveObservation:
        embedding = self.encoder.encode(text).astype(np.float32, copy=False)
        previous_slow = self.slow.state.astype(np.float32, copy=True)
        residual = (embedding - previous_slow).astype(np.float32, copy=False)
        states = self.fast.run_event(residual)
        trace = NeuralTrace(states=states, event_index=self._event_index)
        slow_state = self.slow.update(embedding)
        observation = AdaptiveObservation(
            fast_trace=trace,
            slow_state=slow_state,
            residual_input=residual.copy(),
            adaptation_state=self.fast.adaptation.copy(),
        )
        self._event_index += 1
        return observation

    def consume_sequence(self, texts: list[str] | tuple[str, ...]) -> list[AdaptiveObservation]:
        return [self.consume_event(text) for text in texts]

    def reset_fast(self) -> None:
        self.fast.reset_state()

    def reset_slow(self) -> None:
        self.slow.reset()

    def reset_both(self) -> None:
        self.fast.reset_state()
        self.slow.reset()

    def reset_all(self) -> None:
        self._build_fast()
        self.slow.reset()
        self._event_index = 0
