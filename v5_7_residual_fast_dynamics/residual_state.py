from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from emonet_v5 import DynamicsConfig, NeuralTrace, TextEncoder
from emonet_v5.dynamics import FixedRecurrentDynamics
from v5_6_dual_timescale_state.dual_state import SlowEMAMemory


@dataclass(frozen=True)
class ResidualObservation:
    fast_trace: NeuralTrace
    slow_state: np.ndarray
    residual_input: np.ndarray


class ResidualDrivenState:
    """Slow EMA context plus v5.0 fast recurrence driven by context residuals."""

    def __init__(
        self,
        encoder: TextEncoder,
        *,
        seed: int,
        slow_decay: float = 0.80,
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
        self.slow = SlowEMAMemory(encoder.output_dim, decay=slow_decay)
        self._build_fast()
        self._event_index = 0

    def _build_fast(self) -> None:
        self.fast = FixedRecurrentDynamics(
            input_dim=self.encoder.output_dim,
            config=self.config,
        )

    def consume_event(self, text: str) -> ResidualObservation:
        embedding = self.encoder.encode(text).astype(np.float32, copy=False)
        previous_slow = self.slow.state.astype(np.float32, copy=True)
        residual = (embedding - previous_slow).astype(np.float32, copy=False)
        states = self.fast.run_event(residual)
        trace = NeuralTrace(states=states, event_index=self._event_index)
        slow_state = self.slow.update(embedding)
        self._event_index += 1
        return ResidualObservation(
            fast_trace=trace,
            slow_state=slow_state,
            residual_input=residual.copy(),
        )

    def consume_sequence(self, texts: list[str] | tuple[str, ...]) -> list[ResidualObservation]:
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


def fast_features(observation: ResidualObservation) -> np.ndarray:
    return observation.fast_trace.states.reshape(-1).astype(np.float32, copy=False)


def slow_features(observation: ResidualObservation) -> np.ndarray:
    return observation.slow_state.astype(np.float32, copy=True)
