from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from emonet_v5 import DynamicsConfig, EmoNetV5Clean, NeuralTrace, TextEncoder


@dataclass(frozen=True)
class DualObservation:
    fast_trace: NeuralTrace
    slow_state: np.ndarray


class SlowEMAMemory:
    """Label-free persistent embedding memory used as the slow timescale."""

    def __init__(self, dimension: int, decay: float = 0.80) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        if not 0.0 <= decay < 1.0:
            raise ValueError("decay must be in [0, 1)")
        self.dimension = int(dimension)
        self.decay = float(decay)
        self.reset()

    def reset(self) -> None:
        self.state = np.zeros(self.dimension, dtype=np.float32)

    def update(self, embedding: np.ndarray) -> np.ndarray:
        vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if vector.shape != (self.dimension,):
            raise ValueError("embedding dimension mismatch")
        self.state = (
            self.decay * self.state + (1.0 - self.decay) * vector
        ).astype(np.float32, copy=False)
        return self.read()

    def read(self) -> np.ndarray:
        result = self.state.astype(np.float32, copy=True)
        norm = float(np.linalg.norm(result))
        if norm > 0.0:
            result /= norm
        return result


class DualTimescaleState:
    """Fast fixed recurrent dynamics plus slow persistent semantic memory."""

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
        self.fast = EmoNetV5Clean(encoder=encoder, config=config)
        self.slow = SlowEMAMemory(encoder.output_dim, decay=slow_decay)

    def consume_event(self, text: str) -> DualObservation:
        embedding = self.encoder.encode(text)
        fast_trace = self.fast.consume_event(text)
        slow_state = self.slow.update(embedding)
        return DualObservation(fast_trace=fast_trace, slow_state=slow_state)

    def consume_sequence(self, texts: list[str] | tuple[str, ...]) -> list[DualObservation]:
        return [self.consume_event(text) for text in texts]

    def reset_fast(self) -> None:
        self.fast.reset_episode()

    def reset_slow(self) -> None:
        self.slow.reset()

    def reset_both(self) -> None:
        self.fast.reset_episode()
        self.slow.reset()

    def reset_all(self) -> None:
        self.fast.reset_all()
        self.slow.reset()


def fast_features(observation: DualObservation) -> np.ndarray:
    return observation.fast_trace.states.reshape(-1).astype(np.float32, copy=False)


def slow_features(observation: DualObservation) -> np.ndarray:
    return observation.slow_state.astype(np.float32, copy=True)


def dual_features(observation: DualObservation) -> np.ndarray:
    return np.concatenate(
        [fast_features(observation), slow_features(observation)],
        axis=0,
    ).astype(np.float32, copy=False)
