from __future__ import annotations

from dataclasses import dataclass

from .config import DynamicsConfig
from .dynamics import FixedRecurrentDynamics
from .encoders import TextEncoder
from .trace import NeuralTrace


@dataclass(frozen=True)
class ModelState:
    event_index: int
    topology_fingerprint: str


class EmoNetV5Clean:
    """Minimal stateful EmoNet substrate for clean trace validation."""

    def __init__(self, encoder: TextEncoder, config: DynamicsConfig | None = None) -> None:
        self.encoder = encoder
        self.config = config or DynamicsConfig()
        self._build_dynamics()
        self._event_index = 0
        self._captured_traces: list[NeuralTrace] = []

    def _build_dynamics(self) -> None:
        self.dynamics = FixedRecurrentDynamics(
            input_dim=self.encoder.output_dim,
            config=self.config,
        )

    @property
    def captured_traces(self) -> tuple[NeuralTrace, ...]:
        return tuple(self._captured_traces)

    @property
    def topology_fingerprint(self) -> str:
        return self.dynamics.topology_fingerprint()

    def state_info(self) -> ModelState:
        return ModelState(
            event_index=self._event_index,
            topology_fingerprint=self.topology_fingerprint,
        )

    def consume_event(self, text: str) -> NeuralTrace:
        embedding = self.encoder.encode(text)
        states = self.dynamics.run_event(embedding)
        trace = NeuralTrace(states=states, event_index=self._event_index)
        self._captured_traces.append(trace)
        self._event_index += 1
        return trace

    def consume_sequence(self, texts: list[str] | tuple[str, ...]) -> list[NeuralTrace]:
        return [self.consume_event(text) for text in texts]

    def reset_transient(self) -> None:
        """Clear collected observations while preserving recurrent state.

        This operation intentionally does *not* erase history carried by the
        network. It is useful when the next event should be recorded as a fresh
        measurement while remaining in the same ongoing episode.
        """

        self._captured_traces.clear()

    def reset_episode(self) -> None:
        """Erase recurrent history while preserving the fixed topology."""

        self.dynamics.reset_state()
        self._captured_traces.clear()
        self._event_index = 0

    def reset_all(self) -> None:
        """Recreate topology/projection from the configured seed and clear state."""

        self._build_dynamics()
        self._captured_traces.clear()
        self._event_index = 0
