from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from emonet_v5 import DynamicsConfig
from emonet_v5.dynamics import FixedRecurrentDynamics


@dataclass
class BatchState:
    fast: np.ndarray
    slow: np.ndarray
    adaptation: np.ndarray | None = None

    def copy(self) -> "BatchState":
        return BatchState(
            fast=self.fast.copy(),
            slow=self.slow.copy(),
            adaptation=None if self.adaptation is None else self.adaptation.copy(),
        )


class BatchedResidualDynamics:
    """Vectorized implementation of frozen v5.7/v5.8 equations.

    This class changes only execution order across independent samples. Network
    matrices come from the exact frozen FixedRecurrentDynamics constructor.
    """

    def __init__(
        self,
        input_dim: int,
        config: DynamicsConfig,
        *,
        slow_decay: float = 0.80,
        adaptation_strength: float | None = None,
        adaptation_decay: float = 0.995,
    ) -> None:
        base = FixedRecurrentDynamics(input_dim=input_dim, config=config)
        self.input_dim = int(input_dim)
        self.config = config
        self.input_weight = base.input_weight.copy()
        self.recurrent_weight = base.recurrent_weight.copy()
        self.slow_decay = np.float32(slow_decay)
        self.adaptation_strength = (
            None if adaptation_strength is None else np.float32(adaptation_strength)
        )
        self.adaptation_decay = np.float32(adaptation_decay)

    @property
    def adaptive(self) -> bool:
        return self.adaptation_strength is not None

    def zeros(self, batch_size: int) -> BatchState:
        fast = np.zeros((batch_size, self.config.num_neurons), dtype=np.float32)
        slow = np.zeros((batch_size, self.input_dim), dtype=np.float32)
        adaptation = (
            np.zeros_like(fast) if self.adaptive else None
        )
        return BatchState(fast=fast, slow=slow, adaptation=adaptation)

    def run_event(
        self,
        state: BatchState,
        event: np.ndarray,
    ) -> tuple[BatchState, np.ndarray, np.ndarray]:
        event = np.asarray(event, dtype=np.float32)
        if event.ndim != 2 or event.shape[1] != self.input_dim:
            raise ValueError(
                f"event must have shape [batch,{self.input_dim}], got {event.shape}"
            )
        if event.shape[0] != state.fast.shape[0]:
            raise ValueError("event batch does not match state batch")

        residual = (event - state.slow).astype(np.float32, copy=False)
        input_drive = (residual @ self.input_weight.T).astype(np.float32, copy=False)
        fast = state.fast.copy()
        adaptation = None if state.adaptation is None else state.adaptation.copy()
        traces: list[np.ndarray] = []
        rate = np.float32(self.config.update_rate)

        for tick in range(self.config.event_ticks):
            recurrent = (fast @ self.recurrent_weight.T).astype(np.float32, copy=False)
            pre = recurrent
            if tick < self.config.stimulation_ticks:
                pre = pre + input_drive
            if adaptation is not None:
                pre = pre - self.adaptation_strength * adaptation
            candidate = np.tanh(pre).astype(np.float32, copy=False)
            fast = ((1.0 - rate) * fast + rate * candidate).astype(
                np.float32, copy=False
            )
            if adaptation is not None:
                adaptation = (
                    self.adaptation_decay * adaptation
                    + (1.0 - self.adaptation_decay) * np.abs(fast)
                ).astype(np.float32, copy=False)
            traces.append(fast.copy())

        slow = (
            self.slow_decay * state.slow
            + (1.0 - self.slow_decay) * event
        ).astype(np.float32, copy=False)
        return (
            BatchState(fast=fast, slow=slow, adaptation=adaptation),
            np.stack(traces, axis=1).astype(np.float32, copy=False),
            residual.copy(),
        )

    def reset_fast(self, state: BatchState) -> BatchState:
        result = state.copy()
        result.fast.fill(0.0)
        if result.adaptation is not None:
            result.adaptation.fill(0.0)
        return result

    def reset_slow(self, state: BatchState) -> BatchState:
        result = state.copy()
        result.slow.fill(0.0)
        return result

    def reset_both(self, state: BatchState) -> BatchState:
        return self.zeros(state.fast.shape[0])
