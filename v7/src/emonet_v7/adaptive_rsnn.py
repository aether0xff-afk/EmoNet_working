"""Adaptive sparse recurrent spiking neural network for EmoNet v7.0."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .surrogate import spike_with_surrogate_gradient


@dataclass
class SNNState:
    """Per-neuron dynamic state."""

    membrane: torch.Tensor
    spike: torch.Tensor
    adaptation: torch.Tensor
    threshold: torch.Tensor


@dataclass
class TickTrace:
    """Detached trace snapshot for one simulation tick."""

    tick: int
    membrane: torch.Tensor
    spike: torch.Tensor
    adaptation: torch.Tensor
    threshold: torch.Tensor
    active_edges: torch.Tensor


def create_recurrent_mask(num_neurons: int, density: float, seed: int) -> torch.Tensor:
    """Create a deterministic sparse directed mask without self-loops."""

    if num_neurons <= 0:
        raise ValueError("num_neurons must be positive")
    if not 0.0 <= density <= 1.0:
        raise ValueError("density must be between 0 and 1")

    generator = torch.Generator().manual_seed(seed)
    mask = torch.rand(num_neurons, num_neurons, generator=generator) < density
    mask.fill_diagonal_(False)
    return mask.to(torch.float32)


class AdaptiveSparseRSNN(nn.Module):
    """ALIF-style sparse recurrent SNN with fixed connectivity mask."""

    def __init__(
        self,
        *,
        num_neurons: int,
        recurrent_density: float,
        seed: int,
        threshold_base: float = 1.0,
        adaptation_strength: float = 0.40,
        membrane_decay_min: float = 0.80,
        membrane_decay_max: float = 0.95,
        adaptation_decay_min: float = 0.90,
        adaptation_decay_max: float = 0.995,
        recurrent_weight_std: float = 0.12,
        input_weight_std: float = 0.10,
    ) -> None:
        super().__init__()
        if membrane_decay_min > membrane_decay_max:
            raise ValueError("membrane decay minimum exceeds maximum")
        if adaptation_decay_min > adaptation_decay_max:
            raise ValueError("adaptation decay minimum exceeds maximum")

        self.num_neurons = num_neurons
        self.threshold_base = float(threshold_base)
        self.adaptation_strength = float(adaptation_strength)

        generator = torch.Generator().manual_seed(seed)
        recurrent_mask = create_recurrent_mask(num_neurons, recurrent_density, seed)
        self.register_buffer("recurrent_mask", recurrent_mask)

        input_weight = torch.randn(num_neurons, num_neurons, generator=generator)
        recurrent_weight = torch.randn(num_neurons, num_neurons, generator=generator)
        self.input_weight = nn.Parameter(input_weight * input_weight_std)
        self.recurrent_weight = nn.Parameter(recurrent_weight * recurrent_weight_std)

        membrane_decay = torch.empty(num_neurons).uniform_(
            membrane_decay_min,
            membrane_decay_max,
            generator=generator,
        )
        adaptation_decay = torch.empty(num_neurons).uniform_(
            adaptation_decay_min,
            adaptation_decay_max,
            generator=generator,
        )
        self.register_buffer("membrane_decay", membrane_decay)
        self.register_buffer("adaptation_decay", adaptation_decay)

    def initial_state(self, batch_size: int, device: torch.device | str) -> SNNState:
        """Create an all-zero initial state."""

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        zeros = torch.zeros(batch_size, self.num_neurons, device=device)
        threshold = torch.full_like(zeros, self.threshold_base)
        return SNNState(
            membrane=zeros.clone(),
            spike=zeros.clone(),
            adaptation=zeros.clone(),
            threshold=threshold,
        )

    def step(self, current: torch.Tensor, state: SNNState) -> SNNState:
        """Advance the SNN by one tick."""

        if current.ndim != 2 or current.shape[-1] != self.num_neurons:
            raise ValueError("current must have shape [batch, num_neurons]")
        recurrent_weight = self.recurrent_weight * self.recurrent_mask
        recurrent_input = state.spike @ recurrent_weight.T
        external_input = current @ self.input_weight.T

        membrane = (
            self.membrane_decay * state.membrane
            + recurrent_input
            + external_input
            - state.spike * state.threshold
        )
        adaptation = self.adaptation_decay * state.adaptation + state.spike
        threshold = self.threshold_base + self.adaptation_strength * adaptation
        spike = spike_with_surrogate_gradient(membrane - threshold)
        return SNNState(membrane, spike, adaptation, threshold)

    def run_window(
        self,
        *,
        event_current: torch.Tensor,
        state: SNNState,
        event_ticks: int,
        stimulation_ticks: int,
    ) -> tuple[SNNState, list[TickTrace]]:
        """Run one event window and collect detached traces."""

        if not 0 <= stimulation_ticks <= event_ticks:
            raise ValueError("stimulation_ticks must be between 0 and event_ticks")
        traces: list[TickTrace] = []
        for tick in range(event_ticks):
            current = event_current if tick < stimulation_ticks else torch.zeros_like(event_current)
            previous_spike = state.spike
            state = self.step(current=current, state=state)
            active_edges = (
                previous_spike.unsqueeze(-1)
                * state.spike.unsqueeze(-2)
                * self.recurrent_mask
            )
            traces.append(
                TickTrace(
                    tick=tick,
                    membrane=state.membrane.detach().cpu(),
                    spike=state.spike.detach().cpu(),
                    adaptation=state.adaptation.detach().cpu(),
                    threshold=state.threshold.detach().cpu(),
                    active_edges=active_edges.detach().cpu(),
                )
            )
        return state, traces
