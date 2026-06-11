"""Neuron-level accumulation and memory-consolidation dynamics for EmoNet v7.

This module keeps firing and memory consolidation separate:

- ``threshold`` decides whether a neuron spikes now.
- ``memory_threshold`` decides whether a slowly accumulated candidate is
  consolidated into a persistent neuron-local memory strength.

Weak one-off stimuli fade from ``accumulation``. Repeated weak stimuli can cross
``memory_threshold`` before fading. Strong one-off stimuli can consolidate
immediately. The implementation is intentionally isolated from
``AdaptiveSparseRSNN`` so baseline experiments remain reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .surrogate import spike_with_surrogate_gradient


@dataclass
class MemoryThresholdSNNState:
    """Per-neuron fast dynamics plus pre-consolidation and persistent memory."""

    membrane: torch.Tensor
    spike: torch.Tensor
    adaptation: torch.Tensor
    threshold: torch.Tensor
    accumulation: torch.Tensor
    memory_strength: torch.Tensor


@dataclass
class MemoryThresholdTickTrace:
    """Detached trace snapshot for one simulation tick."""

    tick: int
    membrane: torch.Tensor
    spike: torch.Tensor
    adaptation: torch.Tensor
    threshold: torch.Tensor
    accumulation: torch.Tensor
    memory_strength: torch.Tensor
    active_edges: torch.Tensor


@dataclass
class ConsolidationSnapshot:
    """Differentiable event-boundary memory update diagnostics."""

    accumulation_before_reset: torch.Tensor
    accumulation_after_reset: torch.Tensor
    memory_gate: torch.Tensor
    memory_strength: torch.Tensor


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


class NeuronMemoryThresholdRSNN(nn.Module):
    """Sparse ALIF-style RSNN with neuron-local candidate-memory accumulation.

    Accumulation is updated once per event rather than once per simulation tick.
    This prevents ``event_ticks`` from implicitly changing how strongly one text
    event is remembered.
    """

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
        accumulation_decay: float = 0.85,
        accumulation_scale: float = 1.0,
        memory_threshold: float = 0.60,
        memory_gate_sharpness: float = 20.0,
        consolidation_scale: float = 0.50,
        consolidation_reset_fraction: float = 0.50,
        memory_decay: float = 0.98,
        memory_feedback_strength: float = 0.00,
    ) -> None:
        super().__init__()
        if membrane_decay_min > membrane_decay_max:
            raise ValueError("membrane decay minimum exceeds maximum")
        if adaptation_decay_min > adaptation_decay_max:
            raise ValueError("adaptation decay minimum exceeds maximum")
        for name, value in (
            ("accumulation_decay", accumulation_decay),
            ("memory_decay", memory_decay),
            ("consolidation_reset_fraction", consolidation_reset_fraction),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must remain in [0, 1]")
        if memory_threshold < 0.0:
            raise ValueError("memory_threshold must be non-negative")
        if memory_gate_sharpness <= 0.0:
            raise ValueError("memory_gate_sharpness must be positive")
        if accumulation_scale < 0.0 or consolidation_scale < 0.0 or memory_feedback_strength < 0.0:
            raise ValueError("memory scaling parameters must be non-negative")

        self.num_neurons = int(num_neurons)
        self.threshold_base = float(threshold_base)
        self.adaptation_strength = float(adaptation_strength)
        self.accumulation_decay = float(accumulation_decay)
        self.accumulation_scale = float(accumulation_scale)
        self.memory_threshold = float(memory_threshold)
        self.memory_gate_sharpness = float(memory_gate_sharpness)
        self.consolidation_scale = float(consolidation_scale)
        self.consolidation_reset_fraction = float(consolidation_reset_fraction)
        self.memory_decay = float(memory_decay)
        self.memory_feedback_strength = float(memory_feedback_strength)

        generator = torch.Generator().manual_seed(seed)
        recurrent_mask = create_recurrent_mask(num_neurons, recurrent_density, seed)
        self.register_buffer("recurrent_mask", recurrent_mask)

        input_weight = torch.randn(num_neurons, num_neurons, generator=generator)
        recurrent_weight = torch.randn(num_neurons, num_neurons, generator=generator)
        self.input_weight = nn.Parameter(input_weight * input_weight_std)
        self.recurrent_weight = nn.Parameter(recurrent_weight * recurrent_weight_std)

        membrane_decay = torch.empty(num_neurons).uniform_(membrane_decay_min, membrane_decay_max, generator=generator)
        adaptation_decay = torch.empty(num_neurons).uniform_(adaptation_decay_min, adaptation_decay_max, generator=generator)
        self.register_buffer("membrane_decay", membrane_decay)
        self.register_buffer("adaptation_decay", adaptation_decay)

    def initial_state(self, batch_size: int, device: torch.device | str) -> MemoryThresholdSNNState:
        """Create an all-zero initial state."""

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        zeros = torch.zeros(batch_size, self.num_neurons, device=device)
        threshold = torch.full_like(zeros, self.threshold_base)
        return MemoryThresholdSNNState(
            membrane=zeros.clone(),
            spike=zeros.clone(),
            adaptation=zeros.clone(),
            threshold=threshold,
            accumulation=zeros.clone(),
            memory_strength=zeros.clone(),
        )

    def step(self, current: torch.Tensor, state: MemoryThresholdSNNState) -> MemoryThresholdSNNState:
        """Advance fast spiking dynamics by one tick.

        Neuron-local memory is read here through optional feedback, but written
        only at an event boundary by :meth:`consolidate_event`.
        """

        if current.ndim != 2 or current.shape[-1] != self.num_neurons:
            raise ValueError("current must have shape [batch, num_neurons]")
        recurrent_weight = self.recurrent_weight * self.recurrent_mask
        recurrent_input = state.spike @ recurrent_weight.T
        external_input = current @ self.input_weight.T
        memory_feedback = self.memory_feedback_strength * state.memory_strength
        membrane = (
            self.membrane_decay * state.membrane
            + recurrent_input
            + external_input
            + memory_feedback
            - state.spike * state.threshold
        )
        adaptation = self.adaptation_decay * state.adaptation + state.spike
        threshold = self.threshold_base + self.adaptation_strength * adaptation
        spike = spike_with_surrogate_gradient(membrane - threshold)
        return MemoryThresholdSNNState(
            membrane=membrane,
            spike=spike,
            adaptation=adaptation,
            threshold=threshold,
            accumulation=state.accumulation,
            memory_strength=state.memory_strength,
        )

    def consolidate_event(
        self,
        *,
        event_current: torch.Tensor,
        state: MemoryThresholdSNNState,
    ) -> tuple[MemoryThresholdSNNState, ConsolidationSnapshot]:
        """Update candidate accumulation and persistent memory once per event.

        ``event_current`` is the encoder output before the internal input-weight
        transform. Using the external event current avoids recursively storing
        self-amplified recurrent activity during the first ablation.
        """

        if event_current.ndim != 2 or event_current.shape[-1] != self.num_neurons:
            raise ValueError("event_current must have shape [batch, num_neurons]")
        candidate = torch.tanh(event_current)
        accumulation_before_reset = torch.tanh(
            self.accumulation_decay * state.accumulation
            + self.accumulation_scale * candidate
        )
        memory_gate = torch.sigmoid(
            self.memory_gate_sharpness
            * (accumulation_before_reset.abs() - self.memory_threshold)
        )
        consolidated_update = memory_gate * accumulation_before_reset
        memory_strength = torch.tanh(
            self.memory_decay * state.memory_strength
            + self.consolidation_scale * consolidated_update
        )
        accumulation_after_reset = accumulation_before_reset * (
            1.0 - self.consolidation_reset_fraction * memory_gate
        )
        next_state = MemoryThresholdSNNState(
            membrane=state.membrane,
            spike=state.spike,
            adaptation=state.adaptation,
            threshold=state.threshold,
            accumulation=accumulation_after_reset,
            memory_strength=memory_strength,
        )
        return next_state, ConsolidationSnapshot(
            accumulation_before_reset=accumulation_before_reset,
            accumulation_after_reset=accumulation_after_reset,
            memory_gate=memory_gate,
            memory_strength=memory_strength,
        )

    def active_edges(self, previous_spike: torch.Tensor, current_spike: torch.Tensor) -> torch.Tensor:
        """Return active source-target edge candidates."""

        return previous_spike.unsqueeze(-1) * current_spike.unsqueeze(-2) * self.recurrent_mask.T

    def run_window(
        self,
        *,
        event_current: torch.Tensor,
        state: MemoryThresholdSNNState,
        event_ticks: int,
        stimulation_ticks: int,
    ) -> tuple[MemoryThresholdSNNState, list[MemoryThresholdTickTrace], ConsolidationSnapshot]:
        """Run one event window, then update neuron-local memory once."""

        if not 0 <= stimulation_ticks <= event_ticks:
            raise ValueError("stimulation_ticks must be between 0 and event_ticks")
        traces: list[MemoryThresholdTickTrace] = []
        for tick in range(event_ticks):
            current = event_current if tick < stimulation_ticks else torch.zeros_like(event_current)
            previous_spike = state.spike
            state = self.step(current=current, state=state)
            traces.append(
                MemoryThresholdTickTrace(
                    tick=tick,
                    membrane=state.membrane.detach().cpu(),
                    spike=state.spike.detach().cpu(),
                    adaptation=state.adaptation.detach().cpu(),
                    threshold=state.threshold.detach().cpu(),
                    accumulation=state.accumulation.detach().cpu(),
                    memory_strength=state.memory_strength.detach().cpu(),
                    active_edges=self.active_edges(previous_spike, state.spike).detach().cpu(),
                )
            )
        state, consolidation = self.consolidate_event(event_current=event_current, state=state)
        return state, traces, consolidation
