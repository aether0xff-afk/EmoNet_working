"""Differentiable event-window execution for neuron-memory-threshold RSNNs."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .memory_threshold_rsnn import (
    ConsolidationSnapshot,
    MemoryThresholdSNNState,
    NeuronMemoryThresholdRSNN,
)


@dataclass
class MemoryThresholdDifferentiableWindow:
    """Gradient-preserving sequences with shape [batch, ticks, neurons]."""

    spike: torch.Tensor
    membrane: torch.Tensor
    adaptation: torch.Tensor
    threshold: torch.Tensor
    accumulation: torch.Tensor
    memory_strength: torch.Tensor
    consolidation: ConsolidationSnapshot


def run_memory_threshold_differentiable_window(
    *,
    snn: NeuronMemoryThresholdRSNN,
    event_current: torch.Tensor,
    state: MemoryThresholdSNNState,
    event_ticks: int,
    stimulation_ticks: int,
) -> tuple[MemoryThresholdSNNState, MemoryThresholdDifferentiableWindow]:
    """Run one event and consolidate neuron-local memories at its boundary."""

    if not 0 <= stimulation_ticks <= event_ticks:
        raise ValueError("stimulation_ticks must be between 0 and event_ticks")
    spikes: list[torch.Tensor] = []
    membranes: list[torch.Tensor] = []
    adaptations: list[torch.Tensor] = []
    thresholds: list[torch.Tensor] = []
    accumulations: list[torch.Tensor] = []
    memory_strengths: list[torch.Tensor] = []
    for tick in range(event_ticks):
        current = event_current if tick < stimulation_ticks else torch.zeros_like(event_current)
        state = snn.step(current=current, state=state)
        spikes.append(state.spike)
        membranes.append(state.membrane)
        adaptations.append(state.adaptation)
        thresholds.append(state.threshold)
        accumulations.append(state.accumulation)
        memory_strengths.append(state.memory_strength)
    state, consolidation = snn.consolidate_event(event_current=event_current, state=state)
    return state, MemoryThresholdDifferentiableWindow(
        spike=torch.stack(spikes, dim=1),
        membrane=torch.stack(membranes, dim=1),
        adaptation=torch.stack(adaptations, dim=1),
        threshold=torch.stack(thresholds, dim=1),
        accumulation=torch.stack(accumulations, dim=1),
        memory_strength=torch.stack(memory_strengths, dim=1),
        consolidation=consolidation,
    )
