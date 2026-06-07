"""Differentiable event-window execution for later self-supervised training."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .adaptive_rsnn import AdaptiveSparseRSNN, SNNState


@dataclass
class DifferentiableWindow:
    """Gradient-preserving SNN sequences with shape [batch, ticks, neurons]."""

    spike: torch.Tensor
    membrane: torch.Tensor
    adaptation: torch.Tensor
    threshold: torch.Tensor


def run_differentiable_window(
    *,
    snn: AdaptiveSparseRSNN,
    event_current: torch.Tensor,
    state: SNNState,
    event_ticks: int,
    stimulation_ticks: int,
) -> tuple[SNNState, DifferentiableWindow]:
    """Run an event window without detaching tensors or moving them to CPU.

    Use this function for training. Use ``AdaptiveSparseRSNN.run_window`` when
    collecting detached analysis logs.
    """

    if not 0 <= stimulation_ticks <= event_ticks:
        raise ValueError("stimulation_ticks must be between 0 and event_ticks")

    spikes: list[torch.Tensor] = []
    membranes: list[torch.Tensor] = []
    adaptations: list[torch.Tensor] = []
    thresholds: list[torch.Tensor] = []
    for tick in range(event_ticks):
        current = event_current if tick < stimulation_ticks else torch.zeros_like(event_current)
        state = snn.step(current=current, state=state)
        spikes.append(state.spike)
        membranes.append(state.membrane)
        adaptations.append(state.adaptation)
        thresholds.append(state.threshold)

    return state, DifferentiableWindow(
        spike=torch.stack(spikes, dim=1),
        membrane=torch.stack(membranes, dim=1),
        adaptation=torch.stack(adaptations, dim=1),
        threshold=torch.stack(thresholds, dim=1),
    )
