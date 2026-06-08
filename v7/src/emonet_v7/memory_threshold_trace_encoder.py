"""Trace encoder for neuron-memory-threshold RSNN dynamics."""

from __future__ import annotations

import torch
from torch import nn


class MemoryThresholdTraceEncoder(nn.Module):
    """Compress fast dynamics and neuron-local memory states into latent z."""

    def __init__(self, *, num_neurons: int, hidden_dim: int = 64, output_dim: int = 64) -> None:
        super().__init__()
        self.gru = nn.GRU(num_neurons * 5, hidden_dim, batch_first=True)
        self.proj = nn.Linear(hidden_dim, output_dim)

    def forward(
        self,
        spike_seq: torch.Tensor,
        membrane_seq: torch.Tensor,
        adaptation_seq: torch.Tensor,
        accumulation_seq: torch.Tensor,
        memory_strength_seq: torch.Tensor,
    ) -> torch.Tensor:
        features = torch.cat(
            [
                spike_seq,
                membrane_seq,
                adaptation_seq,
                accumulation_seq,
                memory_strength_seq,
            ],
            dim=-1,
        )
        _, hidden = self.gru(features)
        return self.proj(hidden[-1])
