"""GRU compression of raw SNN traces."""

from __future__ import annotations

import torch
from torch import nn

from .adaptive_rsnn import TickTrace


class TraceEncoder(nn.Module):
    """Compress spike, membrane, and adaptation sequences into latent z."""

    def __init__(self, *, num_neurons: int, hidden_dim: int = 64, output_dim: int = 64) -> None:
        super().__init__()
        self.gru = nn.GRU(num_neurons * 3, hidden_dim, batch_first=True)
        self.proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, spike_seq: torch.Tensor, membrane_seq: torch.Tensor, adaptation_seq: torch.Tensor) -> torch.Tensor:
        features = torch.cat([spike_seq, membrane_seq, adaptation_seq], dim=-1)
        _, hidden = self.gru(features)
        return self.proj(hidden[-1])


def traces_to_sequences(traces: list[TickTrace]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stack TickTrace objects into [batch, ticks, neurons] tensors."""

    if not traces:
        raise ValueError("traces must not be empty")
    return tuple(torch.stack([getattr(trace, name) for trace in traces], dim=1) for name in ("spike", "membrane", "adaptation"))
