"""Self-supervised next-event objectives for EmoNet v7 training smoke tests."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .training_window import DifferentiableWindow


class NextEventPredictor(nn.Module):
    """Predict the next frozen text embedding from a trace latent vector."""

    def __init__(self, *, latent_dim: int, hidden_dim: int, embedding_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, latent_z: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(latent_z), dim=-1)


@dataclass
class ObjectiveBreakdown:
    total: torch.Tensor
    next_event: torch.Tensor
    firing_rate: torch.Tensor
    inactive_neuron: torch.Tensor
    stability: torch.Tensor


def compute_objective(
    *,
    predicted_embedding: torch.Tensor,
    target_embedding: torch.Tensor,
    window: DifferentiableWindow,
    target_rate: float = 0.10,
    rate_weight: float = 0.10,
    inactive_weight: float = 0.01,
    stability_weight: float = 0.01,
    membrane_limit: float = 5.0,
) -> ObjectiveBreakdown:
    """Return a minimal differentiable training objective."""

    target = F.normalize(target_embedding, dim=-1)
    next_event = (1.0 - F.cosine_similarity(predicted_embedding, target, dim=-1)).mean()
    mean_rate = window.spike.mean()
    firing_rate = (mean_rate - target_rate) ** 2
    neuron_activity = window.spike.sum(dim=(0, 1))
    inactive_neuron = torch.exp(-neuron_activity).mean()
    excess = torch.relu(window.membrane.abs() - membrane_limit)
    stability = (excess ** 2).mean()
    total = next_event + rate_weight * firing_rate + inactive_weight * inactive_neuron + stability_weight * stability
    return ObjectiveBreakdown(
        total=total,
        next_event=next_event,
        firing_rate=firing_rate,
        inactive_neuron=inactive_neuron,
        stability=stability,
    )
