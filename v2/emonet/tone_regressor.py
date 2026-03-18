from __future__ import annotations

import torch
from torch import nn

from .config import AppConfig


class ToneRegressor(nn.Module):
    def __init__(self, config: AppConfig, latent_dim: int | None = None) -> None:
        super().__init__()
        self.latent_dim = latent_dim or config.latent.default_latent_dim
        self.net = nn.Sequential(
            nn.Linear(self.latent_dim + config.trait_dim, 256),
            nn.GELU(),
            nn.Dropout(config.text.dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, config.style.num_styles),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor, trait: torch.Tensor) -> torch.Tensor:
        if trait.dim() == 1:
            trait = trait.unsqueeze(0)
        if z.dim() == 1:
            z = z.unsqueeze(0)
        x = torch.cat([z, trait.to(z.device)], dim=-1)
        return self.net(x)
