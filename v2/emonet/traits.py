from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class TraitState:
    p: torch.Tensor

    @classmethod
    def zeros(cls, d_p: int, device: Optional[torch.device] = None) -> "TraitState":
        return cls(torch.zeros(d_p, dtype=torch.float32, device=device))

    def update_ema(self, p_hat: torch.Tensor, eta: float = 0.01) -> torch.Tensor:
        self.p = (1.0 - eta) * self.p + eta * p_hat.detach()
        return self.p


@dataclass
class MemoryState:
    episode: torch.Tensor
    persistent: torch.Tensor

    @classmethod
    def zeros(cls, size: int, device: Optional[torch.device] = None) -> "MemoryState":
        z = torch.zeros(size, dtype=torch.float32, device=device)
        return cls(episode=z.clone(), persistent=z.clone())

    @property
    def total(self) -> torch.Tensor:
        return self.episode + self.persistent

    def reset_episode(self) -> None:
        self.episode.zero_()

    def update_episode(self, delta: torch.Tensor, decay: float = 0.85) -> torch.Tensor:
        self.episode = torch.tanh(decay * self.episode + delta)
        return self.episode

    def update_persistent(self, delta: torch.Tensor, rate: float = 0.01, decay: float = 0.999) -> torch.Tensor:
        self.persistent = decay * self.persistent + rate * torch.tanh(delta)
        return self.persistent
