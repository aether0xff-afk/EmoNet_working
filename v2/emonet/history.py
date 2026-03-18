from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import torch


@dataclass
class HistoryBuffer:
    history_dim: int
    timesteps: List[torch.Tensor] = field(default_factory=list)

    def reset(self) -> None:
        self.timesteps = []

    def add_timestep(self, vec: torch.Tensor) -> None:
        vec = vec.flatten().float()
        if vec.numel() != self.history_dim:
            raise ValueError(f"Expected history dim {self.history_dim}, got {vec.numel()}")
        self.timesteps.append(vec)

    def stacked(self, device: torch.device | None = None) -> torch.Tensor:
        if not self.timesteps:
            out = torch.zeros(1, self.history_dim, dtype=torch.float32)
        else:
            out = torch.stack(self.timesteps, dim=0)
        return out.to(device) if device is not None else out
