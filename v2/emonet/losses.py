from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def control_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred, target)


def tone_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.huber_loss(pred, target)


def style_consistency_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.huber_loss(pred, target)


def summarize_losses(losses: Dict[str, torch.Tensor], weights: Dict[str, float]) -> torch.Tensor:
    total = torch.tensor(0.0, device=next(iter(losses.values())).device if losses else "cpu")
    for name, val in losses.items():
        total = total + weights.get(name, 1.0) * val
    return total
