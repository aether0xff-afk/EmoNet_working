"""Surrogate-gradient spike functions."""

from __future__ import annotations

import torch


class FastSigmoidSpike(torch.autograd.Function):
    """Binary spike in forward pass with a smooth surrogate gradient."""

    @staticmethod
    def forward(ctx: object, x: torch.Tensor, slope: float = 25.0) -> torch.Tensor:
        ctx.save_for_backward(x)
        ctx.slope = slope
        return (x >= 0).to(x.dtype)

    @staticmethod
    def backward(ctx: object, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        (x,) = ctx.saved_tensors
        slope = ctx.slope
        grad = 1.0 / (1.0 + slope * x.abs()) ** 2
        return grad_output * grad, None


def spike_with_surrogate_gradient(x: torch.Tensor) -> torch.Tensor:
    """Return binary spikes while preserving an approximate gradient."""

    return FastSigmoidSpike.apply(x)
