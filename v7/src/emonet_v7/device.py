"""Torch device selection helpers for experiment entrypoints."""

from __future__ import annotations

from collections.abc import Callable

import torch


def resolve_device(
    requested: str,
    *,
    cuda_available: Callable[[], bool] = torch.cuda.is_available,
    allow_cuda_fallback: bool = True,
) -> tuple[torch.device, bool]:
    """Resolve a requested device and report whether CPU fallback was used."""

    normalized = requested.strip().lower()
    if normalized == "auto":
        return (torch.device("cuda"), False) if cuda_available() else (torch.device("cpu"), True)

    device = torch.device(normalized)
    if device.type == "cuda" and not cuda_available():
        if not allow_cuda_fallback:
            raise RuntimeError("CUDA was requested but is not available")
        return torch.device("cpu"), True
    return device, False
