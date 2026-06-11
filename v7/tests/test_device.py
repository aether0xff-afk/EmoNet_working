from __future__ import annotations

import pytest
import torch

from emonet_v7.device import resolve_device


def test_auto_uses_cuda_when_available() -> None:
    device, used_fallback = resolve_device("auto", cuda_available=lambda: True)

    assert device == torch.device("cuda")
    assert used_fallback is False


def test_auto_falls_back_to_cpu_when_cuda_is_unavailable() -> None:
    device, used_fallback = resolve_device("auto", cuda_available=lambda: False)

    assert device == torch.device("cpu")
    assert used_fallback is True


def test_cuda_request_falls_back_to_cpu_when_allowed() -> None:
    device, used_fallback = resolve_device("cuda", cuda_available=lambda: False, allow_cuda_fallback=True)

    assert device == torch.device("cpu")
    assert used_fallback is True


def test_cuda_request_fails_when_fallback_is_disabled() -> None:
    with pytest.raises(RuntimeError, match="CUDA was requested"):
        resolve_device("cuda", cuda_available=lambda: False, allow_cuda_fallback=False)
