from __future__ import annotations

import torch

from emonet_v7.surrogate import spike_with_surrogate_gradient


def test_surrogate_spike_forward_and_backward() -> None:
    x = torch.tensor([-1.0, 0.0, 1.0], requires_grad=True)
    spikes = spike_with_surrogate_gradient(x)
    assert torch.equal(spikes, torch.tensor([0.0, 1.0, 1.0]))
    spikes.sum().backward()
    assert x.grad is not None
    assert torch.all(torch.isfinite(x.grad))
