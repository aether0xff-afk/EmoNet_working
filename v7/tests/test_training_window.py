from __future__ import annotations

import torch

from emonet_v7.adaptive_rsnn import AdaptiveSparseRSNN
from emonet_v7.trace_encoder import TraceEncoder
from emonet_v7.training_window import run_differentiable_window


def test_differentiable_window_preserves_gradient_path() -> None:
    torch.manual_seed(3)
    snn = AdaptiveSparseRSNN(
        num_neurons=8,
        recurrent_density=0.25,
        seed=3,
        input_weight_std=0.20,
        recurrent_weight_std=0.20,
    )
    trace_encoder = TraceEncoder(num_neurons=8, hidden_dim=8, output_dim=4)
    state = snn.initial_state(batch_size=1, device="cpu")
    current = torch.ones(1, 8, requires_grad=True)

    _, window = run_differentiable_window(
        snn=snn,
        event_current=current,
        state=state,
        event_ticks=6,
        stimulation_ticks=3,
    )
    latent = trace_encoder(window.spike, window.membrane, window.adaptation)
    latent.sum().backward()

    assert current.grad is not None
    assert torch.all(torch.isfinite(current.grad))
    assert snn.input_weight.grad is not None
    assert torch.all(torch.isfinite(snn.input_weight.grad))
