from __future__ import annotations

import torch

from emonet_v7.adaptive_rsnn import AdaptiveSparseRSNN, create_recurrent_mask


def build_model(seed: int = 42) -> AdaptiveSparseRSNN:
    return AdaptiveSparseRSNN(
        num_neurons=32,
        recurrent_density=0.10,
        seed=seed,
        threshold_base=1.0,
        adaptation_strength=0.4,
        membrane_decay_min=0.80,
        membrane_decay_max=0.95,
        adaptation_decay_min=0.90,
        adaptation_decay_max=0.995,
        recurrent_weight_std=0.12,
        input_weight_std=0.10,
    )


def test_mask_has_no_self_loops_and_is_deterministic() -> None:
    first = create_recurrent_mask(128, 0.10, seed=7)
    second = create_recurrent_mask(128, 0.10, seed=7)
    assert torch.equal(first, second)
    assert torch.count_nonzero(torch.diag(first)) == 0
    assert 0.07 < float(first.mean()) < 0.13


def test_state_shapes() -> None:
    model = build_model()
    state = model.initial_state(batch_size=3, device="cpu")
    assert state.membrane.shape == (3, 32)
    assert state.spike.shape == (3, 32)
    assert state.adaptation.shape == (3, 32)
    assert state.threshold.shape == (3, 32)


def test_adaptation_increases_after_forced_spike() -> None:
    model = build_model()
    state = model.initial_state(batch_size=1, device="cpu")
    state.spike[:] = 1.0
    current = torch.zeros(1, 32)
    next_state = model.step(current, state)
    assert torch.all(next_state.adaptation > 0)
    assert torch.all(next_state.threshold > model.threshold_base)


def test_active_edges_use_source_target_order() -> None:
    model = AdaptiveSparseRSNN(num_neurons=3, recurrent_density=0.0, seed=1)
    model.recurrent_mask.zero_()
    model.recurrent_mask[1, 0] = 1.0  # stored as [target=1, source=0]
    previous = torch.tensor([[1.0, 0.0, 0.0]])
    current = torch.tensor([[0.0, 1.0, 0.0]])
    active = model.active_edges(previous, current)
    assert active.shape == (1, 3, 3)
    assert active[0, 0, 1] == 1.0  # logged as [source=0, target=1]
    assert active[0, 1, 0] == 0.0


def test_window_returns_expected_trace_count() -> None:
    model = build_model()
    state = model.initial_state(batch_size=1, device="cpu")
    current = torch.ones(1, 32)
    _, traces = model.run_window(
        event_current=current,
        state=state,
        event_ticks=16,
        stimulation_ticks=4,
    )
    assert len(traces) == 16
    assert traces[0].spike.shape == (1, 32)
    assert traces[0].active_edges.shape == (1, 32, 32)
