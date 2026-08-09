from __future__ import annotations

import inspect
from pathlib import Path
import sys

import torch

VERSION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VERSION_ROOT))

from learned_core import LearnedCoreConfig, LearnedLeakyRecurrentCore  # noqa: E402


def build_core(seed: int = 7) -> LearnedLeakyRecurrentCore:
    return LearnedLeakyRecurrentCore(
        input_dim=16,
        config=LearnedCoreConfig(
            hidden_dim=12,
            event_ticks=6,
            stimulation_ticks=2,
            max_lag=3,
        ),
        seed=seed,
    )


def test_trace_shape_and_reset_are_explicit() -> None:
    core = build_core()
    embeddings = torch.randn(4, 5, 16)
    real = core.final_event_trace(embeddings)
    reset = core.reset_final_event_trace(embeddings)
    assert real.shape == (4, 6, 12)
    assert reset.shape == (4, 6, 12)
    assert not torch.equal(real, reset)


def test_delayed_memory_loss_reaches_recurrent_parameters() -> None:
    core = build_core()
    embeddings = torch.randn(5, 5, 16)
    loss, diagnostics = core.delayed_memory_loss(embeddings)
    loss.backward()
    assert set(diagnostics) == {1, 2, 3}
    assert core.input_weight.grad is not None
    assert core.recurrent_weight.grad is not None
    assert float(core.recurrent_weight.grad.abs().sum()) > 0.0


def test_same_seed_is_deterministic() -> None:
    a = build_core(seed=21)
    b = build_core(seed=21)
    assert torch.equal(a.input_weight, b.input_weight)
    assert torch.equal(a.recurrent_weight, b.recurrent_weight)


def test_core_training_api_has_no_task_label_argument() -> None:
    signature = inspect.signature(LearnedLeakyRecurrentCore.delayed_memory_loss)
    assert list(signature.parameters) == ["self", "embeddings"]
