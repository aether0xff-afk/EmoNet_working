from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve()
VERSION_ROOT = HERE.parents[1]
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(VERSION_ROOT))
sys.path.insert(0, str(REPO_ROOT / "v5_6_dual_timescale_state"))

from residual_state import ResidualDrivenState  # noqa: E402
from dual_state import DualTimescaleState  # noqa: E402
from emonet_v5 import HashingTextEncoder  # noqa: E402


def build_pair(seed: int = 7):
    encoder = HashingTextEncoder(dimension=24)
    residual = ResidualDrivenState(encoder, seed=seed, slow_decay=0.8)
    raw = DualTimescaleState(encoder, seed=seed, slow_decay=0.8)
    return encoder, residual, raw


def test_fast_topology_matches_raw_v5_0_for_same_seed() -> None:
    _, residual, raw = build_pair(seed=21)
    assert np.array_equal(residual.fast.recurrent_weight, raw.fast.dynamics.recurrent_weight)
    assert np.array_equal(residual.fast.input_weight, raw.fast.dynamics.input_weight)


def test_residual_is_computed_against_previous_slow_state() -> None:
    encoder, residual, _ = build_pair()
    residual.reset_all()

    first_embedding = encoder.encode("first event")
    first = residual.consume_event("first event")
    assert np.allclose(first.residual_input, first_embedding)

    slow_before_second = residual.slow.state.copy()
    second_embedding = encoder.encode("second event")
    second = residual.consume_event("second event")
    assert np.allclose(second.residual_input, second_embedding - slow_before_second)


def test_slow_memory_is_identical_to_v5_6_for_same_sequence() -> None:
    _, residual, raw = build_pair()
    sequence = ["one", "two", "three", "four"]
    residual.reset_all()
    raw.reset_all()
    residual.consume_sequence(sequence)
    raw.consume_sequence(sequence)
    assert np.allclose(residual.slow.state, raw.slow.state)
    assert np.allclose(residual.slow.read(), raw.slow.read())


def test_fast_reset_preserves_slow_context() -> None:
    _, residual, _ = build_pair()
    residual.consume_sequence(["one", "two", "three"])
    slow_before = residual.slow.state.copy()
    residual.reset_fast()
    assert np.array_equal(slow_before, residual.slow.state)
    assert np.allclose(residual.fast.state, 0.0)
