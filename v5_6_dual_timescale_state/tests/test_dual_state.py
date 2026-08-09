from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve()
VERSION_ROOT = HERE.parents[1]
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(VERSION_ROOT))
sys.path.insert(0, str(REPO_ROOT / "v5_clean" / "src"))

from dual_state import DualTimescaleState, SlowEMAMemory, dual_features  # noqa: E402
from emonet_v5 import HashingTextEncoder  # noqa: E402


def build_model() -> DualTimescaleState:
    return DualTimescaleState(HashingTextEncoder(dimension=16), seed=7, slow_decay=0.8)


def test_slow_memory_matches_declared_ema_equation() -> None:
    memory = SlowEMAMemory(3, decay=0.8)
    a = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    b = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
    memory.update(a)
    memory.update(b)
    expected_raw = 0.8 * (0.2 * a) + 0.2 * b
    expected = expected_raw / np.linalg.norm(expected_raw)
    assert np.allclose(memory.read(), expected.astype(np.float32))


def test_reset_fast_preserves_slow_memory() -> None:
    model = build_model()
    model.reset_all()
    model.consume_sequence(["first", "second"])
    slow_before = model.slow.read()
    model.reset_fast()
    slow_after = model.slow.read()
    assert np.array_equal(slow_before, slow_after)
    assert np.allclose(model.fast.dynamics.state, 0.0)


def test_reset_slow_preserves_fast_state() -> None:
    model = build_model()
    model.reset_all()
    model.consume_sequence(["first", "second"])
    fast_before = model.fast.dynamics.snapshot().state
    model.reset_slow()
    fast_after = model.fast.dynamics.snapshot().state
    assert np.array_equal(fast_before, fast_after)
    assert np.allclose(model.slow.state, 0.0)


def test_dual_features_contain_both_timescales() -> None:
    model = build_model()
    model.reset_all()
    observation = model.consume_event("current event")
    combined = dual_features(observation)
    expected = observation.fast_trace.states.size + model.encoder.output_dim
    assert combined.shape == (expected,)
