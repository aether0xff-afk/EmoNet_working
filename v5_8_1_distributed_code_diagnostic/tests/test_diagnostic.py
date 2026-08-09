from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
VERSION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(VERSION_ROOT))
sys.path.insert(0, str(REPO_ROOT / "v5_8_adaptive_fast_dynamics"))

from v5_8_adaptive_fast_dynamics.adaptive_state import AdaptiveResidualState
from v5_8_adaptive_fast_dynamics.temporal_suite import TASKS, build_case
from emonet_v5 import DynamicsConfig, HashingTextEncoder


def config(seed: int) -> DynamicsConfig:
    return DynamicsConfig(
        num_neurons=32,
        recurrent_density=0.20,
        spectral_radius=0.90,
        update_rate=0.35,
        input_scale=0.80,
        event_ticks=8,
        stimulation_ticks=3,
        seed=seed,
    )


def test_fixed_v58_parameters_are_not_tuned_here() -> None:
    from v5_8_1_distributed_code_diagnostic.experiments.run_distributed_code_diagnostic import (
        ADAPTATION_DECAY,
        ADAPTATION_STRENGTH,
        SLOW_DECAY,
    )

    assert ADAPTATION_DECAY == 0.995
    assert ADAPTATION_STRENGTH == 0.20
    assert SLOW_DECAY == 0.80


def test_neuron_permutation_preserves_each_tick_norm() -> None:
    rng = np.random.default_rng(5812026)
    states = rng.normal(size=(12, 32)).astype(np.float32)
    permutation = rng.permutation(states.shape[1])
    before = np.linalg.norm(states, axis=1)
    after = np.linalg.norm(states[:, permutation], axis=1)
    np.testing.assert_allclose(before, after, atol=1e-6)


def test_fast_reset_remains_independent_of_slow_state() -> None:
    encoder = HashingTextEncoder(dimension=24)
    model = AdaptiveResidualState(
        encoder,
        seed=31,
        adaptation_strength=0.20,
        adaptation_decay=0.995,
        slow_decay=0.80,
        dynamics_config=config(31),
    )
    model.consume_sequence(["a", "b", "c"])
    slow = model.slow.state.copy()
    model.reset_fast()
    np.testing.assert_allclose(model.slow.state, slow, atol=0.0)
    assert np.count_nonzero(model.fast.state) == 0
    assert np.count_nonzero(model.fast.adaptation) == 0


def test_opposite_histories_share_identity_multiset_and_current() -> None:
    for task in TASKS:
        case = build_case(task, 83)
        assert sorted(case.class0[1:5]) == sorted(case.class1[1:5])
        assert case.current == "The identical current observation is now presented."
