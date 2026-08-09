from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
VERSION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(VERSION_ROOT))
sys.path.insert(0, str(REPO_ROOT / "v5_7_residual_fast_dynamics"))

from adaptive_state import AdaptiveResidualState
from temporal_suite import PAIR_COUNT, TASKS, TRAIN_PAIRS, build_case
from emonet_v5 import DynamicsConfig, HashingTextEncoder
from v5_7_residual_fast_dynamics.residual_state import ResidualDrivenState


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


def test_beta_zero_reproduces_v57_tick_by_tick() -> None:
    encoder = HashingTextEncoder(dimension=24)
    adaptive = AdaptiveResidualState(
        encoder,
        seed=17,
        adaptation_strength=0.0,
        adaptation_decay=0.985,
        dynamics_config=config(17),
    )
    frozen = ResidualDrivenState(
        encoder,
        seed=17,
        dynamics_config=config(17),
    )
    sequence = ["alpha event", "beta event", "alpha event", "common current"]
    for text in sequence:
        a = adaptive.consume_event(text)
        b = frozen.consume_event(text)
        np.testing.assert_allclose(a.fast_trace.states, b.fast_trace.states, atol=1e-7)
        np.testing.assert_allclose(a.slow_state, b.slow_state, atol=1e-7)
        np.testing.assert_allclose(a.residual_input, b.residual_input, atol=1e-7)


def test_fast_reset_clears_activity_and_adaptation_but_preserves_slow() -> None:
    encoder = HashingTextEncoder(dimension=24)
    model = AdaptiveResidualState(
        encoder,
        seed=19,
        adaptation_strength=0.5,
        adaptation_decay=0.985,
        dynamics_config=config(19),
    )
    model.consume_sequence(["first event", "second event", "third event"])
    slow_before = model.slow.state.copy()
    assert float(np.linalg.norm(model.fast.state)) > 0.0
    assert float(np.linalg.norm(model.fast.adaptation)) > 0.0
    model.reset_fast()
    assert np.count_nonzero(model.fast.state) == 0
    assert np.count_nonzero(model.fast.adaptation) == 0
    np.testing.assert_allclose(model.slow.state, slow_before, atol=0.0)


def test_adaptation_only_removes_recurrent_matrix() -> None:
    encoder = HashingTextEncoder(dimension=24)
    model = AdaptiveResidualState(
        encoder,
        seed=23,
        adaptation_strength=0.5,
        adaptation_decay=0.985,
        use_recurrence=False,
        dynamics_config=config(23),
    )
    assert np.count_nonzero(model.fast.recurrent_weight) == 0
    assert np.count_nonzero(model.fast.input_weight) > 0


def test_temporal_suite_keeps_class_multisets_equal() -> None:
    for task in TASKS:
        for pair_id in (0, TRAIN_PAIRS - 1, TRAIN_PAIRS, PAIR_COUNT - 1):
            case = build_case(task, pair_id)
            assert Counter(case.class0[1:5]) == Counter(case.class1[1:5])
            assert case.class0[0] == case.class1[0]
            assert case.class0[-1] == case.class1[-1]


def test_train_test_event_identities_are_disjoint() -> None:
    train: set[str] = set()
    test: set[str] = set()
    for task in TASKS:
        for pair_id in range(PAIR_COUNT):
            case = build_case(task, pair_id)
            target = train if pair_id < TRAIN_PAIRS else test
            target.update(case.identities)
    assert train.isdisjoint(test)
