from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
VERSION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(VERSION_ROOT))
sys.path.insert(0, str(VERSION_ROOT / "experiments"))
sys.path.insert(0, str(REPO_ROOT / "v5_7_residual_fast_dynamics"))
sys.path.insert(0, str(REPO_ROOT / "v5_8_adaptive_fast_dynamics"))
sys.path.insert(0, str(REPO_ROOT / "v5_9_2_trajectory_attribution"))

from emonet_v5 import DynamicsConfig
from residual_state import ResidualDrivenState
from hidden_prior_world import (
    DELAYS,
    INPUT_DIM,
    PAIR_COUNT,
    PRIMARY_TASK if False else TASKS,
    TRAIN_PAIRS,
    WORLD_SEEDS,
    build_case,
    build_world,
    ema_state,
)
from run_hidden_prior_benchmark import prepare_hidden, visible_features


PRIMARY = "norm_matched_repeat"


def small_config(seed: int) -> DynamicsConfig:
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


def test_pair_vectors_are_nine_orthonormal_unit_vectors() -> None:
    encoder = build_world(WORLD_SEEDS[0])
    case = build_case(PRIMARY, 55)
    keys = [
        *sorted(set(case.hidden0) | set(case.hidden1)),
        *case.visible,
        case.delay_event,
        case.final_event,
    ]
    # Hidden union is P/Q/R, then A/B/C/D/N/Z = 9 unique vectors.
    assert len(keys) == 9
    vectors = np.stack([encoder.encode(key) for key in keys])
    assert vectors.shape == (9, INPUT_DIM)
    np.testing.assert_allclose(vectors @ vectors.T, np.eye(9), atol=1e-5)


def test_hidden_classes_share_multiset_and_visible_window_exactly() -> None:
    for task in TASKS:
        case = build_case(task, 55)
        assert Counter(case.hidden0) == Counter(case.hidden1)
        assert case.visible == build_case(task, 55).visible
        assert len(case.visible) == 4


def test_primary_slow_ema_norm_is_matched_after_all_delays() -> None:
    encoder = build_world(WORLD_SEEDS[0])
    for pair_id in (0, TRAIN_PAIRS - 1, TRAIN_PAIRS, PAIR_COUNT - 1):
        case = build_case(PRIMARY, pair_id)
        for delay in DELAYS:
            tail = (case.delay_event,) * delay
            state0 = ema_state((*case.hidden0, *tail), encoder, decay=0.80)
            state1 = ema_state((*case.hidden1, *tail), encoder, decay=0.80)
            assert abs(float(np.linalg.norm(state0)) - float(np.linalg.norm(state1))) < 2e-6


def test_visible_input_vectors_are_identical_between_labels() -> None:
    encoder = build_world(WORLD_SEEDS[1])
    case = build_case(PRIMARY, 60)
    visible0 = np.stack([encoder.encode(key) for key in case.visible])
    visible1 = np.stack([encoder.encode(key) for key in case.visible])
    np.testing.assert_allclose(visible0, visible1, atol=0.0)
    np.testing.assert_allclose(visible0 @ visible0.T, np.eye(4), atol=1e-5)


def test_both_reset_makes_visible_neural_trajectory_identical_across_hidden_labels() -> None:
    encoder = build_world(WORLD_SEEDS[0])
    model = ResidualDrivenState(
        encoder,
        seed=17,
        slow_decay=0.80,
        dynamics_config=small_config(17),
    )
    case = build_case(PRIMARY, 55)
    snap0 = prepare_hidden(model, case.hidden0, case.delay_event, 1)
    out0 = visible_features(model, snap0, case.visible, case.final_event, "both")
    snap1 = prepare_hidden(model, case.hidden1, case.delay_event, 1)
    out1 = visible_features(model, snap1, case.visible, case.final_event, "both")
    np.testing.assert_allclose(out0["raw"], out1["raw"], atol=0.0)
    np.testing.assert_allclose(out0["selfsim"], out1["selfsim"], atol=0.0)
