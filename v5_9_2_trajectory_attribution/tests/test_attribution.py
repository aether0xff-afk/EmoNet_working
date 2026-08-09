from __future__ import annotations

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
sys.path.insert(0, str(REPO_ROOT / "v5_9_encoder_free_temporal"))

from attribution_features import geometry_agreement, pairwise_cosines, trace_pairwise_cosines
from emonet_v5 import DynamicsConfig
from emonet_v5.dynamics import FixedRecurrentDynamics
from adaptive_state import AdaptiveFastDynamics
from vector_world import build_case, build_vector_world


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


def test_raw_input_relation_matches_latent_abab_structure() -> None:
    encoder = build_vector_world(101)
    case = build_case("alternation", 83)
    x0 = [encoder.encode(key) for key in case.class0[1:5]]
    x1 = [encoder.encode(key) for key in case.class1[1:5]]
    np.testing.assert_allclose(
        pairwise_cosines(x0),
        np.asarray([0.0, 1.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32),
        atol=1e-5,
    )
    np.testing.assert_allclose(
        pairwise_cosines(x1),
        np.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        atol=1e-5,
    )


def test_isolated_dynamics_use_same_seeded_input_and_recurrent_weights() -> None:
    config = small_config(17)
    fixed_a = FixedRecurrentDynamics(input_dim=24, config=config)
    fixed_b = FixedRecurrentDynamics(input_dim=24, config=config)
    adaptive = AdaptiveFastDynamics(
        input_dim=24,
        config=config,
        adaptation_strength=0.20,
        adaptation_decay=0.995,
        use_recurrence=True,
    )
    np.testing.assert_allclose(fixed_a.input_weight, fixed_b.input_weight, atol=0.0)
    np.testing.assert_allclose(fixed_a.recurrent_weight, fixed_b.recurrent_weight, atol=0.0)
    np.testing.assert_allclose(fixed_a.input_weight, adaptive.input_weight, atol=0.0)
    np.testing.assert_allclose(fixed_a.recurrent_weight, adaptive.recurrent_weight, atol=0.0)


def test_reset_before_each_event_removes_event_to_event_fast_carry() -> None:
    rng = np.random.default_rng(18)
    config = small_config(18)
    dynamics = FixedRecurrentDynamics(input_dim=24, config=config)
    vectors = [rng.normal(size=24).astype(np.float32) for _ in range(2)]
    dynamics.reset_state()
    first = dynamics.run_event(vectors[0])
    dynamics.reset_state()
    second_after_reset = dynamics.run_event(vectors[1])

    fresh = FixedRecurrentDynamics(input_dim=24, config=config)
    second_fresh = fresh.run_event(vectors[1])
    np.testing.assert_allclose(second_after_reset, second_fresh, atol=0.0)
    assert not np.allclose(first, second_after_reset)


def test_geometry_delta_is_zero_for_identical_geometries() -> None:
    a = np.asarray([1.0, 0.0, 0.5, -0.2, 0.1, 0.7], dtype=np.float32)
    cosine, distance = geometry_agreement(a, a.copy())
    assert abs(cosine - 1.0) < 1e-6
    assert distance == 0.0


def test_trace_pairwise_cosines_returns_six_relations() -> None:
    rng = np.random.default_rng(19)
    traces = [rng.normal(size=(8, 32)).astype(np.float32) for _ in range(4)]
    feature = trace_pairwise_cosines(traces)
    assert feature.shape == (6,)
    assert np.all(np.isfinite(feature))
