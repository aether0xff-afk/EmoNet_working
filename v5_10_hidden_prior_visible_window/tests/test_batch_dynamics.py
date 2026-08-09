from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
VERSION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(VERSION_ROOT))
sys.path.insert(0, str(REPO_ROOT / "v5_7_residual_fast_dynamics"))
sys.path.insert(0, str(REPO_ROOT / "v5_8_adaptive_fast_dynamics"))

from batch_dynamics import BatchedResidualDynamics
from emonet_v5 import DynamicsConfig, HashingTextEncoder
from residual_state import ResidualDrivenState
from adaptive_state import AdaptiveResidualState


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


def test_batch_v57_matches_scalar_event_by_event() -> None:
    encoder = HashingTextEncoder(dimension=24)
    cfg = config(71)
    scalar = ResidualDrivenState(
        encoder, seed=71, slow_decay=0.80, dynamics_config=cfg
    )
    batch = BatchedResidualDynamics(24, cfg, slow_decay=0.80)
    state = batch.zeros(1)

    for text in ("alpha", "beta", "gamma", "alpha", "current"):
        scalar_obs = scalar.consume_event(text)
        event = encoder.encode(text)[None, :]
        state, traces, residual = batch.run_event(state, event)
        np.testing.assert_allclose(traces[0], scalar_obs.fast_trace.states, atol=2e-6)
        np.testing.assert_allclose(residual[0], scalar_obs.residual_input, atol=2e-6)
        np.testing.assert_allclose(state.fast[0], scalar.fast.state, atol=2e-6)
        np.testing.assert_allclose(state.slow[0], scalar.slow.state, atol=2e-6)


def test_batch_v58_matches_scalar_event_by_event() -> None:
    encoder = HashingTextEncoder(dimension=24)
    cfg = config(73)
    scalar = AdaptiveResidualState(
        encoder,
        seed=73,
        adaptation_strength=0.20,
        adaptation_decay=0.995,
        slow_decay=0.80,
        dynamics_config=cfg,
    )
    batch = BatchedResidualDynamics(
        24,
        cfg,
        slow_decay=0.80,
        adaptation_strength=0.20,
        adaptation_decay=0.995,
    )
    state = batch.zeros(1)

    for text in ("alpha", "beta", "gamma", "alpha", "current"):
        scalar_obs = scalar.consume_event(text)
        event = encoder.encode(text)[None, :]
        state, traces, residual = batch.run_event(state, event)
        np.testing.assert_allclose(traces[0], scalar_obs.fast_trace.states, atol=3e-6)
        np.testing.assert_allclose(residual[0], scalar_obs.residual_input, atol=2e-6)
        np.testing.assert_allclose(state.fast[0], scalar.fast.state, atol=3e-6)
        np.testing.assert_allclose(state.slow[0], scalar.slow.state, atol=2e-6)
        np.testing.assert_allclose(
            state.adaptation[0], scalar.fast.adaptation, atol=3e-6
        )


def test_batch_reset_semantics_match_v510_interventions() -> None:
    cfg = config(79)
    batch = BatchedResidualDynamics(
        24,
        cfg,
        slow_decay=0.80,
        adaptation_strength=0.20,
        adaptation_decay=0.995,
    )
    state = batch.zeros(3)
    rng = np.random.default_rng(79)
    event = rng.normal(size=(3, 24)).astype(np.float32)
    state, _, _ = batch.run_event(state, event)

    fast_reset = batch.reset_fast(state)
    assert np.count_nonzero(fast_reset.fast) == 0
    assert np.count_nonzero(fast_reset.adaptation) == 0
    np.testing.assert_allclose(fast_reset.slow, state.slow, atol=0.0)

    slow_reset = batch.reset_slow(state)
    assert np.count_nonzero(slow_reset.slow) == 0
    np.testing.assert_allclose(slow_reset.fast, state.fast, atol=0.0)
    np.testing.assert_allclose(slow_reset.adaptation, state.adaptation, atol=0.0)

    both = batch.reset_both(state)
    assert np.count_nonzero(both.fast) == 0
    assert np.count_nonzero(both.slow) == 0
    assert np.count_nonzero(both.adaptation) == 0
