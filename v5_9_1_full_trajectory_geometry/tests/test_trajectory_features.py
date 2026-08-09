from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
VERSION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(VERSION_ROOT))

from emonet_v5.trace import NeuralTrace
from trajectory_features import (
    EPISODE_HASH_DIM,
    event_final_state_similarity,
    event_mean_state_similarity,
    event_trace_similarity,
    hashed_full_episode,
)


def observation(states: np.ndarray, event_index: int = 0):
    return SimpleNamespace(
        fast_trace=NeuralTrace(states=np.asarray(states, dtype=np.float32), event_index=event_index)
    )


def repeated_event(vector: np.ndarray, ticks: int = 4):
    return observation(np.repeat(np.asarray(vector, dtype=np.float32)[None, :], ticks, axis=0))


def test_abab_and_aabb_have_expected_trace_similarity_patterns() -> None:
    a = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    b = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
    abab = [repeated_event(v) for v in (a, b, a, b)]
    aabb = [repeated_event(v) for v in (a, a, b, b)]
    np.testing.assert_allclose(
        event_trace_similarity(abab),
        np.asarray([0.0, 1.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        event_trace_similarity(aabb),
        np.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        atol=1e-6,
    )


def test_similarity_readouts_are_invariant_to_joint_neuron_permutation() -> None:
    rng = np.random.default_rng(591)
    states = [rng.normal(size=(7, 24)).astype(np.float32) for _ in range(4)]
    permutation = rng.permutation(24)
    original = [observation(s, i) for i, s in enumerate(states)]
    permuted = [observation(s[:, permutation], i) for i, s in enumerate(states)]

    np.testing.assert_allclose(
        event_trace_similarity(original), event_trace_similarity(permuted), atol=1e-6
    )
    np.testing.assert_allclose(
        event_final_state_similarity(original),
        event_final_state_similarity(permuted),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        event_mean_state_similarity(original),
        event_mean_state_similarity(permuted),
        atol=1e-6,
    )


def test_hashed_episode_is_deterministic_and_fixed_width() -> None:
    rng = np.random.default_rng(592)
    history = [observation(rng.normal(size=(5, 16)).astype(np.float32), i) for i in range(6)]
    current = observation(rng.normal(size=(5, 16)).astype(np.float32), 6)
    a = hashed_full_episode(history, current)
    b = hashed_full_episode(history, current)
    assert a.shape == (EPISODE_HASH_DIM,)
    np.testing.assert_allclose(a, b, atol=0.0)


def test_hashed_episode_changes_when_trajectory_changes() -> None:
    history = [observation(np.zeros((4, 8), dtype=np.float32), i) for i in range(6)]
    current_a = observation(np.zeros((4, 8), dtype=np.float32), 6)
    current_b_states = np.zeros((4, 8), dtype=np.float32)
    current_b_states[2, 3] = 1.0
    current_b = observation(current_b_states, 6)
    assert not np.allclose(
        hashed_full_episode(history, current_a),
        hashed_full_episode(history, current_b),
    )
