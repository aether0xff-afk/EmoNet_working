from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

VERSION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VERSION_ROOT))

from emonet_v5.trace import NeuralTrace
from geometry import activation_energy, change_energy, full_geometry, population_moments


def trace(states: np.ndarray) -> NeuralTrace:
    return NeuralTrace(states=states.astype(np.float32), event_index=0)


def test_geometry_is_invariant_to_neuron_permutation() -> None:
    rng = np.random.default_rng(7)
    states = rng.normal(size=(12, 32)).astype(np.float32)
    perm = rng.permutation(states.shape[1])
    a = trace(states)
    b = trace(states[:, perm])
    np.testing.assert_allclose(activation_energy(a), activation_energy(b), atol=1e-6)
    np.testing.assert_allclose(change_energy(a), change_energy(b), atol=1e-6)
    np.testing.assert_allclose(population_moments(a), population_moments(b), atol=1e-6)
    np.testing.assert_allclose(full_geometry(a), full_geometry(b), atol=1e-6)


def test_change_energy_detects_temporal_change() -> None:
    constant = trace(np.ones((8, 16), dtype=np.float32))
    changing_states = np.ones((8, 16), dtype=np.float32)
    changing_states[4:] *= -1.0
    changing = trace(changing_states)
    assert float(change_energy(changing)[4]) > float(change_energy(constant)[4]) + 0.5


def test_geometry_dimensions_depend_on_ticks_not_neuron_count() -> None:
    rng = np.random.default_rng(11)
    a = trace(rng.normal(size=(10, 16)).astype(np.float32))
    b = trace(rng.normal(size=(10, 64)).astype(np.float32))
    assert full_geometry(a).shape == full_geometry(b).shape
