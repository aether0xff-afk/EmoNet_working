from __future__ import annotations

import numpy as np

from emonet_v5.trace import NeuralTrace


def _states(trace: NeuralTrace) -> np.ndarray:
    return trace.states.astype(np.float32, copy=False)


def activation_energy(trace: NeuralTrace) -> np.ndarray:
    states = _states(trace)
    return np.sqrt(np.mean(states * states, axis=1) + 1e-12).astype(np.float32)


def change_energy(trace: NeuralTrace) -> np.ndarray:
    states = _states(trace)
    previous = np.concatenate([np.zeros_like(states[:1]), states[:-1]], axis=0)
    delta = states - previous
    return np.sqrt(np.mean(delta * delta, axis=1) + 1e-12).astype(np.float32)


def population_moments(trace: NeuralTrace) -> np.ndarray:
    states = _states(trace)
    mean = states.mean(axis=1)
    std = states.std(axis=1)
    mean_abs = np.abs(states).mean(axis=1)
    rms = np.sqrt(np.mean(states * states, axis=1) + 1e-12)
    return np.stack([mean, std, mean_abs, rms], axis=1).reshape(-1).astype(np.float32)


def full_geometry(trace: NeuralTrace) -> np.ndarray:
    return np.concatenate(
        [activation_energy(trace), change_energy(trace), population_moments(trace)]
    ).astype(np.float32, copy=False)
