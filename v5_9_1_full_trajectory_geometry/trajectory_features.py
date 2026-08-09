from __future__ import annotations

from collections.abc import Sequence

import numpy as np


PAIR_INDICES = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / denominator)


def event_trace_similarity(observations: Sequence[object]) -> np.ndarray:
    """Six pairwise cosines among four complete event traces."""
    if len(observations) != 4:
        raise ValueError(f"expected four transient observations, got {len(observations)}")
    vectors = [obs.fast_trace.states.reshape(-1) for obs in observations]
    return np.asarray([_cosine(vectors[i], vectors[j]) for i, j in PAIR_INDICES], dtype=np.float32)


def event_final_state_similarity(observations: Sequence[object]) -> np.ndarray:
    """Six pairwise cosines among the four event-final neural states."""
    if len(observations) != 4:
        raise ValueError(f"expected four transient observations, got {len(observations)}")
    vectors = [obs.fast_trace.states[-1] for obs in observations]
    return np.asarray([_cosine(vectors[i], vectors[j]) for i, j in PAIR_INDICES], dtype=np.float32)


def event_mean_state_similarity(observations: Sequence[object]) -> np.ndarray:
    """Six pairwise cosines among tick-mean neural states for four events."""
    if len(observations) != 4:
        raise ValueError(f"expected four transient observations, got {len(observations)}")
    vectors = [obs.fast_trace.states.mean(axis=0) for obs in observations]
    return np.asarray([_cosine(vectors[i], vectors[j]) for i, j in PAIR_INDICES], dtype=np.float32)


def full_episode_raw(history_observations: Sequence[object], current_observation: object) -> np.ndarray:
    """Concatenate prefix, four transients, suffix, and current traces."""
    if len(history_observations) != 6:
        raise ValueError(f"expected six history observations, got {len(history_observations)}")
    observations = [*history_observations, current_observation]
    return np.concatenate(
        [obs.fast_trace.states.reshape(-1) for obs in observations]
    ).astype(np.float32, copy=False)


def current_raw(current_observation: object) -> np.ndarray:
    return current_observation.fast_trace.states.reshape(-1).astype(np.float32, copy=False)
