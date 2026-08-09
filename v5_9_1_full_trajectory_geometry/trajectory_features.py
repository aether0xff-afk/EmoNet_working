from __future__ import annotations

from collections.abc import Sequence

import numpy as np


PAIR_INDICES = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
EPISODE_HASH_DIM = 256
EPISODE_HASH_SEED = 5_091_2026


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


def hashed_full_episode(
    history_observations: Sequence[object],
    current_observation: object,
    *,
    output_dim: int = EPISODE_HASH_DIM,
    seed: int = EPISODE_HASH_SEED,
) -> np.ndarray:
    """Deterministic label-free signed feature hash of the full raw episode.

    This diagnostic preserves contributions from every raw trajectory coordinate
    while avoiding repeated 14k-dimensional ridge Gram matrices. Bucket/sign
    assignments depend only on coordinate index and the frozen seed.
    """
    raw = full_episode_raw(history_observations, current_observation)
    index = np.arange(raw.size, dtype=np.uint64)
    mixed = index * np.uint64(0x9E3779B185EBCA87) + np.uint64(seed)
    buckets = np.asarray(mixed % np.uint64(output_dim), dtype=np.int64)
    signs = np.where(((mixed >> np.uint64(17)) & np.uint64(1)) == 0, 1.0, -1.0)
    projected = np.bincount(
        buckets,
        weights=raw.astype(np.float64, copy=False) * signs,
        minlength=output_dim,
    )
    return (projected / np.sqrt(max(raw.size / output_dim, 1.0))).astype(np.float32)


def current_raw(current_observation: object) -> np.ndarray:
    return current_observation.fast_trace.states.reshape(-1).astype(np.float32, copy=False)
