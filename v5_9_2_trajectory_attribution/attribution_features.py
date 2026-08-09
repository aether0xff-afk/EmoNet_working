from __future__ import annotations

from collections.abc import Sequence

import numpy as np


PAIR_INDICES = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / denominator)


def pairwise_cosines(vectors: Sequence[np.ndarray]) -> np.ndarray:
    if len(vectors) != 4:
        raise ValueError(f"expected four vectors, got {len(vectors)}")
    return np.asarray(
        [cosine(vectors[i], vectors[j]) for i, j in PAIR_INDICES],
        dtype=np.float32,
    )


def trace_pairwise_cosines(trace_states: Sequence[np.ndarray]) -> np.ndarray:
    """Pairwise cosines among four complete event traces."""
    return pairwise_cosines([np.asarray(states).reshape(-1) for states in trace_states])


def geometry_agreement(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    return cosine(a, b), float(np.linalg.norm(a - b))
