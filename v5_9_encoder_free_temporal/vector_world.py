from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np


TASKS = ("alternation", "palindrome", "repeat_gap", "repeat_position")
WORLD_SEEDS = (101, 211, 307)
RECURRENT_SEEDS = (7, 13, 21, 42, 100)
PAIR_COUNT = 120
TRAIN_PAIRS = 80
INPUT_DIM = 384


@dataclass(frozen=True)
class VectorCase:
    task: str
    pair_id: int
    class0: tuple[str, ...]
    class1: tuple[str, ...]
    current: str


class LookupVectorEncoder:
    """A deterministic lookup encoder with no language or learned semantics."""

    def __init__(self, mapping: dict[str, np.ndarray], output_dim: int = INPUT_DIM) -> None:
        self.mapping = {
            key: np.asarray(value, dtype=np.float32).reshape(output_dim).copy()
            for key, value in mapping.items()
        }
        self.output_dim = int(output_dim)

    def encode(self, text: str) -> np.ndarray:
        try:
            return self.mapping[text].copy()
        except KeyError as exc:
            raise KeyError(f"unknown vector-world event key: {text}") from exc


def _orthonormal_vectors(seed: int, count: int, dimension: int = INPUT_DIM) -> np.ndarray:
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(dimension, count))
    q, _ = np.linalg.qr(matrix)
    return q[:, :count].T.astype(np.float32)


def _event_seed(world_seed: int, task_index: int, pair_id: int) -> int:
    return int(world_seed * 10_000_000 + task_index * 100_000 + pair_id)


def build_vector_world(world_seed: int) -> LookupVectorEncoder:
    if world_seed not in WORLD_SEEDS:
        raise ValueError(f"unregistered vector world seed: {world_seed}")

    mapping: dict[str, np.ndarray] = {}
    neutral = _orthonormal_vectors(world_seed * 10_000_000 + 9_999_991, 3)
    mapping["neutral/prefix"] = neutral[0]
    mapping["neutral/suffix"] = neutral[1]
    mapping["neutral/current"] = neutral[2]

    for task_index, task in enumerate(TASKS):
        for pair_id in range(PAIR_COUNT):
            vectors = _orthonormal_vectors(_event_seed(world_seed, task_index, pair_id), 3)
            for side, name in enumerate(("A", "B", "C")):
                mapping[f"{task}/{pair_id:03d}/{name}"] = vectors[side]
    return LookupVectorEncoder(mapping)


def build_case(task: str, pair_id: int) -> VectorCase:
    if task not in TASKS:
        raise ValueError(task)
    if not 0 <= pair_id < PAIR_COUNT:
        raise ValueError(pair_id)

    a = f"{task}/{pair_id:03d}/A"
    b = f"{task}/{pair_id:03d}/B"
    c = f"{task}/{pair_id:03d}/C"
    patterns: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
        "alternation": ((a, b, a, b), (a, a, b, b)),
        "palindrome": ((a, b, b, a), (a, a, b, b)),
        "repeat_gap": ((a, b, c, a), (a, a, b, c)),
        "repeat_position": ((a, b, c, a), (a, b, a, c)),
    }
    p0, p1 = patterns[task]
    return VectorCase(
        task=task,
        pair_id=pair_id,
        class0=("neutral/prefix", *p0, "neutral/suffix"),
        class1=("neutral/prefix", *p1, "neutral/suffix"),
        current="neutral/current",
    )


def transient_multiset(case: VectorCase, label: int) -> Counter[str]:
    sequence = case.class0 if label == 0 else case.class1
    return Counter(sequence[1:5])


def relational_features(sequence: tuple[str, ...], encoder: LookupVectorEncoder) -> np.ndarray:
    vectors = np.stack([encoder.encode(key) for key in sequence[1:5]])
    values: list[float] = []
    for i in range(4):
        for j in range(i + 1, 4):
            values.append(float(np.dot(vectors[i], vectors[j])))
    return np.asarray(values, dtype=np.float32)
