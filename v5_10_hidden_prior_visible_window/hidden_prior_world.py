from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np


WORLD_SEEDS = (101, 211, 307)
RECURRENT_SEEDS = (7, 13, 21, 42, 100)
TASKS = ("norm_matched_repeat", "easy_alternation")
DELAYS = (0, 1, 3)
PAIR_COUNT = 80
TRAIN_PAIRS = 50
INPUT_DIM = 384
SYMBOLS = ("P", "Q", "R", "A", "B", "C", "D", "N", "Z")


@dataclass(frozen=True)
class HiddenPriorCase:
    task: str
    pair_id: int
    hidden0: tuple[str, ...]
    hidden1: tuple[str, ...]
    visible: tuple[str, ...]
    delay_event: str
    final_event: str


class LookupVectorEncoder:
    """Deterministic vector lookup; key strings never enter a language model."""

    def __init__(self, mapping: dict[str, np.ndarray]) -> None:
        self.mapping = {
            key: np.asarray(value, dtype=np.float32).reshape(INPUT_DIM).copy()
            for key, value in mapping.items()
        }
        self.output_dim = INPUT_DIM

    def encode(self, key: str) -> np.ndarray:
        return self.mapping[key].copy()


def _orthonormal_vectors(seed: int, count: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(INPUT_DIM, count))
    q, _ = np.linalg.qr(matrix)
    return q[:, :count].T.astype(np.float32)


def _pair_seed(world_seed: int, task_index: int, pair_id: int) -> int:
    return int(world_seed * 10_000_000 + task_index * 100_000 + pair_id)


def build_world(world_seed: int) -> LookupVectorEncoder:
    if world_seed not in WORLD_SEEDS:
        raise ValueError(world_seed)
    mapping: dict[str, np.ndarray] = {}
    for task_index, task in enumerate(TASKS):
        for pair_id in range(PAIR_COUNT):
            vectors = _orthonormal_vectors(
                _pair_seed(world_seed, task_index, pair_id), len(SYMBOLS)
            )
            for symbol, vector in zip(SYMBOLS, vectors, strict=True):
                mapping[f"{task}/{pair_id:03d}/{symbol}"] = vector
    return LookupVectorEncoder(mapping)


def build_case(task: str, pair_id: int) -> HiddenPriorCase:
    if task not in TASKS:
        raise ValueError(task)
    if not 0 <= pair_id < PAIR_COUNT:
        raise ValueError(pair_id)
    key = lambda symbol: f"{task}/{pair_id:03d}/{symbol}"
    p, q, r = key("P"), key("Q"), key("R")
    if task == "norm_matched_repeat":
        # Same {P,P,Q,R} multiset. Under orthonormal inputs and EMA decay .8,
        # these two orderings have exactly equal slow-state norm.
        hidden0 = (p, q, r, p)  # repeated P at positions 1 and 4
        hidden1 = (q, p, p, r)  # repeated P at positions 2 and 3
    else:
        hidden0 = (p, q, p, q)
        hidden1 = (p, p, q, q)
    return HiddenPriorCase(
        task=task,
        pair_id=pair_id,
        hidden0=hidden0,
        hidden1=hidden1,
        visible=(key("A"), key("B"), key("C"), key("D")),
        delay_event=key("N"),
        final_event=key("Z"),
    )


def pairwise_relational(keys: tuple[str, ...], encoder: LookupVectorEncoder) -> np.ndarray:
    vectors = np.stack([encoder.encode(key) for key in keys])
    values: list[float] = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            values.append(float(np.dot(vectors[i], vectors[j])))
    return np.asarray(values, dtype=np.float32)


def ema_state(keys: tuple[str, ...], encoder: LookupVectorEncoder, decay: float = 0.80) -> np.ndarray:
    state = np.zeros(INPUT_DIM, dtype=np.float32)
    d = np.float32(decay)
    for key in keys:
        vector = encoder.encode(key)
        state = (d * state + (1.0 - d) * vector).astype(np.float32, copy=False)
    return state


def hidden_multiset(case: HiddenPriorCase, label: int) -> Counter[str]:
    return Counter(case.hidden0 if label == 0 else case.hidden1)
