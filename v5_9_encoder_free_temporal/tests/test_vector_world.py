from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
VERSION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(VERSION_ROOT))

from vector_world import (
    PAIR_COUNT,
    TASKS,
    TRAIN_PAIRS,
    WORLD_SEEDS,
    build_case,
    build_vector_world,
)


def test_pair_event_vectors_are_unit_and_orthogonal() -> None:
    encoder = build_vector_world(WORLD_SEEDS[0])
    for task in TASKS:
        case = build_case(task, 83)
        keys = sorted(set(case.class0[1:5]) | set(case.class1[1:5]))
        vectors = np.stack([encoder.encode(key) for key in keys])
        gram = vectors @ vectors.T
        np.testing.assert_allclose(gram, np.eye(len(keys)), atol=1e-5)


def test_competing_classes_have_same_event_multiset() -> None:
    for task in TASKS:
        for pair_id in (0, TRAIN_PAIRS - 1, TRAIN_PAIRS, PAIR_COUNT - 1):
            case = build_case(task, pair_id)
            assert Counter(case.class0[1:5]) == Counter(case.class1[1:5])
            assert case.class0[0] == case.class1[0] == "neutral/prefix"
            assert case.class0[-1] == case.class1[-1] == "neutral/suffix"
            assert case.current == "neutral/current"


def test_train_test_event_keys_are_disjoint() -> None:
    train: set[str] = set()
    test: set[str] = set()
    for task in TASKS:
        for pair_id in range(PAIR_COUNT):
            case = build_case(task, pair_id)
            keys = set(case.class0[1:5]) | set(case.class1[1:5])
            (train if pair_id < TRAIN_PAIRS else test).update(keys)
    assert train.isdisjoint(test)


def test_vector_worlds_are_deterministic_but_distinct() -> None:
    a1 = build_vector_world(WORLD_SEEDS[0])
    a2 = build_vector_world(WORLD_SEEDS[0])
    b = build_vector_world(WORLD_SEEDS[1])
    key = "alternation/083/A"
    np.testing.assert_allclose(a1.encode(key), a2.encode(key), atol=0.0)
    assert not np.allclose(a1.encode(key), b.encode(key))


def test_event_key_text_does_not_define_vector_semantics() -> None:
    encoder = build_vector_world(WORLD_SEEDS[0])
    # The lookup keys are only addresses. Their strings never enter a language model.
    vector = encoder.encode("repeat_gap/100/B")
    assert vector.shape == (384,)
    assert abs(float(np.linalg.norm(vector)) - 1.0) < 1e-5
