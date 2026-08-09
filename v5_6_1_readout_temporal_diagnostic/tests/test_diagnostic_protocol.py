from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve()
VERSION_ROOT = HERE.parents[1]
sys.path.insert(0, str(VERSION_ROOT / "experiments"))

from run_readout_temporal_diagnostic import (  # noqa: E402
    PROJECTION_DIMS,
    deterministic_projection,
    structural_pair,
    token_name,
)


def test_projection_is_deterministic_and_label_free() -> None:
    for dim in PROJECTION_DIMS:
        a = deterministic_projection(128, dim, "fast")
        b = deterministic_projection(128, dim, "fast")
        slow = deterministic_projection(128, dim, "slow")
        assert np.array_equal(a, b)
        assert a.shape == (128, dim)
        assert not np.array_equal(a, slow)


def test_structural_pair_has_same_multiset_and_current() -> None:
    for pair_id in (0, 1, 17, 80, 119):
        class0, class1, current, event_a, event_b = structural_pair(pair_id)
        assert class0[0] == class1[0]
        assert class0[-1] == class1[-1]
        assert current == "The identical current observation is now presented."
        assert sorted(class0[1:5]) == sorted(class1[1:5])
        assert class0[1:5].count(event_a) == 2
        assert class0[1:5].count(event_b) == 2
        assert class1[1:5].count(event_a) == 2
        assert class1[1:5].count(event_b) == 2
        assert class0[1:5] != class1[1:5]


def test_pair_token_identities_are_disjoint_across_pairs() -> None:
    names: set[str] = set()
    for pair_id in range(120):
        for side in (0, 1):
            name = token_name(pair_id, side)
            assert name not in names
            names.add(name)
    assert len(names) == 240


def test_train_and_test_token_identities_are_disjoint() -> None:
    train_names = {
        token_name(pair_id, side)
        for pair_id in range(80)
        for side in (0, 1)
    }
    test_names = {
        token_name(pair_id, side)
        for pair_id in range(80, 120)
        for side in (0, 1)
    }
    assert train_names.isdisjoint(test_names)
