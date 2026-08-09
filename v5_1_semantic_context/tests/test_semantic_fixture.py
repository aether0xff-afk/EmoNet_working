from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys

VERSION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VERSION_ROOT))

from semantic_fixture import build_semantic_pairs, flatten_pairs  # noqa: E402


def test_version_fixture_sizes_and_balance() -> None:
    train, test = build_semantic_pairs()
    assert len(train) == 60
    assert len(test) == 20

    for pairs in (train, test):
        arms = flatten_pairs(pairs)
        counts = Counter(arm.label for arm in arms)
        assert counts[0] == counts[1]
        assert set(counts) == {0, 1}


def test_pair_controls_hide_label_from_current_and_last_event() -> None:
    train, test = build_semantic_pairs()
    for pair in train + test:
        a = pair.arm_usable
        b = pair.arm_blocked
        assert a.current_text == b.current_text
        assert a.history[-1] == b.history[-1]
        assert a.history[0] == b.history[0]
        assert a.history[2:] == b.history[2:]
        assert a.history[1] != b.history[1]
        assert a.label == 1
        assert b.label == 0


def test_held_out_semantic_statements_are_not_train_templates() -> None:
    train, test = build_semantic_pairs()
    train_semantic = {arm.history[1] for arm in flatten_pairs(train)}
    test_semantic = {arm.history[1] for arm in flatten_pairs(test)}
    assert train_semantic.isdisjoint(test_semantic)


def test_pair_ids_do_not_cross_split() -> None:
    train, test = build_semantic_pairs()
    train_ids = {pair.pair_id for pair in train}
    test_ids = {pair.pair_id for pair in test}
    assert train_ids.isdisjoint(test_ids)
