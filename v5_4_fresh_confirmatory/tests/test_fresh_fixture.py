from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys

HERE = Path(__file__).resolve()
VERSION_ROOT = HERE.parents[1]
REPO_ROOT = HERE.parents[2]
V51_ROOT = REPO_ROOT / "v5_1_semantic_context"
sys.path.insert(0, str(VERSION_ROOT))
sys.path.insert(0, str(V51_ROOT))

from fresh_fixture import DOMAINS, build_fresh_pairs, flatten_pairs  # noqa: E402
from semantic_fixture import DOMAINS as OLD_DOMAINS  # noqa: E402


def test_fresh_domains_are_disjoint_from_v51_domains() -> None:
    assert set(DOMAINS).isdisjoint(set(OLD_DOMAINS))
    assert set(DOMAINS) == {
        "connectivity",
        "capacity",
        "integrity",
        "route",
        "assignment",
    }


def test_fresh_fixture_size_and_balance() -> None:
    train, test = build_fresh_pairs()
    assert len(train) == 60
    assert len(test) == 20
    for pairs in (train, test):
        arms = flatten_pairs(pairs)
        counts = Counter(arm.label for arm in arms)
        assert counts == Counter({0: len(pairs), 1: len(pairs)})


def test_pair_controls_hide_label_from_current_and_shared_events() -> None:
    train, test = build_fresh_pairs()
    for pair in train + test:
        pos = pair.positive
        neg = pair.negative
        assert pos.current_text == neg.current_text
        assert pos.history[0] == neg.history[0]
        assert pos.history[2:] == neg.history[2:]
        assert pos.history[1] != neg.history[1]
        assert pos.label == 1
        assert neg.label == 0


def test_train_and_test_semantic_templates_are_disjoint() -> None:
    train, test = build_fresh_pairs()
    train_semantic = {arm.history[1] for arm in flatten_pairs(train)}
    test_semantic = {arm.history[1] for arm in flatten_pairs(test)}
    assert train_semantic.isdisjoint(test_semantic)


def test_no_semantic_sentence_reused_from_v51_fixture() -> None:
    old_sentences: set[str] = set()
    for spec in OLD_DOMAINS.values():
        for key, values in spec.items():
            if key.startswith("train_") or key.startswith("test_"):
                old_sentences.update(values)

    train, test = build_fresh_pairs()
    new_semantic = {arm.history[1] for arm in flatten_pairs(train + test)}
    assert new_semantic.isdisjoint(old_sentences)
