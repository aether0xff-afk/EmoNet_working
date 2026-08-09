from __future__ import annotations

import inspect
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve()
VERSION_ROOT = HERE.parents[1]
REPO_ROOT = HERE.parents[2]
V52_EXPERIMENTS = REPO_ROOT / "v5_2_learned_memory" / "experiments"
sys.path.insert(0, str(VERSION_ROOT))
sys.path.insert(0, str(VERSION_ROOT / "experiments"))
sys.path.insert(0, str(V52_EXPERIMENTS))

from predictive_fixture import build_predictive_pairs, flatten_pairs  # noqa: E402
from run_predictive_state_benchmark import train_predictive_core  # noqa: E402
from run_learned_memory_benchmark import sequence_tensor  # noqa: E402


class RecordingEncoder:
    def __init__(self, dimension: int = 8) -> None:
        self.dimension = dimension
        self.seen: list[str] = []

    @property
    def output_dim(self) -> int:
        return self.dimension

    def encode(self, text: str) -> np.ndarray:
        self.seen.append(str(text))
        vector = np.zeros(self.dimension, dtype=np.float32)
        vector[sum(ord(ch) for ch in str(text)) % self.dimension] = 1.0
        return vector


def test_pair_controls_and_future_are_distinct() -> None:
    train, test = build_predictive_pairs()
    assert len(train) == 60
    assert len(test) == 20
    for pair in train + test:
        pos = pair.positive
        neg = pair.negative
        assert pos.current_text == neg.current_text
        assert pos.history[0] == neg.history[0]
        assert pos.history[2:] == neg.history[2:]
        assert pos.history[1] != neg.history[1]
        assert pos.future_text != neg.future_text
        assert pos.label == 1
        assert neg.label == 0


def test_future_paraphrases_are_disjoint_between_train_and_test() -> None:
    train, test = build_predictive_pairs()
    train_future = {arm.future_text for arm in flatten_pairs(train)}
    test_future = {arm.future_text for arm in flatten_pairs(test)}
    assert train_future.isdisjoint(test_future)


def test_evaluated_sequence_excludes_future_consequence() -> None:
    train, _ = build_predictive_pairs()
    arms = flatten_pairs(train[:2])
    encoder = RecordingEncoder()
    tensor = sequence_tensor(arms, encoder)

    expected_events = len(arms[0].history) + 1
    assert tensor.shape == (len(arms), expected_events, encoder.output_dim)

    future_texts = {arm.future_text for arm in arms}
    assert future_texts.isdisjoint(set(encoder.seen))
    for arm in arms:
        assert arm.current_text in encoder.seen
        for event in arm.history:
            assert event in encoder.seen


def test_training_api_accepts_no_state_or_emotion_labels() -> None:
    signature = inspect.signature(train_predictive_core)
    names = set(signature.parameters)
    assert names == {
        "seed",
        "train_sequences",
        "future_target_ids",
        "future_vocab_embeddings",
    }
    forbidden = {
        "label",
        "labels",
        "positive",
        "negative",
        "usable",
        "blocked",
        "emotion",
        "valence",
        "arousal",
        "domain",
    }
    assert names.isdisjoint(forbidden)
