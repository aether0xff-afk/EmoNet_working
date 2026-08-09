from __future__ import annotations

import inspect
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve()
VERSION_ROOT = HERE.parents[1]
REPO_ROOT = HERE.parents[2]
V51_ROOT = REPO_ROOT / "v5_1_semantic_context"
V52_ROOT = REPO_ROOT / "v5_2_learned_memory"
V52_EXPERIMENTS = V52_ROOT / "experiments"
sys.path.insert(0, str(VERSION_ROOT / "experiments"))
sys.path.insert(0, str(V51_ROOT))
sys.path.insert(0, str(V52_ROOT))
sys.path.insert(0, str(V52_EXPERIMENTS))

from learned_core import LearnedCoreConfig, LearnedLeakyRecurrentCore  # noqa: E402
from run_contrastive_memory_benchmark import (  # noqa: E402
    build_event_vocabulary,
    contrastive_delayed_loss,
    train_contrastive_core,
)
from semantic_fixture import build_semantic_pairs, flatten_pairs  # noqa: E402


class TinyEncoder:
    def __init__(self, dimension: int = 8) -> None:
        self.dimension = dimension

    @property
    def output_dim(self) -> int:
        return self.dimension

    def encode(self, text: str) -> np.ndarray:
        seed = sum((i + 1) * ord(ch) for i, ch in enumerate(text)) % (2**32)
        rng = np.random.default_rng(seed)
        vector = rng.normal(size=self.dimension).astype(np.float32)
        vector /= max(float(np.linalg.norm(vector)), 1e-8)
        return vector


def test_event_vocabulary_deduplicates_shared_text() -> None:
    train, _ = build_semantic_pairs()
    arms = flatten_pairs(train[:3])
    encoder = TinyEncoder()
    vocab, vectors, ids = build_event_vocabulary(arms, encoder)

    all_events = [text for arm in arms for text in (*arm.history, arm.current_text)]
    assert len(vocab) == len(set(all_events))
    assert vectors.shape == (len(vocab), encoder.output_dim)
    assert ids.shape == (len(arms), len(arms[0].history) + 1)


def test_contrastive_loss_reaches_recurrent_weights() -> None:
    torch.manual_seed(7)
    config = LearnedCoreConfig(
        hidden_dim=12,
        event_ticks=5,
        stimulation_ticks=2,
        max_lag=3,
    )
    model = LearnedLeakyRecurrentCore(input_dim=8, config=config, seed=7)
    embeddings = torch.randn(6, 5, 8)
    embeddings = torch.nn.functional.normalize(embeddings, dim=-1)

    # Every sequence position has a unique self-supervised event identity.
    vocab = embeddings.reshape(-1, 8).clone()
    vocab = torch.nn.functional.normalize(vocab, dim=-1)
    event_ids = torch.arange(30, dtype=torch.long).reshape(6, 5)

    loss, lag_losses = contrastive_delayed_loss(model, embeddings, event_ids, vocab)
    loss.backward()

    assert set(lag_losses) == {1, 2, 3}
    assert torch.isfinite(loss)
    assert model.recurrent_weight.grad is not None
    assert float(model.recurrent_weight.grad.abs().sum()) > 0.0
    assert model.input_weight.grad is not None
    assert float(model.input_weight.grad.abs().sum()) > 0.0


def test_training_api_contains_no_semantic_or_emotion_label_argument() -> None:
    signature = inspect.signature(train_contrastive_core)
    names = set(signature.parameters)
    forbidden = {
        "label",
        "labels",
        "emotion",
        "emotions",
        "valence",
        "arousal",
        "usable",
        "blocked",
        "target_class",
    }
    assert names.isdisjoint(forbidden)
    assert names == {
        "seed",
        "train_sequences",
        "train_event_ids",
        "train_vocab_embeddings",
    }


def test_event_ids_encode_identity_not_task_label() -> None:
    train, _ = build_semantic_pairs()
    arms = flatten_pairs(train[:5])
    encoder = TinyEncoder()
    vocab, _, event_ids = build_event_vocabulary(arms, encoder)

    # Each id must round-trip to the literal event text at that sequence position.
    for row_index, arm in enumerate(arms):
        expected = (*arm.history, arm.current_text)
        observed = tuple(vocab[int(idx)] for idx in event_ids[row_index])
        assert observed == expected
