from __future__ import annotations

from pathlib import Path

import torch

from emonet_v7.context_objective import (
    ContextFreeMLP,
    context_margin,
    context_ranking_loss,
    load_contrast_pairs,
    validate_contrast_pairs,
)
from emonet_v7.episode_dataset import load_episodes


def test_context_fixture_contains_train_and_validation_pairs() -> None:
    fixture = Path("fixtures/context_dependence_episodes.yaml")
    episodes = load_episodes(fixture)
    train_pairs = load_contrast_pairs(fixture, split="train")
    validation_pairs = load_contrast_pairs(fixture, split="validation")
    assert len(train_pairs) == 2
    assert len(validation_pairs) == 2
    validate_contrast_pairs(episodes, train_pairs + validation_pairs)


def test_context_ranking_loss_prefers_correct_targets() -> None:
    left_target = torch.tensor([[1.0, 0.0]])
    right_target = torch.tensor([[0.0, 1.0]])
    correct_loss = context_ranking_loss(
        left_prediction=left_target,
        left_target=left_target,
        right_prediction=right_target,
        right_target=right_target,
        margin=0.05,
    )
    swapped_loss = context_ranking_loss(
        left_prediction=right_target,
        left_target=left_target,
        right_prediction=left_target,
        right_target=right_target,
        margin=0.05,
    )
    assert float(correct_loss) == 0.0
    assert float(swapped_loss) > float(correct_loss)


def test_context_margin_is_positive_for_correct_pairing() -> None:
    left_target = torch.tensor([[1.0, 0.0]])
    right_target = torch.tensor([[0.0, 1.0]])
    value = context_margin(
        left_prediction=left_target,
        left_target=left_target,
        right_prediction=right_target,
        right_target=right_target,
    )
    assert float(value) > 0.0


def test_context_free_mlp_returns_same_prediction_for_same_current_text() -> None:
    torch.manual_seed(2)
    model = ContextFreeMLP(embedding_dim=4, hidden_dim=8)
    current = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
    assert torch.allclose(model(current), model(current))
