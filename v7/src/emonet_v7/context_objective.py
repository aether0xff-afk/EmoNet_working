"""Context-sensitive objectives and lightweight comparison baselines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch import nn
from torch.nn import functional as F
import yaml

from .episode_dataset import Episode


@dataclass(frozen=True)
class ContrastPair:
    """Two episodes with the same current event but different prior context."""

    split: str
    left_episode_id: str
    right_episode_id: str
    step_index: int
    relation: str


def load_contrast_pairs(path: str | Path, *, split: str | None = None) -> list[ContrastPair]:
    """Load contrast-pair metadata from a YAML episode fixture."""

    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    raw_pairs = data.get("contrast_pairs") if isinstance(data, dict) else None
    if not isinstance(raw_pairs, list) or not raw_pairs:
        raise ValueError("fixture must contain a non-empty contrast_pairs list")

    pairs: list[ContrastPair] = []
    for raw_pair in raw_pairs:
        if not isinstance(raw_pair, dict):
            raise ValueError("each contrast pair must be an object")
        pair = ContrastPair(
            split=str(raw_pair.get("split", "validation")),
            left_episode_id=str(raw_pair["left"]),
            right_episode_id=str(raw_pair["right"]),
            step_index=int(raw_pair["step_index"]),
            relation=str(raw_pair.get("relation", "context_contrast")),
        )
        if pair.split not in {"train", "validation", "test"}:
            raise ValueError(f"unsupported contrast split: {pair.split}")
        pairs.append(pair)
    if split is None:
        return pairs
    return [pair for pair in pairs if pair.split == split]


def validate_contrast_pairs(episodes: Iterable[Episode], pairs: Iterable[ContrastPair]) -> None:
    """Validate that every pair isolates history while sharing current text."""

    episode_by_id = {episode.episode_id: episode for episode in episodes}
    for pair in pairs:
        try:
            left = episode_by_id[pair.left_episode_id]
            right = episode_by_id[pair.right_episode_id]
        except KeyError as exc:
            raise ValueError(f"contrast pair references unknown episode: {exc.args[0]}") from exc
        if left.split != pair.split or right.split != pair.split:
            raise ValueError(f"contrast pair split mismatch: {pair.relation}")
        if pair.step_index <= 0:
            raise ValueError("contrast step_index must be greater than zero so prior context exists")
        if pair.step_index + 1 >= len(left.events) or pair.step_index + 1 >= len(right.events):
            raise ValueError(f"contrast step_index out of range: {pair.relation}")
        if left.events[pair.step_index].text != right.events[pair.step_index].text:
            raise ValueError(f"current text must match inside contrast pair: {pair.relation}")
        if left.events[: pair.step_index] == right.events[: pair.step_index]:
            raise ValueError(f"prior context must differ inside contrast pair: {pair.relation}")
        if left.events[pair.step_index + 1].text == right.events[pair.step_index + 1].text:
            raise ValueError(f"target text must differ inside contrast pair: {pair.relation}")


def cosine_distance(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Return batch-mean cosine distance."""

    return (1.0 - F.cosine_similarity(left, right, dim=-1)).mean()


def context_margin(
    *,
    left_prediction: torch.Tensor,
    left_target: torch.Tensor,
    right_prediction: torch.Tensor,
    right_target: torch.Tensor,
) -> torch.Tensor:
    """Return mean positive-is-better separation between own and opposite targets."""

    left_own = F.cosine_similarity(left_prediction, left_target, dim=-1)
    left_other = F.cosine_similarity(left_prediction, right_target, dim=-1)
    right_own = F.cosine_similarity(right_prediction, right_target, dim=-1)
    right_other = F.cosine_similarity(right_prediction, left_target, dim=-1)
    return ((left_own - left_other) + (right_own - right_other)).mean() / 2.0


def context_ranking_loss(
    *,
    left_prediction: torch.Tensor,
    left_target: torch.Tensor,
    right_prediction: torch.Tensor,
    right_target: torch.Tensor,
    margin: float = 0.05,
) -> torch.Tensor:
    """Encourage each prediction to prefer its own contextual target."""

    left_own = F.cosine_similarity(left_prediction, left_target, dim=-1)
    left_other = F.cosine_similarity(left_prediction, right_target, dim=-1)
    right_own = F.cosine_similarity(right_prediction, right_target, dim=-1)
    right_other = F.cosine_similarity(right_prediction, left_target, dim=-1)
    left_loss = torch.relu(torch.as_tensor(margin, device=left_own.device) - left_own + left_other)
    right_loss = torch.relu(torch.as_tensor(margin, device=right_own.device) - right_own + right_other)
    return (left_loss + right_loss).mean() / 2.0


class ContextFreeMLP(nn.Module):
    """Predict the next event from the current text embedding only."""

    def __init__(self, *, embedding_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def encode_context(self, current_embedding: torch.Tensor) -> torch.Tensor:
        """Return the context-free representation used by this baseline."""

        return current_embedding

    def forward(self, current_embedding: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(self.encode_context(current_embedding)), dim=-1)


class GRUContextPredictor(nn.Module):
    """Standard recurrent baseline over ordered event embeddings."""

    def __init__(self, *, embedding_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.gru = nn.GRU(input_size=embedding_dim, hidden_size=hidden_dim, batch_first=True)
        self.projection = nn.Linear(hidden_dim, embedding_dim)

    def encode_context(self, sequence_embeddings: torch.Tensor) -> torch.Tensor:
        """Return the final recurrent hidden representation."""

        _, hidden = self.gru(sequence_embeddings)
        return hidden[-1]

    def forward(self, sequence_embeddings: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.projection(self.encode_context(sequence_embeddings)), dim=-1)
