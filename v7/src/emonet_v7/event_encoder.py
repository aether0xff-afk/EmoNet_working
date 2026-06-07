"""Trainable projection from text events to SNN input currents."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn

from .schemas import EVENT_KIND_TO_ID, Event


class EventEncoder(nn.Module):
    """Convert frozen text embeddings and structural metadata into currents."""

    def __init__(
        self,
        *,
        text_embedding_dim: int,
        num_neurons: int,
        event_kind_embedding_dim: int = 8,
        speaker_embedding_dim: int = 8,
        num_speakers: int = 16,
        hidden_dim: int = 256,
        current_scale: float = 0.75,
    ) -> None:
        super().__init__()
        self.current_scale = float(current_scale)
        self.speaker_to_id: dict[str, int] = {}
        self.kind_embedding = nn.Embedding(len(EVENT_KIND_TO_ID), event_kind_embedding_dim)
        self.speaker_embedding = nn.Embedding(num_speakers, speaker_embedding_dim)
        input_dim = text_embedding_dim + event_kind_embedding_dim + speaker_embedding_dim + 2
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_neurons),
            nn.Tanh(),
        )

    def _speaker_id(self, speaker: str) -> int:
        if speaker not in self.speaker_to_id:
            next_id = len(self.speaker_to_id)
            if next_id >= self.speaker_embedding.num_embeddings:
                raise ValueError("speaker vocabulary capacity exceeded")
            self.speaker_to_id[speaker] = next_id
        return self.speaker_to_id[speaker]

    def forward(self, text_embeddings: torch.Tensor, events: Sequence[Event]) -> torch.Tensor:
        if text_embeddings.ndim != 2 or text_embeddings.shape[0] != len(events):
            raise ValueError("text_embeddings must have shape [len(events), embedding_dim]")
        device = text_embeddings.device
        kind_ids = torch.tensor([EVENT_KIND_TO_ID[event.kind] for event in events], device=device)
        speaker_ids = torch.tensor([self._speaker_id(event.speaker_id) for event in events], device=device)
        elapsed_features = torch.tensor(
            [[math.log1p(max(0.0, event.elapsed_seconds)), 1.0 if event.elapsed_seconds > 0 else 0.0] for event in events],
            dtype=text_embeddings.dtype,
            device=device,
        )
        features = torch.cat([text_embeddings, self.kind_embedding(kind_ids), self.speaker_embedding(speaker_ids), elapsed_features], dim=-1)
        return self.net(features) * self.current_scale
