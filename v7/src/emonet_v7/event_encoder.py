"""Trainable projection from text events to SNN input currents."""

from __future__ import annotations

import math
from collections.abc import Sequence
from hashlib import sha256

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
        num_speakers: int = 256,
        hidden_dim: int = 256,
        current_scale: float = 0.75,
    ) -> None:
        super().__init__()
        self.current_scale = float(current_scale)
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

    def speaker_id(self, speaker: str) -> int:
        """Map a speaker name to a stable embedding slot.

        A deterministic hash keeps speaker identities reproducible across
        sessions and independent of event processing order. Hash collisions are
        possible but rare enough for the small module counts used in v7.0.
        """

        digest = sha256(speaker.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], byteorder="big") % self.speaker_embedding.num_embeddings

    def forward(self, text_embeddings: torch.Tensor, events: Sequence[Event]) -> torch.Tensor:
        if text_embeddings.ndim != 2 or text_embeddings.shape[0] != len(events):
            raise ValueError("text_embeddings must have shape [len(events), embedding_dim]")
        device = text_embeddings.device
        kind_ids = torch.tensor([EVENT_KIND_TO_ID[event.kind] for event in events], device=device)
        speaker_ids = torch.tensor([self.speaker_id(event.speaker_id) for event in events], device=device)
        elapsed_features = torch.tensor(
            [[math.log1p(max(0.0, event.elapsed_seconds)), 1.0 if event.elapsed_seconds > 0 else 0.0] for event in events],
            dtype=text_embeddings.dtype,
            device=device,
        )
        features = torch.cat([text_embeddings, self.kind_embedding(kind_ids), self.speaker_embedding(speaker_ids), elapsed_features], dim=-1)
        return self.net(features) * self.current_scale
