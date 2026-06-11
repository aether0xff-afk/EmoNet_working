"""Frozen text encoders for semantic events."""

from __future__ import annotations

from hashlib import sha256
from typing import Protocol, Sequence

import torch
from torch.nn import functional as F


class TextEncoder(Protocol):
    """Minimal encoder interface used by the SNN input pipeline."""

    output_dim: int

    def encode(self, texts: Sequence[str]) -> torch.Tensor:
        """Return normalized embeddings with shape [batch, output_dim]."""


class EmbeddingClient(Protocol):
    """Minimal LM Studio-compatible embedding client interface."""

    def embed(self, texts: list[str], *, model: str | None = None) -> list[list[float]]:
        """Return embedding rows from a local API."""


class DeterministicHashTextEncoder:
    """Offline-only smoke-test encoder. It is deterministic but not semantic."""

    def __init__(self, output_dim: int = 384) -> None:
        self.output_dim = output_dim

    def encode(self, texts: Sequence[str]) -> torch.Tensor:
        rows: list[torch.Tensor] = []
        for text in texts:
            values: list[float] = []
            counter = 0
            while len(values) < self.output_dim:
                digest = sha256(f"{counter}:{text}".encode("utf-8")).digest()
                values.extend((byte / 127.5) - 1.0 for byte in digest)
                counter += 1
            row = torch.tensor(values[: self.output_dim], dtype=torch.float32)
            rows.append(row / row.norm().clamp_min(1e-8))
        return torch.stack(rows)


class SentenceTransformerTextEncoder:
    """Frozen multilingual sentence-transformers adapter."""

    def __init__(self, model_name: str, device: str = "cpu") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("Install the optional 'text' dependency") from exc
        self.model = SentenceTransformer(model_name, device=device)
        if hasattr(self.model, "get_embedding_dimension"):
            self.output_dim = int(self.model.get_embedding_dimension())
        else:
            self.output_dim = int(self.model.get_sentence_embedding_dimension())

    def encode(self, texts: Sequence[str]) -> torch.Tensor:
        return (
            self.model.encode(
                list(texts),
                convert_to_tensor=True,
                normalize_embeddings=True,
            )
            .detach()
            .cpu()
            .to(torch.float32)
        )


class LMStudioEmbeddingTextEncoder:
    """Frozen text encoder backed by LM Studio's embeddings endpoint."""

    def __init__(self, client: EmbeddingClient, model_name: str) -> None:
        self.client = client
        self.model_name = model_name
        probe = self.encode(["dimension probe"])
        if probe.ndim != 2 or probe.shape[0] != 1:
            raise RuntimeError("LM Studio embedding probe returned an invalid shape")
        self.output_dim = int(probe.shape[-1])

    def encode(self, texts: Sequence[str]) -> torch.Tensor:
        rows = self.client.embed(list(texts), model=self.model_name)
        tensor = torch.tensor(rows, dtype=torch.float32)
        if tensor.ndim != 2:
            raise RuntimeError("LM Studio embeddings must have shape [batch, embedding_dim]")
        return F.normalize(tensor, dim=-1)
