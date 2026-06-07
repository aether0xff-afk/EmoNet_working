"""Frozen text encoders for semantic events."""

from __future__ import annotations

from hashlib import sha256
from typing import Protocol, Sequence

import torch


class TextEncoder(Protocol):
    """Minimal encoder interface used by the SNN input pipeline."""

    output_dim: int

    def encode(self, texts: Sequence[str]) -> torch.Tensor:
        """Return normalized embeddings with shape [batch, output_dim]."""


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
