from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from .config import AppConfig
from .utils import hash_tokens, tokenize


class HashingTextEncoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, max_tokens: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.max_tokens = max_tokens
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, mode="mean")
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, texts: Sequence[str]) -> torch.Tensor:
        vectors = []
        device = self.embedding.weight.device
        for text in texts:
            ids = hash_tokens(tokenize(text, self.max_tokens), self.vocab_size).to(device)
            offsets = torch.tensor([0], dtype=torch.long, device=device)
            emb = self.embedding(ids, offsets)
            vectors.append(self.proj(emb))
        return torch.cat(vectors, dim=0)


class ControlEncoder(nn.Module):
    def __init__(self, config: AppConfig) -> None:
        super().__init__()
        self.config = config
        self.text_encoder = HashingTextEncoder(
            vocab_size=config.text.vocab_size,
            embed_dim=config.text.embed_dim,
            max_tokens=config.text.max_tokens,
        )
        input_dim = config.text.embed_dim + config.trait_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, config.text.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.text.dropout),
            nn.LayerNorm(config.text.hidden_dim),
            nn.Linear(config.text.hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, config.control_dim),
            nn.Tanh(),
        )

    def forward(self, texts: Sequence[str], trait_batch: torch.Tensor) -> torch.Tensor:
        text_features = self.text_encoder(texts)
        if trait_batch.dim() == 1:
            trait_batch = trait_batch.unsqueeze(0)
        feat = torch.cat([text_features, trait_batch.to(text_features.device)], dim=-1)
        return self.mlp(feat)


class FrozenTextRegressor(nn.Module):
    def __init__(self, config: AppConfig, out_dim: int) -> None:
        super().__init__()
        self.encoder = HashingTextEncoder(
            vocab_size=config.text.vocab_size,
            embed_dim=config.text.embed_dim,
            max_tokens=config.text.max_tokens,
        )
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.head = nn.Sequential(
            nn.Linear(config.text.embed_dim, config.style.scorer_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.text.dropout),
            nn.Linear(config.style.scorer_hidden_dim, out_dim),
            nn.Tanh(),
        )

    def forward(self, texts: Sequence[str]) -> torch.Tensor:
        with torch.no_grad():
            enc = self.encoder(texts)
        return self.head(enc)
