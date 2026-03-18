from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from .branching import BranchPath
from .config import AppConfig
from .utils import pad_or_truncate


class GlobalHistoryEncoder(nn.Module):
    def __init__(self, config: AppConfig) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=config.latent.history_dim,
            hidden_size=config.latent.global_history_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )
        self.proj = nn.Linear(config.latent.global_history_hidden * 2, config.latent.path_model_dim)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        if history.dim() == 2:
            history = history.unsqueeze(0)
        out, hidden = self.gru(history)
        last = out[:, -1, :]
        return self.proj(last)


class DominantPathEncoder(nn.Module):
    def __init__(self, config: AppConfig) -> None:
        super().__init__()
        d_model = config.latent.path_model_dim
        self.neuron_emb = nn.Embedding(config.dynamics.num_neurons + 1, d_model // 4)
        self.cluster_emb = nn.Embedding(config.latent.max_cluster_embeddings, d_model // 4)
        self.pos_emb = nn.Embedding(config.branch.l_max + 2, d_model // 4)
        self.scalar_proj = nn.Linear(2, d_model // 4)
        self.in_proj = nn.Linear(d_model, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=config.latent.path_transformer_heads,
            dim_feedforward=d_model * 2,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.latent.path_transformer_layers)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.out_proj = nn.Linear(d_model, d_model)

    def _branch_to_tokens(self, branch: Optional[BranchPath], device: torch.device) -> torch.Tensor:
        if branch is None or not branch.neuron_path:
            return torch.zeros(1, 1, self.out_proj.in_features, device=device)
        neurons = torch.tensor(branch.neuron_path, dtype=torch.long, device=device)
        clusters = torch.tensor(branch.cluster_path, dtype=torch.long, device=device).clamp(min=0, max=self.cluster_emb.num_embeddings - 1)
        positions = torch.arange(len(branch.neuron_path), dtype=torch.long, device=device).clamp(max=self.pos_emb.num_embeddings - 1)
        edge_vals = pad_or_truncate(branch.edge_weights + [0.0], len(branch.neuron_path), 0.0)
        step_vals = pad_or_truncate([float(s) for s in branch.step_ids], len(branch.neuron_path), 0.0)
        scalars = torch.tensor(list(zip(edge_vals, step_vals)), dtype=torch.float32, device=device)
        tok = torch.cat(
            [
                self.neuron_emb(neurons),
                self.cluster_emb(clusters),
                self.pos_emb(positions),
                self.scalar_proj(scalars),
            ],
            dim=-1,
        )
        return self.in_proj(tok).unsqueeze(0)

    def forward(self, branch: Optional[BranchPath], device: torch.device) -> torch.Tensor:
        tokens = self._branch_to_tokens(branch, device)
        cls = self.cls.expand(tokens.shape[0], -1, -1).to(device)
        x = torch.cat([cls, tokens], dim=1)
        out = self.transformer(x)
        return self.out_proj(out[:, 0, :])


class HistoryEncoder(nn.Module):
    def __init__(self, config: AppConfig, latent_dim: int | None = None) -> None:
        super().__init__()
        self.latent_dim = latent_dim or config.latent.default_latent_dim
        self.global_encoder = GlobalHistoryEncoder(config)
        self.path_encoder = DominantPathEncoder(config)
        fusion_dim = config.latent.path_model_dim * 2
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.GELU(),
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, self.latent_dim),
        )

    def forward(self, history: torch.Tensor, branch: Optional[BranchPath]) -> torch.Tensor:
        device = history.device
        z_hist = self.global_encoder(history)
        z_path = self.path_encoder(branch, device)
        z = torch.cat([z_hist, z_path], dim=-1)
        return self.fusion(z)
