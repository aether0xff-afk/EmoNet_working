from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import torch
from torch import nn

from .config import AppConfig
from .losses import control_loss, style_consistency_loss, summarize_losses, tone_loss
from .model import EmotionArchitecture
from .style_scorer import StyleScorer


@dataclass
class Batch:
    texts: Sequence[str]
    h_target: torch.Tensor | None = None
    s_target: torch.Tensor | None = None


class EmotionTrainer:
    def __init__(self, model: EmotionArchitecture, config: AppConfig) -> None:
        self.model = model
        self.config = config
        self.style_scorer = StyleScorer(config)
        params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(params, lr=config.training.lr, weight_decay=config.training.weight_decay)

    def train_step(self, batch: Batch) -> Dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad()
        # Training is performed one example at a time because branch tracing is episode-centric.
        losses_accum = []
        for idx, text in enumerate(batch.texts):
            out = self.model._run_episode(text)
            losses: Dict[str, torch.Tensor] = {}
            if batch.h_target is not None:
                losses["control"] = control_loss(out.h_t.unsqueeze(0), batch.h_target[idx : idx + 1])
            if batch.s_target is not None:
                losses["tone"] = tone_loss(out.s.unsqueeze(0), batch.s_target[idx : idx + 1])
            scorer_pred = self.style_scorer([text])
            losses["style_consistency"] = style_consistency_loss(out.s.unsqueeze(0), scorer_pred.detach())
            total = summarize_losses(losses, {"control": 0.15, "tone": 0.12, "style_consistency": 0.10})
            total.backward()
            losses_accum.append(total.detach())
        nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.grad_clip)
        self.optimizer.step()
        return {"loss": float(torch.stack(losses_accum).mean().item()) if losses_accum else 0.0}
