from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn

from .config import AppConfig
from .encoders import FrozenTextRegressor


class StyleScorer(nn.Module):
    def __init__(self, config: AppConfig) -> None:
        super().__init__()
        self.regressor = FrozenTextRegressor(config, out_dim=config.style.num_styles)

    def forward(self, texts: Sequence[str]) -> torch.Tensor:
        return self.regressor(texts)


@dataclass
class StyleLabelRecord:
    text: str
    pseudo_label: torch.Tensor
    corrected_label: torch.Tensor | None = None


class StyleLabelPipeline:
    """Offline helper. LLM pseudo-labeling must be plugged in by the user."""

    def __init__(self) -> None:
        self.records: list[StyleLabelRecord] = []

    def add_pseudo_label(self, text: str, label: torch.Tensor) -> None:
        self.records.append(StyleLabelRecord(text=text, pseudo_label=label.detach().cpu()))

    def add_human_correction(self, index: int, corrected_label: torch.Tensor) -> None:
        self.records[index].corrected_label = corrected_label.detach().cpu()

    def export_training_pairs(self) -> list[tuple[str, torch.Tensor]]:
        out = []
        for rec in self.records:
            out.append((rec.text, rec.corrected_label if rec.corrected_label is not None else rec.pseudo_label))
        return out
