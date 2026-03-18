from __future__ import annotations

from typing import Dict, List

import torch

from .config import AppConfig, STYLE_NAMES


class PromptGenerator:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.idx = {name: i for i, name in enumerate(STYLE_NAMES)}

    def _value(self, s: torch.Tensor, name: str) -> float:
        return float(s[self.idx[name]].item())

    def sanitize_style(self, s: torch.Tensor) -> torch.Tensor:
        s = s.clone()
        s[self.idx["hostility"]] = min(float(s[self.idx["hostility"]].item()), self.config.style.safety_hostility_clip)
        s[self.idx["confrontationality"]] = min(float(s[self.idx["confrontationality"]].item()), self.config.style.safety_confront_clip)
        return s

    def generate_constraints(self, style_vec: torch.Tensor) -> Dict[str, List[str] | str | Dict[str, float]]:
        s = self.sanitize_style(style_vec.flatten())
        instructions: List[str] = []
        if self._value(s, "warmth") > 0.4:
            instructions.append("따뜻하고 배려 있게 말하기")
        if self._value(s, "directness") > 0.4:
            instructions.append("돌려 말하지 말고 분명하게 말하기")
        if self._value(s, "formality") > 0.3:
            instructions.append("약간 격식 있는 표현 사용")
        if self._value(s, "verbosity") < -0.3:
            instructions.append("짧고 핵심 위주로 답하기")
        if self._value(s, "certainty") > 0.4:
            instructions.append("너무 망설이지 말고 비교적 단정적으로 답하기")
        if self._value(s, "hedging") > 0.4:
            instructions.append("완곡 표현을 적절히 사용하기")
        if self._value(s, "reassurance") > 0.4:
            instructions.append("상대가 안심할 수 있게 표현하기")
        if self._value(s, "emphasis_intensity") > 0.5:
            instructions.append("중요한 부분은 분명하게 강조하기")
        prompt = " / ".join(instructions) if instructions else "자연스럽고 균형 있게 답하기"
        return {
            "prompt": prompt,
            "instructions": instructions,
            "style": {name: float(s[i].item()) for i, name in enumerate(STYLE_NAMES)},
        }
