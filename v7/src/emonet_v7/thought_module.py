"""Minimal LLM thought module for Milestone 3 feedback experiments."""

from __future__ import annotations

import json
from typing import Protocol


class ChatClient(Protocol):
    def chat(self, messages: list[dict[str, str]], *, temperature: float = 0.7) -> str:
        """Return a local model response."""


class ThoughtModule:
    """Generate one short internal thought from a user event and neutral state."""

    def __init__(self, client: ChatClient, *, module_id: str = "module_0") -> None:
        self.client = client
        self.module_id = module_id

    def build_messages(
        self,
        *,
        user_text: str,
        state_report: dict,
        condition_instruction: str | None = None,
    ) -> list[dict[str, str]]:
        system = (
            "너는 EmoNet 내부 사고 모듈이다. 사용자에게 직접 답하지 말고, "
            "현재 사건을 해석하는 짧은 내부 생각 하나만 작성하라. "
            "감정 라벨을 단정하지 말고, 새로운 근거가 없으면 과도하게 확신하지 말라. "
            "한 문장으로만 답하라."
        )
        if condition_instruction:
            system = f"{system} 추가 실험 조건: {condition_instruction}"
        user = (
            f"<user_event>\n{user_text}\n</user_event>\n\n"
            "<neutral_internal_state>\n"
            f"{json.dumps(state_report, ensure_ascii=False, indent=2)}\n"
            "</neutral_internal_state>"
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def generate_internal_thought(
        self,
        *,
        user_text: str,
        state_report: dict,
        condition_instruction: str | None = None,
        temperature: float = 0.7,
    ) -> str:
        messages = self.build_messages(
            user_text=user_text,
            state_report=state_report,
            condition_instruction=condition_instruction,
        )
        thought = self.client.chat(messages, temperature=temperature)
        cleaned = " ".join(thought.strip().splitlines()).strip()
        if not cleaned:
            raise RuntimeError("thought module returned an empty thought")
        return cleaned
