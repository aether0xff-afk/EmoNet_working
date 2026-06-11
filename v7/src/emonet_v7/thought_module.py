"""Minimal LLM thought module for Milestone 3 feedback experiments."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from typing import Callable, Literal, Protocol


class ChatClient(Protocol):
    def chat(self, messages: list[dict[str, str]], *, temperature: float = 0.7) -> str:
        """Return a local model response."""


MessageKind = Literal["user_message", "internal_thought", "module_message", "elapsed_time"]
ModuleStatus = Literal["active", "quiet", "saturated", "retired"]
TerminationVote = Literal[
    "answer_ready",
    "needs_one_more_round",
    "stay_silent",
    "blocked_by_missing_context",
]
TerminationReason = Literal[
    "answer_ready",
    "stay_silent",
    "max_rounds",
    "budget_exhausted",
    "blocked_by_missing_context",
]


@dataclass(frozen=True)
class ThoughtMessage:
    """Natural-language message envelope shared between thought modules."""

    message_id: str
    kind: MessageKind
    source_module_id: str | None
    target_module_id: str | None
    round_index: int
    text: str
    state_report: dict | None = None
    created_from_event_id: str | None = None


@dataclass(frozen=True)
class ThoughtModuleState:
    """Runtime state exposed to the fixed two-module coordinator."""

    module_id: str
    role_hint: str
    local_memory_summary: str = ""
    participation_budget_remaining: int = 1
    status: ModuleStatus = "active"
    last_trace_report: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ThoughtTurn:
    """Structured output from one module in one round."""

    module_id: str
    internal_thought: str
    module_message: str
    candidate_output: str
    termination_vote: TerminationVote


@dataclass(frozen=True)
class ThoughtRound:
    """Round buffer for one coordinator step."""

    round_index: int
    input_messages: list[ThoughtMessage]
    per_module_state_reports: dict[str, dict]
    per_module_internal_thoughts: dict[str, str]
    per_module_candidate_outputs: dict[str, str]
    termination_vote: str


@dataclass(frozen=True)
class ThoughtRuntimeResult:
    """Complete output from a fixed multi-module discussion."""

    rounds: list[ThoughtRound]
    messages: list[ThoughtMessage]
    final_response: str
    termination_reason: TerminationReason
    module_states: dict[str, ThoughtModuleState]


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

    def build_discussion_messages(
        self,
        *,
        user_text: str,
        state_report: dict,
        role_hint: str,
        local_memory_summary: str,
        peer_messages: list[ThoughtMessage],
        round_index: int,
    ) -> list[dict[str, str]]:
        system = (
            "너는 EmoNet 내부 사고 모듈이다. 사용자에게 직접 답하지 말고, "
            "다른 모듈과 공유할 짧은 내부 토의 턴을 JSON으로만 작성하라. "
            "감정 라벨을 단정하지 말고, 새로운 근거가 없으면 과도하게 확신하지 말라. "
            "반드시 internal_thought, module_message, candidate_output, termination_vote 키를 포함하라. "
            "termination_vote는 answer_ready, needs_one_more_round, stay_silent, "
            "blocked_by_missing_context 중 하나여야 한다."
        )
        peers = [
            {
                "kind": message.kind,
                "source_module_id": message.source_module_id,
                "round_index": message.round_index,
                "text": message.text,
            }
            for message in peer_messages
        ]
        user = (
            f"<module_id>{self.module_id}</module_id>\n"
            f"<role_hint>{role_hint}</role_hint>\n"
            f"<round_index>{round_index}</round_index>\n"
            f"<user_event>\n{user_text}\n</user_event>\n\n"
            f"<local_memory_summary>{local_memory_summary}</local_memory_summary>\n\n"
            "<neutral_internal_state>\n"
            f"{json.dumps(state_report, ensure_ascii=False, indent=2)}\n"
            "</neutral_internal_state>\n\n"
            "<peer_messages>\n"
            f"{json.dumps(peers, ensure_ascii=False, indent=2)}\n"
            "</peer_messages>"
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

    def generate_discussion_turn(
        self,
        *,
        user_text: str,
        state_report: dict,
        role_hint: str,
        local_memory_summary: str,
        peer_messages: list[ThoughtMessage],
        round_index: int,
        temperature: float = 0.7,
    ) -> ThoughtTurn:
        messages = self.build_discussion_messages(
            user_text=user_text,
            state_report=state_report,
            role_hint=role_hint,
            local_memory_summary=local_memory_summary,
            peer_messages=peer_messages,
            round_index=round_index,
        )
        raw = self.client.chat(messages, temperature=temperature)
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"discussion turn must be JSON: {raw}") from exc
        vote = payload.get("termination_vote")
        allowed_votes = {
            "answer_ready",
            "needs_one_more_round",
            "stay_silent",
            "blocked_by_missing_context",
        }
        if vote not in allowed_votes:
            raise ValueError(f"invalid termination_vote: {vote}")
        internal_thought = _clean_required_text(payload, "internal_thought")
        return ThoughtTurn(
            module_id=self.module_id,
            internal_thought=internal_thought,
            module_message=_clean_optional_text(payload, "module_message"),
            candidate_output=_clean_optional_text(payload, "candidate_output"),
            termination_vote=vote,
        )


class TwoModuleThoughtRuntime:
    """Fixed two-module coordinator for protocol smoke tests."""

    def __init__(
        self,
        *,
        modules: dict[str, ThoughtModule],
        module_states: dict[str, ThoughtModuleState],
        state_report_provider: Callable[[ThoughtModuleState, int], dict],
        temperature: float = 0.7,
    ) -> None:
        if len(modules) != 2:
            raise ValueError("TwoModuleThoughtRuntime requires exactly two modules")
        if set(modules) != set(module_states):
            raise ValueError("modules and module_states must have matching module IDs")
        self.modules = modules
        self.module_states = {key: replace(value) for key, value in module_states.items()}
        self.state_report_provider = state_report_provider
        self.temperature = temperature

    def run(self, *, user_text: str, max_rounds: int = 2) -> ThoughtRuntimeResult:
        if max_rounds <= 0:
            raise ValueError("max_rounds must be positive")
        if self._budget_exhausted():
            return ThoughtRuntimeResult(
                rounds=[],
                messages=[],
                final_response="",
                termination_reason="budget_exhausted",
                module_states=self.module_states,
            )

        messages = [
            ThoughtMessage(
                message_id="msg_0",
                kind="user_message",
                source_module_id=None,
                target_module_id=None,
                round_index=0,
                text=user_text,
            )
        ]
        rounds: list[ThoughtRound] = []
        final_response = ""
        termination_reason: TerminationReason = "max_rounds"
        next_message_index = 1

        for round_index in range(max_rounds):
            reports: dict[str, dict] = {}
            thoughts: dict[str, str] = {}
            candidates: dict[str, str] = {}
            votes: list[TerminationVote] = []
            round_start_messages = list(messages)

            for module_id, module in self.modules.items():
                state = self.module_states[module_id]
                if state.status != "active" or state.participation_budget_remaining <= 0:
                    continue
                report = self.state_report_provider(state, round_index)
                reports[module_id] = report
                peer_messages = [
                    message
                    for message in messages
                    if message.source_module_id != module_id and message.kind == "module_message"
                ]
                turn = module.generate_discussion_turn(
                    user_text=user_text,
                    state_report=report,
                    role_hint=state.role_hint,
                    local_memory_summary=state.local_memory_summary,
                    peer_messages=peer_messages,
                    round_index=round_index,
                    temperature=self.temperature,
                )
                thoughts[module_id] = turn.internal_thought
                candidates[module_id] = turn.candidate_output
                votes.append(turn.termination_vote)
                messages.append(
                    ThoughtMessage(
                        message_id=f"msg_{next_message_index}",
                        kind="internal_thought",
                        source_module_id=module_id,
                        target_module_id=None,
                        round_index=round_index,
                        text=turn.internal_thought,
                        state_report=report,
                    )
                )
                next_message_index += 1
                if turn.module_message:
                    messages.append(
                        ThoughtMessage(
                            message_id=f"msg_{next_message_index}",
                            kind="module_message",
                            source_module_id=module_id,
                            target_module_id=None,
                            round_index=round_index,
                            text=turn.module_message,
                            state_report=report,
                        )
                    )
                    next_message_index += 1
                self.module_states[module_id] = replace(
                    state,
                    participation_budget_remaining=state.participation_budget_remaining - 1,
                    last_trace_report=report,
                )

            round_vote = self._aggregate_vote(votes, candidates, round_index, max_rounds)
            rounds.append(
                ThoughtRound(
                    round_index=round_index,
                    input_messages=round_start_messages,
                    per_module_state_reports=reports,
                    per_module_internal_thoughts=thoughts,
                    per_module_candidate_outputs=candidates,
                    termination_vote=round_vote,
                )
            )
            if round_vote != "needs_one_more_round":
                termination_reason = round_vote
                final_response = _join_candidates(candidates)
                break

        return ThoughtRuntimeResult(
            rounds=rounds,
            messages=messages,
            final_response=final_response,
            termination_reason=termination_reason,
            module_states=self.module_states,
        )

    def _budget_exhausted(self) -> bool:
        return all(state.participation_budget_remaining <= 0 for state in self.module_states.values())

    def _aggregate_vote(
        self,
        votes: list[TerminationVote],
        candidates: dict[str, str],
        round_index: int,
        max_rounds: int,
    ) -> str:
        nonempty_candidates = [candidate for candidate in candidates.values() if candidate]
        if votes and all(vote == "answer_ready" for vote in votes):
            return "answer_ready"
        if "stay_silent" in votes and not nonempty_candidates:
            return "stay_silent"
        if "blocked_by_missing_context" in votes:
            return "blocked_by_missing_context"
        if round_index + 1 >= max_rounds:
            return "max_rounds"
        if self._budget_exhausted():
            return "budget_exhausted"
        return "needs_one_more_round"


def _clean_required_text(payload: dict, key: str) -> str:
    value = _clean_optional_text(payload, key)
    if not value:
        raise ValueError(f"{key} must not be empty")
    return value


def _clean_optional_text(payload: dict, key: str) -> str:
    value = payload.get(key, "")
    if value is None:
        return ""
    return " ".join(str(value).strip().splitlines()).strip()


def _join_candidates(candidates: dict[str, str]) -> str:
    return " ".join(candidate for candidate in candidates.values() if candidate).strip()
