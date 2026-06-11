from __future__ import annotations

import torch

from emonet_v7.adaptive_rsnn import AdaptiveSparseRSNN
from emonet_v7.state_bridge import build_neutral_state_report
from emonet_v7.thought_module import (
    ThoughtModule,
    ThoughtModuleState,
    TwoModuleThoughtRuntime,
)
from emonet_v7.trace_encoder import TraceEncoder, traces_to_sequences


class FakeChatClient:
    def __init__(self, response: str) -> None:
        self.response = response
        self.messages = None

    def chat(self, messages, *, temperature: float = 0.7) -> str:
        self.messages = messages
        return self.response


class QueueChatClient:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.messages = []

    def chat(self, messages, *, temperature: float = 0.7) -> str:
        self.messages.append(messages)
        if not self.responses:
            raise AssertionError("no queued response")
        return self.responses.pop(0)


def test_neutral_state_report_contains_no_emotion_labels() -> None:
    torch.manual_seed(4)
    snn = AdaptiveSparseRSNN(num_neurons=8, recurrent_density=0.25, seed=4)
    state = snn.initial_state(batch_size=1, device="cpu")
    _, traces = snn.run_window(
        event_current=torch.ones(1, 8),
        state=state,
        event_ticks=6,
        stimulation_ticks=2,
    )
    sequences = traces_to_sequences(traces)
    latent = TraceEncoder(num_neurons=8, hidden_dim=8, output_dim=4)(*sequences)
    report = build_neutral_state_report(traces=traces, latent_z=latent, stimulation_ticks=2)
    assert set(report) == {
        "active_ratio",
        "trace_persistence",
        "peak_spike_count",
        "final_spike_count",
        "latent_signature",
    }


def test_thought_module_builds_prompt_and_cleans_output() -> None:
    client = FakeChatClient("  바빠서 답장을 못 했을 수도 있다.\n")
    module = ThoughtModule(client)
    thought = module.generate_internal_thought(
        user_text="친구가 답장을 하지 않았다.",
        state_report={"active_ratio": 0.1},
    )
    assert thought == "바빠서 답장을 못 했을 수도 있다."
    assert client.messages is not None
    assert client.messages[0]["role"] == "system"
    assert "<neutral_internal_state>" in client.messages[1]["content"]


def test_two_module_runtime_runs_one_answer_ready_round() -> None:
    planner_client = QueueChatClient(
        [
            """
            {
              "internal_thought": "가능한 설명을 나누어 보자.",
              "module_message": "상황적 가능성을 먼저 제시하자.",
              "candidate_output": "바빠서 못 봤을 수도 있다.",
              "termination_vote": "answer_ready"
            }
            """
        ]
    )
    skeptic_client = QueueChatClient(
        [
            """
            {
              "internal_thought": "정보가 부족하니 단정하지 말자.",
              "module_message": "확신 표현은 낮추자.",
              "candidate_output": "필요하면 짧게 확인해 보자.",
              "termination_vote": "answer_ready"
            }
            """
        ]
    )
    runtime = TwoModuleThoughtRuntime(
        modules={
            "module_planner": ThoughtModule(planner_client, module_id="module_planner"),
            "module_skeptic": ThoughtModule(skeptic_client, module_id="module_skeptic"),
        },
        module_states={
            "module_planner": ThoughtModuleState(
                module_id="module_planner",
                role_hint="checks sequence, constraints, and next action",
                participation_budget_remaining=2,
            ),
            "module_skeptic": ThoughtModuleState(
                module_id="module_skeptic",
                role_hint="checks uncertainty, missing evidence, and risk",
                participation_budget_remaining=2,
            ),
        },
        state_report_provider=lambda state, round_index: {
            "module_id": state.module_id,
            "round_index": round_index,
            "active_ratio": 0.1,
        },
    )

    result = runtime.run(user_text="친구가 답장을 하지 않았다.", max_rounds=2)

    assert result.termination_reason == "answer_ready"
    assert len(result.rounds) == 1
    assert result.rounds[0].termination_vote == "answer_ready"
    assert result.final_response == "바빠서 못 봤을 수도 있다. 필요하면 짧게 확인해 보자."
    assert [message.kind for message in result.messages] == [
        "user_message",
        "internal_thought",
        "module_message",
        "internal_thought",
        "module_message",
    ]
    assert result.module_states["module_planner"].participation_budget_remaining == 1
    assert result.module_states["module_skeptic"].participation_budget_remaining == 1
    assert "<peer_messages>" in skeptic_client.messages[0][1]["content"]


def test_two_module_runtime_stops_when_budget_is_exhausted() -> None:
    runtime = TwoModuleThoughtRuntime(
        modules={
            "module_planner": ThoughtModule(QueueChatClient([]), module_id="module_planner"),
            "module_skeptic": ThoughtModule(QueueChatClient([]), module_id="module_skeptic"),
        },
        module_states={
            "module_planner": ThoughtModuleState(
                module_id="module_planner",
                role_hint="planner",
                participation_budget_remaining=0,
            ),
            "module_skeptic": ThoughtModuleState(
                module_id="module_skeptic",
                role_hint="skeptic",
                participation_budget_remaining=0,
            ),
        },
        state_report_provider=lambda state, round_index: {},
    )

    result = runtime.run(user_text="친구가 답장을 하지 않았다.", max_rounds=2)

    assert result.termination_reason == "budget_exhausted"
    assert result.rounds == []
    assert result.final_response == ""


def test_two_module_runtime_stops_on_stay_silent_without_candidates() -> None:
    silent_response = """
    {
      "internal_thought": "지금은 답하지 않는 편이 낫다.",
      "module_message": "",
      "candidate_output": "",
      "termination_vote": "stay_silent"
    }
    """
    runtime = TwoModuleThoughtRuntime(
        modules={
            "module_planner": ThoughtModule(QueueChatClient([silent_response]), module_id="module_planner"),
            "module_skeptic": ThoughtModule(QueueChatClient([silent_response]), module_id="module_skeptic"),
        },
        module_states={
            "module_planner": ThoughtModuleState("module_planner", "planner", participation_budget_remaining=1),
            "module_skeptic": ThoughtModuleState("module_skeptic", "skeptic", participation_budget_remaining=1),
        },
        state_report_provider=lambda state, round_index: {},
    )

    result = runtime.run(user_text="지금 바로 답해야 할까?", max_rounds=2)

    assert result.termination_reason == "stay_silent"
    assert result.final_response == ""


def test_two_module_runtime_stops_at_max_rounds() -> None:
    more_round_response = """
    {
      "internal_thought": "한 번 더 검토하자.",
      "module_message": "근거를 더 비교하자.",
      "candidate_output": "",
      "termination_vote": "needs_one_more_round"
    }
    """
    runtime = TwoModuleThoughtRuntime(
        modules={
            "module_planner": ThoughtModule(QueueChatClient([more_round_response]), module_id="module_planner"),
            "module_skeptic": ThoughtModule(QueueChatClient([more_round_response]), module_id="module_skeptic"),
        },
        module_states={
            "module_planner": ThoughtModuleState("module_planner", "planner", participation_budget_remaining=2),
            "module_skeptic": ThoughtModuleState("module_skeptic", "skeptic", participation_budget_remaining=2),
        },
        state_report_provider=lambda state, round_index: {},
    )

    result = runtime.run(user_text="한 번 더 볼까?", max_rounds=1)

    assert result.termination_reason == "max_rounds"
    assert len(result.rounds) == 1
