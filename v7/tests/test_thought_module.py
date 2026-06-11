from __future__ import annotations

import torch

from emonet_v7.adaptive_rsnn import AdaptiveSparseRSNN
from emonet_v7.state_bridge import build_neutral_state_report
from emonet_v7.thought_module import ThoughtModule
from emonet_v7.trace_encoder import TraceEncoder, traces_to_sequences


class FakeChatClient:
    def __init__(self, response: str) -> None:
        self.response = response
        self.messages = None

    def chat(self, messages, *, temperature: float = 0.7) -> str:
        self.messages = messages
        return self.response


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
