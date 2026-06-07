from __future__ import annotations

import torch

from emonet_v7.adaptive_rsnn import AdaptiveSparseRSNN
from emonet_v7.event_encoder import EventEncoder
from emonet_v7.lmstudio_client import LMStudioClient
from emonet_v7.schemas import Event
from emonet_v7.selectivity import cosine_distance, encode_event_trace
from emonet_v7.text_encoder import DeterministicHashTextEncoder
from emonet_v7.trace_encoder import TraceEncoder, traces_to_sequences


def test_hash_encoder_is_deterministic_and_normalized() -> None:
    encoder = DeterministicHashTextEncoder(output_dim=32)
    first = encoder.encode(["hello"])
    second = encoder.encode(["hello"])
    assert torch.allclose(first, second)
    assert torch.allclose(first.norm(dim=-1), torch.ones(1))


def test_event_encoder_shape_and_speaker_mapping() -> None:
    torch.manual_seed(1)
    encoder = EventEncoder(text_embedding_dim=32, num_neurons=16)
    events = [
        Event("1", "user_message", "a", "human"),
        Event("2", "internal_thought", "b", "module_0"),
    ]
    output = encoder(torch.randn(2, 32), events)
    assert output.shape == (2, 16)
    assert set(encoder.speaker_to_id) == {"human", "module_0"}


def test_trace_encoder_shape() -> None:
    torch.manual_seed(1)
    snn = AdaptiveSparseRSNN(num_neurons=16, recurrent_density=0.10, seed=1)
    state = snn.initial_state(batch_size=1, device="cpu")
    _, traces = snn.run_window(
        event_current=torch.ones(1, 16),
        state=state,
        event_ticks=8,
        stimulation_ticks=2,
    )
    sequences = traces_to_sequences(traces)
    latent_z = TraceEncoder(num_neurons=16)(*sequences)
    assert latent_z.shape == (1, 64)


def test_same_event_repeats_exactly_with_fixed_state() -> None:
    torch.manual_seed(2)
    text_encoder = DeterministicHashTextEncoder(output_dim=32)
    event_encoder = EventEncoder(text_embedding_dim=32, num_neurons=16)
    snn = AdaptiveSparseRSNN(num_neurons=16, recurrent_density=0.20, seed=2)
    trace_encoder = TraceEncoder(num_neurons=16)
    event = Event("1", "user_message", "same sentence", "human")
    first = encode_event_trace(
        event=event,
        text_encoder=text_encoder,
        event_encoder=event_encoder,
        snn=snn,
        trace_encoder=trace_encoder,
        event_ticks=8,
        stimulation_ticks=2,
    )
    second = encode_event_trace(
        event=event,
        text_encoder=text_encoder,
        event_encoder=event_encoder,
        snn=snn,
        trace_encoder=trace_encoder,
        event_ticks=8,
        stimulation_ticks=2,
    )
    assert cosine_distance(first.latent_z, second.latent_z) < 1e-6


def test_lmstudio_base_url_is_normalized() -> None:
    client = LMStudioClient(base_url="http://localhost:1234", model="local-model")
    assert client.base_url == "http://localhost:1234/v1"
