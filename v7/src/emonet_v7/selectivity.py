"""Milestone 2 sentence-to-trace selectivity pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from .adaptive_rsnn import AdaptiveSparseRSNN
from .event_encoder import EventEncoder
from .schemas import Event
from .text_encoder import TextEncoder
from .trace_encoder import TraceEncoder, traces_to_sequences


@dataclass
class TraceResult:
    event: Event
    text_embedding: torch.Tensor
    current: torch.Tensor
    latent_z: torch.Tensor
    spike_count: int


def encode_event_trace(
    *,
    event: Event,
    text_encoder: TextEncoder,
    event_encoder: EventEncoder,
    snn: AdaptiveSparseRSNN,
    trace_encoder: TraceEncoder,
    event_ticks: int,
    stimulation_ticks: int,
    device: str = "cpu",
) -> TraceResult:
    """Encode one event from a zero initial state and return its trace latent."""

    embedding = text_encoder.encode([event.text]).to(device)
    current = event_encoder(embedding, [event])
    state = snn.initial_state(batch_size=1, device=device)
    _, traces = snn.run_window(
        event_current=current,
        state=state,
        event_ticks=event_ticks,
        stimulation_ticks=stimulation_ticks,
    )
    sequences = traces_to_sequences(traces)
    latent_z = trace_encoder(*(sequence.to(device) for sequence in sequences))
    return TraceResult(
        event=event,
        text_embedding=embedding.detach().cpu(),
        current=current.detach().cpu(),
        latent_z=latent_z.detach().cpu(),
        spike_count=sum(int(trace.spike.sum()) for trace in traces),
    )


def cosine_distance(left: torch.Tensor, right: torch.Tensor) -> float:
    """Return mean cosine distance for matching batches."""

    return float((1.0 - F.cosine_similarity(left, right, dim=-1)).mean())
