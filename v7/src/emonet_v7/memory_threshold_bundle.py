"""Model-bundle helpers for neuron-memory-threshold SNN experiments."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .event_encoder import EventEncoder
from .memory_threshold_rsnn import NeuronMemoryThresholdRSNN
from .memory_threshold_trace_encoder import MemoryThresholdTraceEncoder
from .self_supervised import NextEventPredictor


@dataclass
class MemoryThresholdModelBundle:
    """Modules required for memory-threshold semantic next-event prediction."""

    event_encoder: EventEncoder
    snn: NeuronMemoryThresholdRSNN
    trace_encoder: MemoryThresholdTraceEncoder
    predictor: NextEventPredictor

    def eval(self) -> None:
        self.event_encoder.eval()
        self.snn.eval()
        self.trace_encoder.eval()
        self.predictor.eval()


def build_memory_threshold_bundle(
    *,
    text_dim: int,
    num_neurons: int,
    seed: int,
    device: torch.device,
    memory_feedback_strength: float,
    accumulation_decay: float = 0.85,
    memory_threshold: float = 0.60,
    memory_decay: float = 0.98,
) -> MemoryThresholdModelBundle:
    """Build one deterministically initialized memory-threshold SNN bundle."""

    torch.manual_seed(seed)
    event_encoder = EventEncoder(text_embedding_dim=text_dim, num_neurons=num_neurons).to(device)
    snn = NeuronMemoryThresholdRSNN(
        num_neurons=num_neurons,
        recurrent_density=0.10,
        seed=seed,
        recurrent_weight_std=0.30,
        input_weight_std=0.15,
        accumulation_decay=accumulation_decay,
        memory_threshold=memory_threshold,
        memory_decay=memory_decay,
        memory_feedback_strength=memory_feedback_strength,
    ).to(device)
    trace_encoder = MemoryThresholdTraceEncoder(num_neurons=num_neurons, hidden_dim=64, output_dim=64).to(device)
    predictor = NextEventPredictor(latent_dim=64, hidden_dim=128, embedding_dim=text_dim).to(device)
    return MemoryThresholdModelBundle(
        event_encoder=event_encoder,
        snn=snn,
        trace_encoder=trace_encoder,
        predictor=predictor,
    )
