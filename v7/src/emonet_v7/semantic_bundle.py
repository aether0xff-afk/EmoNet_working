"""Shared model-bundle helpers for semantic training evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .adaptive_rsnn import AdaptiveSparseRSNN
from .event_encoder import EventEncoder
from .self_supervised import NextEventPredictor
from .trace_encoder import TraceEncoder


@dataclass
class ModelBundle:
    """Modules required for semantic next-event prediction."""

    event_encoder: EventEncoder
    snn: AdaptiveSparseRSNN
    trace_encoder: TraceEncoder
    predictor: NextEventPredictor

    def eval(self) -> None:
        self.event_encoder.eval()
        self.snn.eval()
        self.trace_encoder.eval()
        self.predictor.eval()


def build_bundle(*, text_dim: int, num_neurons: int, seed: int, device: torch.device) -> ModelBundle:
    """Build a deterministically initialized semantic-dynamics model bundle."""

    torch.manual_seed(seed)
    event_encoder = EventEncoder(text_embedding_dim=text_dim, num_neurons=num_neurons).to(device)
    snn = AdaptiveSparseRSNN(
        num_neurons=num_neurons,
        recurrent_density=0.10,
        seed=seed,
        recurrent_weight_std=0.30,
        input_weight_std=0.15,
    ).to(device)
    trace_encoder = TraceEncoder(num_neurons=num_neurons, hidden_dim=64, output_dim=64).to(device)
    predictor = NextEventPredictor(latent_dim=64, hidden_dim=128, embedding_dim=text_dim).to(device)
    return ModelBundle(
        event_encoder=event_encoder,
        snn=snn,
        trace_encoder=trace_encoder,
        predictor=predictor,
    )


def load_trained_bundle(*, checkpoint: dict, text_dim: int, device: torch.device) -> ModelBundle:
    """Rebuild and load one checkpoint bundle."""

    saved_args = checkpoint["args"]
    bundle = build_bundle(
        text_dim=text_dim,
        num_neurons=int(saved_args["num_neurons"]),
        seed=int(saved_args["seed"]),
        device=device,
    )
    bundle.event_encoder.load_state_dict(checkpoint["event_encoder"])
    bundle.snn.load_state_dict(checkpoint["snn"])
    bundle.trace_encoder.load_state_dict(checkpoint["trace_encoder"])
    bundle.predictor.load_state_dict(checkpoint["predictor"])
    return bundle
