"""Benchmark context-sensitive objectives and recurrent baselines across seeds."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from emonet_v7.context_objective import (  # noqa: E402
    ContrastPair,
    ContextFreeMLP,
    GRUContextPredictor,
    context_margin,
    context_ranking_loss,
    cosine_distance,
    load_contrast_pairs,
    validate_contrast_pairs,
)
from emonet_v7.embedding_cache import CachedTextEncoder  # noqa: E402
from emonet_v7.episode_dataset import Episode, load_episodes  # noqa: E402
from emonet_v7.lmstudio_client import LMStudioClient  # noqa: E402
from emonet_v7.remote_config import load_default_lmstudio_base_url  # noqa: E402
from emonet_v7.run_logger import RunLogger  # noqa: E402
from emonet_v7.semantic_bundle import ModelBundle, build_bundle  # noqa: E402
from emonet_v7.text_encoder import DeterministicHashTextEncoder, LMStudioEmbeddingTextEncoder  # noqa: E402
from emonet_v7.trace_encoder import traces_to_sequences  # noqa: E402
from emonet_v7.training_window import DifferentiableWindow, run_differentiable_window  # noqa: E402


MODEL_TYPES = (
    "snn_next_only",
    "snn_context_contrastive",
    "context_free_mlp",
    "gru_context_contrastive",
)


@dataclass
class PredictionOutput:
    prediction: torch.Tensor
    target: torch.Tensor
    representation: torch.Tensor
    regularization: torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", default="fixtures/context_dependence_episodes.yaml")
    parser.add_argument("--output", default="runs/context_objective_benchmark")
    parser.add_argument("--encoder", choices=["hash", "lmstudio"], default="hash")
    parser.add_argument("--base-url")
    parser.add_argument("--embedding-model", default="text-embedding-nomic-embed-text-v1.5")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--num-neurons", type=int, default=128)
    parser.add_argument("--event-ticks", type=int, default=16)
    parser.add_argument("--stimulation-ticks", type=int, default=6)
    parser.add_argument("--context-weight", type=float, default=1.0)
    parser.add_argument("--context-margin", type=float, default=0.05)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 13, 21, 42, 100])
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def build_text_encoder(args: argparse.Namespace, output: Path):
    if args.encoder == "hash":
        encoder = DeterministicHashTextEncoder(output_dim=128)
    else:
        base_url = args.base_url or load_default_lmstudio_base_url()
        if not base_url:
            raise ValueError(
                "--base-url is required when --encoder lmstudio is used unless "
                "EMONET_LMSTUDIO_BASE_URL is configured"
            )
        client = LMStudioClient(base_url=base_url, model=args.embedding_model)
        encoder = LMStudioEmbeddingTextEncoder(client, args.embedding_model)
    return CachedTextEncoder(encoder, output / "embedding_cache.json")


def sequence_events(episode: Episode, step_index: int, history_source: Episode | None = None):
    """Return prior events plus the current event, optionally with swapped history."""

    source = history_source or episode
    return tuple(source.events[:step_index]) + (episode.events[step_index],)


def snn_regularization(window: DifferentiableWindow) -> torch.Tensor:
    """Keep firing activity finite without imposing semantic labels."""

    firing_rate = (window.spike.mean() - 0.10) ** 2
    neuron_activity = window.spike.sum(dim=(0, 1))
    inactive_neuron = torch.exp(-neuron_activity).mean()
    excess = torch.relu(window.membrane.abs() - 5.0)
    stability = (excess ** 2).mean()
    return 0.10 * firing_rate + 0.01 * inactive_neuron + 0.01 * stability


def run_snn(
    *,
    bundle: ModelBundle,
    episode: Episode,
    step_index: int,
    text_encoder,
    event_ticks: int,
    stimulation_ticks: int,
    device: torch.device,
    history_source: Episode | None = None,
) -> PredictionOutput:
    state = bundle.snn.initial_state(batch_size=1, device=device)
    last_window = None
    for event in sequence_events(episode, step_index, history_source):
        embedding = text_encoder.encode([event.text]).to(device)
        current = bundle.event_encoder(embedding, [event])
        state, last_window = run_differentiable_window(
            snn=bundle.snn,
            event_current=current,
            state=state,
            event_ticks=event_ticks,
            stimulation_ticks=stimulation_ticks,
        )
    if last_window is None:
        raise RuntimeError("SNN sequence must contain at least one event")
    latent = bundle.trace_encoder(last_window.spike, last_window.membrane, last_window.adaptation)
    prediction = bundle.predictor(latent)
    target = text_encoder.encode([episode.events[step_index + 1].text]).to(device)
    return PredictionOutput(
        prediction=prediction,
        target=target,
        representation=latent,
        regularization=snn_regularization(last_window),
    )


def run_context_free(*, model: ContextFreeMLP, episode: Episode, step_index: int, text_encoder, device: torch.device) -> PredictionOutput:
    current = text_encoder.encode([episode.events[step_index].text]).to(device)
    target = text_encoder.encode([episode.events[step_index + 1].text]).to(device)
    return PredictionOutput(
        prediction=model(current),
        target=target,
        representation=model.encode_context(current),
        regularization=torch.zeros((), device=device),
    )


def run_gru(
    *,
    model: GRUContextPredictor,
    episode: Episode,
    step_index: int,
    text_encoder,
    device: torch.device,
    history_source: Episode | None = None,
) -> PredictionOutput:
    texts = [event.text for event in sequence_events(episode, step_index, history_source)]
    sequence = text_encoder.encode(texts).unsqueeze(0).to(device)
    target = text_encoder.encode([episode.events[step_index + 1].text]).to(device)
    representation = model.encode_context(sequence)
    return PredictionOutput(
        prediction=F.normalize(model.projection(representation), dim=-1),
        target=target,
        representation=representation,
        regularization=torch.zeros((), device=device),
    )


def build_model(model_type: str, *, text_dim: int, num_neurons: int, seed: int, device: torch.device):
    torch.manual_seed(seed)
    if model_type.startswith("snn_"):
        return build_bundle(text_dim=text_dim, num_neurons=num_neurons, seed=seed, device=device)
    if model_type == "context_free_mlp":
        return ContextFreeMLP(embedding_dim=text_dim).to(device)
    if model_type == "gru_context_contrastive":
        return GRUContextPredictor(embedding_dim=text_dim).to(device)
    raise ValueError(f"unsupported model type: {model_type}")
