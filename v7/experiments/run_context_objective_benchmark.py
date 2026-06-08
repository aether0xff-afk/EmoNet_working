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
        if not args.base_url:
            raise ValueError("--base-url is required when --encoder lmstudio is used")
        client = LMStudioClient(base_url=args.base_url, model=args.embedding_model)
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


def parameters_for(model_type: str, model) -> list[torch.nn.Parameter]:
    if model_type.startswith("snn_"):
        return (
            list(model.event_encoder.parameters())
            + list(model.snn.parameters())
            + list(model.trace_encoder.parameters())
            + list(model.predictor.parameters())
        )
    return list(model.parameters())


def set_mode(model_type: str, model, *, training: bool) -> None:
    if model_type.startswith("snn_"):
        modules = [model.event_encoder, model.snn, model.trace_encoder, model.predictor]
    else:
        modules = [model]
    for module in modules:
        module.train(training)


def predict(
    *,
    model_type: str,
    model,
    episode: Episode,
    step_index: int,
    text_encoder,
    args: argparse.Namespace,
    device: torch.device,
    history_source: Episode | None = None,
) -> PredictionOutput:
    if model_type.startswith("snn_"):
        return run_snn(
            bundle=model,
            episode=episode,
            step_index=step_index,
            text_encoder=text_encoder,
            event_ticks=args.event_ticks,
            stimulation_ticks=args.stimulation_ticks,
            device=device,
            history_source=history_source,
        )
    if model_type == "context_free_mlp":
        return run_context_free(model=model, episode=episode, step_index=step_index, text_encoder=text_encoder, device=device)
    return run_gru(
        model=model,
        episode=episode,
        step_index=step_index,
        text_encoder=text_encoder,
        device=device,
        history_source=history_source,
    )


def pair_objective(
    *,
    model_type: str,
    left: PredictionOutput,
    right: PredictionOutput,
    context_weight: float,
    ranking_margin: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    next_loss = (
        (1.0 - F.cosine_similarity(left.prediction, left.target, dim=-1)).mean()
        + (1.0 - F.cosine_similarity(right.prediction, right.target, dim=-1)).mean()
    ) / 2.0
    ranking = context_ranking_loss(
        left_prediction=left.prediction,
        left_target=left.target,
        right_prediction=right.prediction,
        right_target=right.target,
        margin=ranking_margin,
    )
    effective_weight = 0.0 if model_type == "snn_next_only" else context_weight
    regularization = (left.regularization + right.regularization) / 2.0
    total = next_loss + effective_weight * ranking + regularization
    margin_value = context_margin(
        left_prediction=left.prediction,
        left_target=left.target,
        right_prediction=right.prediction,
        right_target=right.target,
    )
    metrics = {
        "total": float(total.detach()),
        "next_event": float(next_loss.detach()),
        "context_ranking": float(ranking.detach()),
        "regularization": float(regularization.detach()),
        "context_margin": float(margin_value.detach()),
        "prediction_distance": float(cosine_distance(left.prediction, right.prediction).detach()),
        "representation_distance": float(cosine_distance(left.representation, right.representation).detach()),
    }
    return total, metrics


def aggregate(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = ["total", "next_event", "context_ranking", "regularization", "context_margin", "prediction_distance", "representation_distance"]
    return {key: sum(row[key] for row in rows) / max(1, len(rows)) for key in keys}


def evaluate(
    *,
    model_type: str,
    model,
    pairs: list[ContrastPair],
    episode_by_id: dict[str, Episode],
    text_encoder,
    args: argparse.Namespace,
    device: torch.device,
    shuffle_history: bool = False,
) -> dict[str, float]:
    set_mode(model_type, model, training=False)
    rows = []
    with torch.no_grad():
        for pair in pairs:
            left_episode = episode_by_id[pair.left_episode_id]
            right_episode = episode_by_id[pair.right_episode_id]
            left_history = right_episode if shuffle_history else None
            right_history = left_episode if shuffle_history else None
            left = predict(
                model_type=model_type,
                model=model,
                episode=left_episode,
                step_index=pair.step_index,
                text_encoder=text_encoder,
                args=args,
                device=device,
                history_source=left_history,
            )
            right = predict(
                model_type=model_type,
                model=model,
                episode=right_episode,
                step_index=pair.step_index,
                text_encoder=text_encoder,
                args=args,
                device=device,
                history_source=right_history,
            )
            _, metrics = pair_objective(
                model_type=model_type,
                left=left,
                right=right,
                context_weight=args.context_weight,
                ranking_margin=args.context_margin,
            )
            rows.append(metrics)
    return aggregate(rows)


def state_dict_for(model_type: str, model) -> dict[str, Any]:
    if model_type.startswith("snn_"):
        return {
            "event_encoder": model.event_encoder.state_dict(),
            "snn": model.snn.state_dict(),
            "trace_encoder": model.trace_encoder.state_dict(),
            "predictor": model.predictor.state_dict(),
        }
    return {"model": model.state_dict()}


def train_one(
    *,
    model_type: str,
    seed: int,
    train_pairs: list[ContrastPair],
    validation_pairs: list[ContrastPair],
    episode_by_id: dict[str, Episode],
    text_encoder,
    args: argparse.Namespace,
    device: torch.device,
    output: Path,
    logger: RunLogger,
) -> dict[str, Any]:
    model = build_model(model_type, text_dim=text_encoder.output_dim, num_neurons=args.num_neurons, seed=seed, device=device)
    parameters = parameters_for(model_type, model)
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=1e-4)
    model_output = output / f"seed_{seed}" / model_type
    model_output.mkdir(parents=True, exist_ok=True)
    best_validation = float("inf")
    best_metrics: dict[str, float] | None = None
    history: list[dict[str, float | int]] = []

    logger.log("model.start", "모델 학습을 시작한다.", seed=seed, model_type=model_type, parameter_count=sum(parameter.numel() for parameter in parameters))
    for epoch in range(args.epochs):
        set_mode(model_type, model, training=True)
        epoch_rows = []
        for pair in train_pairs:
            optimizer.zero_grad()
            left_episode = episode_by_id[pair.left_episode_id]
            right_episode = episode_by_id[pair.right_episode_id]
            left = predict(model_type=model_type, model=model, episode=left_episode, step_index=pair.step_index, text_encoder=text_encoder, args=args, device=device)
            right = predict(model_type=model_type, model=model, episode=right_episode, step_index=pair.step_index, text_encoder=text_encoder, args=args, device=device)
            total, metrics = pair_objective(
                model_type=model_type,
                left=left,
                right=right,
                context_weight=args.context_weight,
                ranking_margin=args.context_margin,
            )
            total.backward()
            nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
            optimizer.step()
            epoch_rows.append(metrics)
        train_metrics = aggregate(epoch_rows)
        validation_metrics = evaluate(
            model_type=model_type,
            model=model,
            pairs=validation_pairs,
            episode_by_id=episode_by_id,
            text_encoder=text_encoder,
            args=args,
            device=device,
        )
        row = {
            "epoch": epoch,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"validation_{key}": value for key, value in validation_metrics.items()},
        }
        history.append(row)
        logger.log("epoch.done", "모델 epoch를 마쳤다.", seed=seed, model_type=model_type, **row)
        if validation_metrics["total"] < best_validation:
            best_validation = validation_metrics["total"]
            best_metrics = validation_metrics
            torch.save(
                {
                    "seed": seed,
                    "model_type": model_type,
                    "args": vars(args),
                    "validation": validation_metrics,
                    **state_dict_for(model_type, model),
                },
                model_output / "best_checkpoint.pt",
            )

    pd.DataFrame(history).to_csv(model_output / "history.csv", index=False, encoding="utf-8-sig")
    validation_real = evaluate(
        model_type=model_type,
        model=model,
        pairs=validation_pairs,
        episode_by_id=episode_by_id,
        text_encoder=text_encoder,
        args=args,
        device=device,
    )
    validation_shuffled = None
    if model_type.startswith("snn_") or model_type == "gru_context_contrastive":
        validation_shuffled = evaluate(
            model_type=model_type,
            model=model,
            pairs=validation_pairs,
            episode_by_id=episode_by_id,
            text_encoder=text_encoder,
            args=args,
            device=device,
            shuffle_history=True,
        )
    result = {
        "seed": seed,
        "model_type": model_type,
        "best_validation_total": best_validation,
        "best_validation": best_metrics,
        "final_validation_real": validation_real,
        "final_validation_shuffled": validation_shuffled,
    }
    (model_output / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.log("model.done", "모델 학습과 평가를 마쳤다.", **result)
    return result


def flatten_result(result: dict[str, Any]) -> dict[str, Any]:
    real = result["final_validation_real"]
    shuffled = result["final_validation_shuffled"]
    row: dict[str, Any] = {
        "seed": result["seed"],
        "model_type": result["model_type"],
        "best_validation_total": result["best_validation_total"],
        **{f"real_{key}": value for key, value in real.items()},
    }
    if shuffled is not None:
        row.update({f"shuffled_{key}": value for key, value in shuffled.items()})
        row["real_minus_shuffled_context_margin"] = real["context_margin"] - shuffled["context_margin"]
    return row


def main() -> None:
    args = parse_args()
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if not args.seeds:
        raise ValueError("--seeds must not be empty")
    if args.context_weight < 0:
        raise ValueError("--context-weight must be non-negative")
    if args.context_margin < 0:
        raise ValueError("--context-margin must be non-negative")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    logger = RunLogger(output_dir=output, verbose=not args.quiet)
    logger.section("context objective benchmark")
    logger.log("config", "Benchmark 설정을 불러왔다.", **vars(args))

    device = torch.device(args.device)
    episodes = load_episodes(args.fixture)
    episode_by_id = {episode.episode_id: episode for episode in episodes}
    train_pairs = load_contrast_pairs(args.fixture, split="train")
    validation_pairs = load_contrast_pairs(args.fixture, split="validation")
    validate_contrast_pairs(episodes, train_pairs + validation_pairs)
    if not train_pairs or not validation_pairs:
        raise ValueError("fixture must contain train and validation contrast pairs")
    logger.log("dataset.ready", "맥락 대조 fixture를 불러왔다.", episode_count=len(episodes), train_pair_count=len(train_pairs), validation_pair_count=len(validation_pairs))

    text_encoder = build_text_encoder(args, output)
    logger.log("embedding.ready", "Embedding encoder와 캐시가 준비됐다.", output_dim=text_encoder.output_dim)

    results: list[dict[str, Any]] = []
    for seed in args.seeds:
        logger.section(f"seed={seed}")
        for model_type in MODEL_TYPES:
            results.append(
                train_one(
                    model_type=model_type,
                    seed=seed,
                    train_pairs=train_pairs,
                    validation_pairs=validation_pairs,
                    episode_by_id=episode_by_id,
                    text_encoder=text_encoder,
                    args=args,
                    device=device,
                    output=output,
                    logger=logger,
                )
            )

    frame = pd.DataFrame([flatten_result(result) for result in results])
    frame.to_csv(output / "by_seed_model.csv", index=False, encoding="utf-8-sig")
    numeric_columns = [column for column in frame.columns if column not in {"seed", "model_type"}]
    summary = frame.groupby("model_type")[numeric_columns].agg(["mean", "std", "min", "max"])
    summary.columns = ["_".join(column) for column in summary.columns]
    summary.reset_index().to_csv(output / "summary_by_model.csv", index=False, encoding="utf-8-sig")
    metadata = {
        "fixture": args.fixture,
        "encoder": args.encoder,
        "seeds": args.seeds,
        "epochs": args.epochs,
        "models": list(MODEL_TYPES),
        "context_weight": args.context_weight,
        "context_margin": args.context_margin,
        "note": (
            "Benchmark compares next-only SNN, context-contrastive SNN, context-free MLP, and GRU. "
            "Positive real-minus-shuffled context margin suggests reliance on correct prior context. "
            "This is not evidence of emotional semantics."
        ),
    }
    (output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.log("benchmark.done", "Context objective benchmark를 마쳤다.", files=["run_log.jsonl", "embedding_cache.json", "by_seed_model.csv", "summary_by_model.csv", "metadata.json"])
    print(summary.to_string())


if __name__ == "__main__":
    main()
