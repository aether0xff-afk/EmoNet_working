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


def parameters_for(model_type: str, model) -> list[torch.nn.Parameter]:
    if model_type.startswith("snn_"):
        return [
            *model.event_encoder.parameters(),
            *model.snn.parameters(),
            *model.trace_encoder.parameters(),
            *model.predictor.parameters(),
        ]
    return list(model.parameters())


def set_mode(model_type: str, model, *, training: bool) -> None:
    modules = (
        (model.event_encoder, model.snn, model.trace_encoder, model.predictor)
        if model_type.startswith("snn_")
        else (model,)
    )
    for module in modules:
        module.train(training)


def state_dict_for(model_type: str, model) -> dict[str, Any]:
    if model_type.startswith("snn_"):
        return {
            "event_encoder": model.event_encoder.state_dict(),
            "snn": model.snn.state_dict(),
            "trace_encoder": model.trace_encoder.state_dict(),
            "predictor": model.predictor.state_dict(),
        }
    return {"model": model.state_dict()}


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
        return run_context_free(
            model=model,
            episode=episode,
            step_index=step_index,
            text_encoder=text_encoder,
            device=device,
        )
    if model_type == "gru_context_contrastive":
        return run_gru(
            model=model,
            episode=episode,
            step_index=step_index,
            text_encoder=text_encoder,
            device=device,
            history_source=history_source,
        )
    raise ValueError(f"unsupported model type: {model_type}")


def pair_objective(
    *,
    model_type: str,
    left: PredictionOutput,
    right: PredictionOutput,
    context_weight: float,
    ranking_margin: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    next_loss = (cosine_distance(left.prediction, left.target) + cosine_distance(right.prediction, right.target)) / 2.0
    ranking = torch.zeros((), device=left.prediction.device)
    if model_type != "snn_next_only":
        ranking = context_ranking_loss(
            left_prediction=left.prediction,
            left_target=left.target,
            right_prediction=right.prediction,
            right_target=right.target,
            margin=ranking_margin,
        )
    regularization = (left.regularization + right.regularization) / 2.0
    total = next_loss + context_weight * ranking + regularization
    margin_value = context_margin(
        left_prediction=left.prediction,
        left_target=left.target,
        right_prediction=right.prediction,
        right_target=right.target,
    )
    representation_distance = cosine_distance(left.representation, right.representation)
    return total, {
        "total": float(total.detach().cpu()),
        "next_loss": float(next_loss.detach().cpu()),
        "ranking_loss": float(ranking.detach().cpu()),
        "regularization": float(regularization.detach().cpu()),
        "context_margin": float(margin_value.detach().cpu()),
        "representation_distance": float(representation_distance.detach().cpu()),
    }


def aggregate(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = rows[0].keys()
    return {key: sum(row[key] for row in rows) / len(rows) for key in keys}


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
    metrics = aggregate(rows)
    if shuffle_history:
        metrics = {f"shuffled_{key}": value for key, value in metrics.items()}
    return metrics


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
    model = build_model(
        model_type,
        text_dim=text_encoder.output_dim,
        num_neurons=args.num_neurons,
        seed=seed,
        device=device,
    )
    parameters = parameters_for(model_type, model)
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=1e-4)
    model_output = output / f"seed_{seed}" / model_type
    model_output.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float | int]] = []
    best_validation = float("inf")
    best_metrics: dict[str, float] | None = None
    best_epoch = -1

    logger.log("model.start", "모델 학습을 시작한다.", seed=seed, model_type=model_type)
    for epoch in range(args.epochs):
        set_mode(model_type, model, training=True)
        epoch_rows = []
        for pair in train_pairs:
            optimizer.zero_grad()
            left_episode = episode_by_id[pair.left_episode_id]
            right_episode = episode_by_id[pair.right_episode_id]
            left = predict(
                model_type=model_type,
                model=model,
                episode=left_episode,
                step_index=pair.step_index,
                text_encoder=text_encoder,
                args=args,
                device=device,
            )
            right = predict(
                model_type=model_type,
                model=model,
                episode=right_episode,
                step_index=pair.step_index,
                text_encoder=text_encoder,
                args=args,
                device=device,
            )
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
        if validation_metrics["total"] < best_validation:
            best_validation = validation_metrics["total"]
            best_metrics = validation_metrics
            best_epoch = epoch
            torch.save(
                {
                    "seed": seed,
                    "model_type": model_type,
                    "args": vars(args),
                    "epoch": epoch,
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
        "best_epoch": best_epoch,
        "best_validation_total": best_validation,
        "best_validation": best_metrics,
        "final_validation_real": validation_real,
        "final_validation_shuffled": validation_shuffled,
    }
    (model_output / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.log("model.done", "모델 평가를 마쳤다.", **result)
    return result


def summarize_results(results: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for result in results:
        real = result["final_validation_real"]
        shuffled = result["final_validation_shuffled"] or {}
        row = {
            "seed": result["seed"],
            "model_type": result["model_type"],
            "best_epoch": result["best_epoch"],
            "best_validation_total": result["best_validation_total"],
            "real_context_margin": real.get("context_margin"),
            "real_total": real.get("total"),
            "real_representation_distance": real.get("representation_distance"),
        }
        if shuffled:
            shuffled_margin = shuffled.get("shuffled_context_margin")
            row["shuffled_context_margin"] = shuffled_margin
            row["real_minus_shuffled_context_margin"] = (
                row["real_context_margin"] - shuffled_margin
                if row["real_context_margin"] is not None and shuffled_margin is not None
                else None
            )
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    logger = RunLogger(output_dir=output, verbose=not args.quiet)
    logger.section("context objective benchmark")
    device = torch.device(args.device)
    episodes = load_episodes(args.fixture)
    train_pairs = load_contrast_pairs(args.fixture, split="train")
    validation_pairs = load_contrast_pairs(args.fixture, split="validation")
    validate_contrast_pairs(episodes, train_pairs + validation_pairs)
    episode_by_id = {episode.episode_id: episode for episode in episodes}
    text_encoder = build_text_encoder(args, output)

    results: list[dict[str, Any]] = []
    for seed in args.seeds:
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

    by_seed = summarize_results(results)
    by_seed.to_csv(output / "by_seed_model.csv", index=False, encoding="utf-8-sig")
    metric_columns = [
        column
        for column in by_seed.columns
        if column not in {"seed", "model_type"} and pd.api.types.is_numeric_dtype(by_seed[column])
    ]
    summary = by_seed.groupby("model_type")[metric_columns].agg(["mean", "std", "min", "max"])
    summary.columns = ["_".join(column) for column in summary.columns]
    summary.reset_index().to_csv(output / "summary_by_model.csv", index=False, encoding="utf-8-sig")
    metadata = {
        "fixture": args.fixture,
        "encoder": args.encoder,
        "embedding_model": args.embedding_model if args.encoder == "lmstudio" else None,
        "base_url": args.base_url or load_default_lmstudio_base_url() if args.encoder == "lmstudio" else None,
        "epochs": args.epochs,
        "seeds": args.seeds,
        "model_types": MODEL_TYPES,
        "device": args.device,
        "note": "Context objective benchmark. This is not evidence of emotional semantics.",
    }
    (output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.log(
        "output.saved",
        "benchmark artifacts를 저장했다.",
        files=["run_log.jsonl", "embedding_cache.json", "by_seed_model.csv", "summary_by_model.csv", "metadata.json"],
        output_dir=str(output),
    )


if __name__ == "__main__":
    main()
