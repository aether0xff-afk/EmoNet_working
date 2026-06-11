"""Train and evaluate a history-reconstructive SNN without semantic-label supervision.

The auxiliary objective uses frozen text embeddings of prior events only.  It
never reads semantic axis labels.  The goal is to test whether explicitly
preserving prior-event information inside the SNN trace improves later
semantic readability.
"""

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

import run_context_objective_benchmark as base
import run_trace_semantic_alignment_benchmark as semantic

from emonet_v7.context_objective import context_margin, context_ranking_loss, cosine_distance
from emonet_v7.semantic_bundle import ModelBundle, build_bundle
from emonet_v7.self_supervised import NextEventPredictor


MODEL_TYPE = "snn_context_history_reconstructive"
INTERPRETATION_BOUNDARY = (
    "This experiment tests whether a label-free prior-event reconstruction objective improves semantic readability of SNN traces. "
    "It does not establish ground-truth emotions, biological fidelity, emergent clusters, or broad real-world generalization."
)


@dataclass
class HistoryReconstructiveBundle:
    base: ModelBundle
    history_predictor: NextEventPredictor


@dataclass
class PredictionOutput:
    prediction: torch.Tensor
    target: torch.Tensor
    representation: torch.Tensor
    regularization: torch.Tensor
    history_prediction: torch.Tensor
    history_target: torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", default="fixtures/semantic_alignment_episodes.yaml")
    parser.add_argument("--output", default="runs/history_reconstructive_snn_benchmark_lmstudio")
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
    parser.add_argument("--history-reconstruction-weight", type=float, default=1.0)
    parser.add_argument("--history-ranking-weight", type=float, default=1.0)
    parser.add_argument("--history-ranking-margin", type=float, default=0.05)
    parser.add_argument("--representation-separation-weight", type=float, default=0.25)
    parser.add_argument("--representation-separation-margin", type=float, default=0.10)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 13, 21, 42, 100])
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def build_model(*, text_dim: int, num_neurons: int, seed: int, device: torch.device) -> HistoryReconstructiveBundle:
    bundle = build_bundle(text_dim=text_dim, num_neurons=num_neurons, seed=seed, device=device)
    history_predictor = NextEventPredictor(latent_dim=64, hidden_dim=128, embedding_dim=text_dim).to(device)
    return HistoryReconstructiveBundle(base=bundle, history_predictor=history_predictor)


def parameters_for(model: HistoryReconstructiveBundle) -> list[torch.nn.Parameter]:
    return (
        list(model.base.event_encoder.parameters())
        + list(model.base.snn.parameters())
        + list(model.base.trace_encoder.parameters())
        + list(model.base.predictor.parameters())
        + list(model.history_predictor.parameters())
    )


def set_mode(model: HistoryReconstructiveBundle, *, training: bool) -> None:
    modules = [
        model.base.event_encoder,
        model.base.snn,
        model.base.trace_encoder,
        model.base.predictor,
        model.history_predictor,
    ]
    for module in modules:
        module.train(training)


def state_dict_for(model: HistoryReconstructiveBundle) -> dict[str, Any]:
    return {
        "event_encoder": model.base.event_encoder.state_dict(),
        "snn": model.base.snn.state_dict(),
        "trace_encoder": model.base.trace_encoder.state_dict(),
        "predictor": model.base.predictor.state_dict(),
        "history_predictor": model.history_predictor.state_dict(),
    }


def load_state_dict_for(model: HistoryReconstructiveBundle, checkpoint: dict[str, Any]) -> None:
    model.base.event_encoder.load_state_dict(checkpoint["event_encoder"])
    model.base.snn.load_state_dict(checkpoint["snn"])
    model.base.trace_encoder.load_state_dict(checkpoint["trace_encoder"])
    model.base.predictor.load_state_dict(checkpoint["predictor"])
    model.history_predictor.load_state_dict(checkpoint["history_predictor"])


def history_target(*, episode, step_index: int, text_encoder, device: torch.device) -> torch.Tensor:
    if step_index <= 0:
        raise ValueError("history reconstruction requires step_index > 0")
    embeddings = text_encoder.encode([event.text for event in episode.events[:step_index]]).to(device)
    return F.normalize(embeddings.mean(dim=0, keepdim=True), dim=-1)


def predict(*, model: HistoryReconstructiveBundle, episode, step_index: int, text_encoder, args, device: torch.device) -> PredictionOutput:
    output = base.run_snn(
        bundle=model.base,
        episode=episode,
        step_index=step_index,
        text_encoder=text_encoder,
        event_ticks=args.event_ticks,
        stimulation_ticks=args.stimulation_ticks,
        device=device,
    )
    return PredictionOutput(
        prediction=output.prediction,
        target=output.target,
        representation=output.representation,
        regularization=output.regularization,
        history_prediction=model.history_predictor(output.representation),
        history_target=history_target(episode=episode, step_index=step_index, text_encoder=text_encoder, device=device),
    )


def history_ranking_loss(*, left: PredictionOutput, right: PredictionOutput, margin: float) -> torch.Tensor:
    left_own = F.cosine_similarity(left.history_prediction, left.history_target, dim=-1)
    left_other = F.cosine_similarity(left.history_prediction, right.history_target, dim=-1)
    right_own = F.cosine_similarity(right.history_prediction, right.history_target, dim=-1)
    right_other = F.cosine_similarity(right.history_prediction, left.history_target, dim=-1)
    left_loss = torch.relu(torch.as_tensor(margin, device=left_own.device) - left_own + left_other)
    right_loss = torch.relu(torch.as_tensor(margin, device=right_own.device) - right_own + right_other)
    return (left_loss + right_loss).mean() / 2.0


def pair_objective(*, left: PredictionOutput, right: PredictionOutput, args) -> tuple[torch.Tensor, dict[str, float]]:
    next_loss = (
        (1.0 - F.cosine_similarity(left.prediction, left.target, dim=-1)).mean()
        + (1.0 - F.cosine_similarity(right.prediction, right.target, dim=-1)).mean()
    ) / 2.0
    next_ranking = context_ranking_loss(
        left_prediction=left.prediction,
        left_target=left.target,
        right_prediction=right.prediction,
        right_target=right.target,
        margin=args.context_margin,
    )
    history_reconstruction = (
        (1.0 - F.cosine_similarity(left.history_prediction, left.history_target, dim=-1)).mean()
        + (1.0 - F.cosine_similarity(right.history_prediction, right.history_target, dim=-1)).mean()
    ) / 2.0
    history_ranking = history_ranking_loss(left=left, right=right, margin=args.history_ranking_margin)
    representation_distance = cosine_distance(left.representation, right.representation)
    representation_separation = torch.relu(
        torch.as_tensor(args.representation_separation_margin, device=representation_distance.device)
        - representation_distance
    )
    regularization = (left.regularization + right.regularization) / 2.0
    total = (
        next_loss
        + args.context_weight * next_ranking
        + args.history_reconstruction_weight * history_reconstruction
        + args.history_ranking_weight * history_ranking
        + args.representation_separation_weight * representation_separation
        + regularization
    )
    margin_value = context_margin(
        left_prediction=left.prediction,
        left_target=left.target,
        right_prediction=right.prediction,
        right_target=right.target,
    )
    metrics = {
        "total": float(total.detach()),
        "next_event": float(next_loss.detach()),
        "context_ranking": float(next_ranking.detach()),
        "context_margin": float(margin_value.detach()),
        "history_reconstruction": float(history_reconstruction.detach()),
        "history_ranking": float(history_ranking.detach()),
        "representation_distance": float(representation_distance.detach()),
        "representation_separation": float(representation_separation.detach()),
        "regularization": float(regularization.detach()),
    }
    return total, metrics


def aggregate(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = [
        "total",
        "next_event",
        "context_ranking",
        "context_margin",
        "history_reconstruction",
        "history_ranking",
        "representation_distance",
        "representation_separation",
        "regularization",
    ]
    return {key: sum(row[key] for row in rows) / max(1, len(rows)) for key in keys}


def evaluate_objective(*, model: HistoryReconstructiveBundle, pairs, episode_by_id, text_encoder, args, device: torch.device) -> dict[str, float]:
    set_mode(model, training=False)
    rows: list[dict[str, float]] = []
    with torch.no_grad():
        for pair in pairs:
            left = predict(model=model, episode=episode_by_id[pair.left_episode_id], step_index=pair.step_index, text_encoder=text_encoder, args=args, device=device)
            right = predict(model=model, episode=episode_by_id[pair.right_episode_id], step_index=pair.step_index, text_encoder=text_encoder, args=args, device=device)
            _, metrics = pair_objective(left=left, right=right, args=args)
            rows.append(metrics)
    return aggregate(rows)


def semantic_metrics(*, model: HistoryReconstructiveBundle, train_pairs, validation_pairs, episode_by_id, semantic_labels, text_encoder, args, device: torch.device) -> dict[str, Any]:
    return semantic.evaluate_model(
        model_type=MODEL_TYPE,
        model=model.base,
        train_pairs=train_pairs,
        validation_pairs=validation_pairs,
        episode_by_id=episode_by_id,
        semantic_labels=semantic_labels,
        text_encoder=text_encoder,
        args=args,
        device=device,
    )


def train_one(*, seed: int, train_pairs, validation_pairs, episode_by_id, semantic_labels, text_encoder, args, device: torch.device, output: Path, logger) -> dict[str, Any]:
    model = build_model(text_dim=text_encoder.output_dim, num_neurons=args.num_neurons, seed=seed, device=device)
    parameters = parameters_for(model)
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=1e-4)
    model_output = output / f"seed_{seed}"
    model_output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = model_output / "best_checkpoint.pt"
    best_validation = float("inf")
    best_epoch = -1
    history: list[dict[str, float | int]] = []

    logger.log("model.start", "History-reconstructive SNN 학습을 시작한다.", seed=seed, parameter_count=sum(parameter.numel() for parameter in parameters))
    for epoch in range(args.epochs):
        set_mode(model, training=True)
        epoch_rows: list[dict[str, float]] = []
        for pair in train_pairs:
            optimizer.zero_grad()
            left = predict(model=model, episode=episode_by_id[pair.left_episode_id], step_index=pair.step_index, text_encoder=text_encoder, args=args, device=device)
            right = predict(model=model, episode=episode_by_id[pair.right_episode_id], step_index=pair.step_index, text_encoder=text_encoder, args=args, device=device)
            total, metrics = pair_objective(left=left, right=right, args=args)
            total.backward()
            nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
            optimizer.step()
            epoch_rows.append(metrics)
        train_metrics = aggregate(epoch_rows)
        validation_metrics = evaluate_objective(model=model, pairs=validation_pairs, episode_by_id=episode_by_id, text_encoder=text_encoder, args=args, device=device)
        row = {"epoch": epoch, **{f"train_{key}": value for key, value in train_metrics.items()}, **{f"validation_{key}": value for key, value in validation_metrics.items()}}
        history.append(row)
        logger.log("epoch.done", "History-reconstructive epoch를 마쳤다.", seed=seed, **row)
        if validation_metrics["total"] < best_validation:
            best_validation = validation_metrics["total"]
            best_epoch = epoch
            torch.save({"seed": seed, "model_type": MODEL_TYPE, "args": vars(args), "epoch": epoch, "validation": validation_metrics, **state_dict_for(model)}, checkpoint_path)

    pd.DataFrame(history).to_csv(model_output / "history.csv", index=False, encoding="utf-8-sig")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    load_state_dict_for(model, checkpoint)
    objective = evaluate_objective(model=model, pairs=validation_pairs, episode_by_id=episode_by_id, text_encoder=text_encoder, args=args, device=device)
    semantic_result = semantic_metrics(model=model, train_pairs=train_pairs, validation_pairs=validation_pairs, episode_by_id=episode_by_id, semantic_labels=semantic_labels, text_encoder=text_encoder, args=args, device=device)
    result = {"seed": seed, "model_type": MODEL_TYPE, "best_epoch": best_epoch, "best_validation_total": best_validation, **{f"objective_{key}": value for key, value in objective.items()}, **semantic_result}
    (model_output / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.log("model.done", "History-reconstructive SNN 평가를 마쳤다.", **result)
    return result


def main() -> None:
    args = parse_args()
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if not args.seeds:
        raise ValueError("--seeds must not be empty")
    non_negative = (
        "context_weight",
        "context_margin",
        "history_reconstruction_weight",
        "history_ranking_weight",
        "history_ranking_margin",
        "representation_separation_weight",
        "representation_separation_margin",
        "ridge_alpha",
    )
    for name in non_negative:
        if getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    logger = base.RunLogger(output_dir=output, verbose=not args.quiet)
    logger.section("history-reconstructive SNN benchmark")
    logger.log("config", "History-reconstructive benchmark 설정을 불러왔다.", **vars(args))

    device = torch.device(args.device)
    episodes = base.load_episodes(args.fixture)
    episode_by_id = {episode.episode_id: episode for episode in episodes}
    semantic_labels = semantic.load_semantic_labels(args.fixture)
    train_pairs = base.load_contrast_pairs(args.fixture, split="train")
    validation_pairs = base.load_contrast_pairs(args.fixture, split="validation")
    base.validate_contrast_pairs(episodes, train_pairs + validation_pairs)
    if not train_pairs or not validation_pairs:
        raise ValueError("fixture must contain train and validation contrast pairs")
    text_encoder = base.build_text_encoder(args, output)
    logger.log("embedding.ready", "Embedding encoder와 캐시가 준비됐다.", output_dim=text_encoder.output_dim)

    results = [
        train_one(seed=seed, train_pairs=train_pairs, validation_pairs=validation_pairs, episode_by_id=episode_by_id, semantic_labels=semantic_labels, text_encoder=text_encoder, args=args, device=device, output=output, logger=logger)
        for seed in args.seeds
    ]
    frame = pd.DataFrame(results)
    frame.to_csv(output / "by_seed_model.csv", index=False, encoding="utf-8-sig")
    numeric_columns = [column for column in frame.columns if column not in {"seed", "model_type"}]
    summary = frame.groupby("model_type")[numeric_columns].agg(["mean", "std", "min", "max"])
    summary.columns = ["_".join(column) for column in summary.columns]
    summary.reset_index().to_csv(output / "summary_by_model.csv", index=False, encoding="utf-8-sig")
    metadata = {
        "fixture": args.fixture,
        "encoder": args.encoder,
        "embedding_model": args.embedding_model,
        "seeds": args.seeds,
        "epochs": args.epochs,
        "num_neurons": args.num_neurons,
        "event_ticks": args.event_ticks,
        "stimulation_ticks": args.stimulation_ticks,
        "objective": {
            "semantic_labels_used_for_training": False,
            "next_event_prediction": True,
            "next_event_context_ranking": True,
            "prior_event_embedding_reconstruction": True,
            "prior_event_embedding_ranking": True,
            "latent_representation_pair_separation": True,
        },
        "interpretation_boundary": INTERPRETATION_BOUNDARY,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.log("benchmark.done", "History-reconstructive SNN benchmark를 마쳤다.", files=["run_log.jsonl", "embedding_cache.json", "by_seed_model.csv", "summary_by_model.csv", "metadata.json"])
    print(summary.to_string())


if __name__ == "__main__":
    main()
