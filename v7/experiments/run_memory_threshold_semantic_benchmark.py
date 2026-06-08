"""Train and evaluate neuron-memory-threshold SNN variants.

Semantic labels are used only by an evaluation-only ridge probe after training.
The SNN itself is trained with next-event prediction and contrastive context
ranking, matching the prior controlled benchmark.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import run_context_objective_benchmark as base
import run_trace_semantic_alignment_benchmark as semantic

from emonet_v7.context_objective import context_margin, context_ranking_loss
from emonet_v7.memory_threshold_bundle import MemoryThresholdModelBundle, build_memory_threshold_bundle
from emonet_v7.memory_threshold_training_window import (
    MemoryThresholdDifferentiableWindow,
    run_memory_threshold_differentiable_window,
)


MODEL_CONFIGS = {
    "snn_memory_readout_only": 0.0,
    "snn_memory_feedback": 0.05,
}
INTERPRETATION_BOUNDARY = (
    "This benchmark evaluates whether neuron-local accumulation and a separate memory threshold improve controlled semantic readability. "
    "It does not establish ground-truth emotions, biological fidelity, emergent clusters, or broad real-world generalization."
)


@dataclass
class PredictionOutput:
    prediction: torch.Tensor
    target: torch.Tensor
    latent: torch.Tensor
    regularization: torch.Tensor
    raw_representation: torch.Tensor
    final_memory_strength: torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", default="fixtures/semantic_alignment_episodes.yaml")
    parser.add_argument("--output", default="runs/memory_threshold_semantic_benchmark_lmstudio")
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
    parser.add_argument("--accumulation-decay", type=float, default=0.85)
    parser.add_argument("--memory-threshold", type=float, default=0.60)
    parser.add_argument("--memory-decay", type=float, default=0.98)
    parser.add_argument("--feedback-strength", type=float, default=0.05)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 13, 21, 42, 100])
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def temporal_stats(sequence: torch.Tensor) -> torch.Tensor:
    mean = sequence.mean(dim=1)
    std = sequence.std(dim=1, unbiased=False)
    final = sequence[:, -1, :]
    if sequence.shape[1] > 1:
        delta = (sequence[:, 1:, :] - sequence[:, :-1, :]).abs().mean(dim=1)
    else:
        delta = torch.zeros_like(final)
    return torch.cat([mean, std, final, delta], dim=-1)


def pool_memory_window(window: MemoryThresholdDifferentiableWindow, final_state) -> torch.Tensor:
    return torch.cat(
        [
            temporal_stats(window.spike),
            temporal_stats(window.membrane),
            temporal_stats(window.adaptation),
            temporal_stats(window.accumulation),
            temporal_stats(window.memory_strength),
            final_state.accumulation,
            final_state.memory_strength,
        ],
        dim=-1,
    )


def memory_regularization(window: MemoryThresholdDifferentiableWindow, final_state) -> torch.Tensor:
    firing_rate = (window.spike.mean() - 0.10) ** 2
    neuron_activity = window.spike.sum(dim=(0, 1))
    inactive_neuron = torch.exp(-neuron_activity).mean()
    membrane_excess = torch.relu(window.membrane.abs() - 5.0)
    memory_excess = torch.relu(final_state.memory_strength.abs() - 0.95)
    return (
        0.10 * firing_rate
        + 0.01 * inactive_neuron
        + 0.01 * (membrane_excess ** 2).mean()
        + 0.01 * (memory_excess ** 2).mean()
    )


def build_model(*, text_dim: int, num_neurons: int, seed: int, feedback_strength: float, args, device: torch.device):
    return build_memory_threshold_bundle(
        text_dim=text_dim,
        num_neurons=num_neurons,
        seed=seed,
        device=device,
        memory_feedback_strength=feedback_strength,
        accumulation_decay=args.accumulation_decay,
        memory_threshold=args.memory_threshold,
        memory_decay=args.memory_decay,
    )


def parameters_for(model: MemoryThresholdModelBundle) -> list[torch.nn.Parameter]:
    return (
        list(model.event_encoder.parameters())
        + list(model.snn.parameters())
        + list(model.trace_encoder.parameters())
        + list(model.predictor.parameters())
    )


def set_mode(model: MemoryThresholdModelBundle, *, training: bool) -> None:
    for module in (model.event_encoder, model.snn, model.trace_encoder, model.predictor):
        module.train(training)


def state_dict_for(model: MemoryThresholdModelBundle) -> dict[str, Any]:
    return {
        "event_encoder": model.event_encoder.state_dict(),
        "snn": model.snn.state_dict(),
        "trace_encoder": model.trace_encoder.state_dict(),
        "predictor": model.predictor.state_dict(),
    }


def load_state_dict_for(model: MemoryThresholdModelBundle, checkpoint: dict[str, Any]) -> None:
    model.event_encoder.load_state_dict(checkpoint["event_encoder"])
    model.snn.load_state_dict(checkpoint["snn"])
    model.trace_encoder.load_state_dict(checkpoint["trace_encoder"])
    model.predictor.load_state_dict(checkpoint["predictor"])


def sequence_for_condition(*, episode, step_index: int, condition: str, swapped_history_source=None):
    if condition == "real_history":
        return base.sequence_events(episode, step_index)
    if condition == "shuffled_history":
        if swapped_history_source is None:
            raise ValueError("shuffled_history requires swapped_history_source")
        return base.sequence_events(episode, step_index, swapped_history_source)
    if condition == "reset_history":
        return (episode.events[step_index],)
    raise ValueError(f"unsupported history condition: {condition}")


def run_sequence(
    *,
    model: MemoryThresholdModelBundle,
    episode,
    step_index: int,
    text_encoder,
    args,
    device: torch.device,
    condition: str = "real_history",
    swapped_history_source=None,
) -> PredictionOutput:
    state = model.snn.initial_state(batch_size=1, device=device)
    last_window = None
    for event in sequence_for_condition(
        episode=episode,
        step_index=step_index,
        condition=condition,
        swapped_history_source=swapped_history_source,
    ):
        embedding = text_encoder.encode([event.text]).to(device)
        current = model.event_encoder(embedding, [event])
        state, last_window = run_memory_threshold_differentiable_window(
            snn=model.snn,
            event_current=current,
            state=state,
            event_ticks=args.event_ticks,
            stimulation_ticks=args.stimulation_ticks,
        )
    if last_window is None:
        raise RuntimeError("memory-threshold SNN sequence must contain at least one event")
    latent = model.trace_encoder(
        last_window.spike,
        last_window.membrane,
        last_window.adaptation,
        last_window.accumulation,
        last_window.memory_strength,
        state.accumulation,
        state.memory_strength,
    )
    prediction = model.predictor(latent)
    target = text_encoder.encode([episode.events[step_index + 1].text]).to(device)
    return PredictionOutput(
        prediction=prediction,
        target=target,
        latent=latent,
        regularization=memory_regularization(last_window, state),
        raw_representation=pool_memory_window(last_window, state),
        final_memory_strength=state.memory_strength,
    )


def pair_objective(*, left: PredictionOutput, right: PredictionOutput, args) -> tuple[torch.Tensor, dict[str, float]]:
    next_loss = (
        (1.0 - F.cosine_similarity(left.prediction, left.target, dim=-1)).mean()
        + (1.0 - F.cosine_similarity(right.prediction, right.target, dim=-1)).mean()
    ) / 2.0
    ranking = context_ranking_loss(
        left_prediction=left.prediction,
        left_target=left.target,
        right_prediction=right.prediction,
        right_target=right.target,
        margin=args.context_margin,
    )
    regularization = (left.regularization + right.regularization) / 2.0
    total = next_loss + args.context_weight * ranking + regularization
    margin_value = context_margin(
        left_prediction=left.prediction,
        left_target=left.target,
        right_prediction=right.prediction,
        right_target=right.target,
    )
    return total, {
        "total": float(total.detach()),
        "next_event": float(next_loss.detach()),
        "context_ranking": float(ranking.detach()),
        "context_margin": float(margin_value.detach()),
        "regularization": float(regularization.detach()),
        "memory_strength_mean_abs": float(((left.final_memory_strength.abs().mean() + right.final_memory_strength.abs().mean()) / 2.0).detach()),
    }


def aggregate(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = rows[0].keys()
    return {key: sum(row[key] for row in rows) / len(rows) for key in keys}


def evaluate_objective(*, model, pairs, episode_by_id, text_encoder, args, device) -> dict[str, float]:
    set_mode(model, training=False)
    rows: list[dict[str, float]] = []
    with torch.no_grad():
        for pair in pairs:
            left = run_sequence(model=model, episode=episode_by_id[pair.left_episode_id], step_index=pair.step_index, text_encoder=text_encoder, args=args, device=device)
            right = run_sequence(model=model, episode=episode_by_id[pair.right_episode_id], step_index=pair.step_index, text_encoder=text_encoder, args=args, device=device)
            _, metrics = pair_objective(left=left, right=right, args=args)
            rows.append(metrics)
    return aggregate(rows)


def collect_semantic_rows(*, model, pairs, episode_by_id, semantic_labels, text_encoder, args, device, condition: str):
    rows: list[dict[str, Any]] = []
    set_mode(model, training=False)
    with torch.no_grad():
        for pair in pairs:
            targeted_axis = semantic.targeted_axis_for(pair.relation)
            for side, episode_id, swapped_id in (
                ("left", pair.left_episode_id, pair.right_episode_id),
                ("right", pair.right_episode_id, pair.left_episode_id),
            ):
                episode = episode_by_id[episode_id]
                output = run_sequence(
                    model=model,
                    episode=episode,
                    step_index=pair.step_index,
                    text_encoder=text_encoder,
                    args=args,
                    device=device,
                    condition=condition,
                    swapped_history_source=episode_by_id[swapped_id],
                )
                rows.append(
                    {
                        "episode_id": episode_id,
                        "relation": pair.relation,
                        "side": side,
                        "targeted_axis": targeted_axis,
                        "trace": output.raw_representation.detach().cpu().reshape(-1).numpy(),
                        "current_text": semantic.current_text_vector(episode=episode, step_index=pair.step_index, text_encoder=text_encoder),
                        "label": semantic_labels[episode_id],
                    }
                )
    return rows


def semantic_metrics(*, model, train_pairs, validation_pairs, episode_by_id, semantic_labels, text_encoder, args, device) -> dict[str, Any]:
    train_real = collect_semantic_rows(model=model, pairs=train_pairs, episode_by_id=episode_by_id, semantic_labels=semantic_labels, text_encoder=text_encoder, args=args, device=device, condition="real_history")
    validation_real = collect_semantic_rows(model=model, pairs=validation_pairs, episode_by_id=episode_by_id, semantic_labels=semantic_labels, text_encoder=text_encoder, args=args, device=device, condition="real_history")
    validation_shuffled = collect_semantic_rows(model=model, pairs=validation_pairs, episode_by_id=episode_by_id, semantic_labels=semantic_labels, text_encoder=text_encoder, args=args, device=device, condition="shuffled_history")
    validation_reset = collect_semantic_rows(model=model, pairs=validation_pairs, episode_by_id=episode_by_id, semantic_labels=semantic_labels, text_encoder=text_encoder, args=args, device=device, condition="reset_history")
    train_ids = {row["episode_id"] for row in train_real}
    validation_ids = {row["episode_id"] for row in validation_real}
    overlap = sorted(train_ids.intersection(validation_ids))
    if overlap:
        raise ValueError(f"semantic probe group leakage detected: {overlap}")

    trace_probe = semantic.fit_probe(train_real, feature_key="trace", alpha=args.ridge_alpha)
    text_probe = semantic.fit_probe(train_real, feature_key="current_text", alpha=args.ridge_alpha)
    real_predictions = semantic.predict_rows(trace_probe, validation_real, feature_key="trace")
    shuffled_predictions = semantic.predict_rows(trace_probe, validation_shuffled, feature_key="trace")
    reset_predictions = semantic.predict_rows(trace_probe, validation_reset, feature_key="trace")
    text_predictions = semantic.predict_rows(text_probe, validation_real, feature_key="current_text")
    train_labels = np.stack([row["label"] for row in train_real])
    constant_vector = train_labels.mean(axis=0)
    constant_predictions = np.repeat(constant_vector[None, :], len(validation_real), axis=0)

    real_mae = semantic.targeted_mae(validation_real, real_predictions)
    shuffled_mae = semantic.targeted_mae(validation_shuffled, shuffled_predictions)
    reset_mae = semantic.targeted_mae(validation_reset, reset_predictions)
    text_mae = semantic.targeted_mae(validation_real, text_predictions)
    constant_mae = semantic.targeted_mae(validation_real, constant_predictions)
    result: dict[str, Any] = {
        "trace_dim": int(train_real[0]["trace"].shape[0]),
        "train_episode_count": len(train_ids),
        "validation_episode_count": len(validation_ids),
        "group_overlap_count": 0,
        "real_targeted_mae": real_mae,
        "shuffled_targeted_mae": shuffled_mae,
        "reset_targeted_mae": reset_mae,
        "current_text_baseline_targeted_mae": text_mae,
        "constant_baseline_targeted_mae": constant_mae,
        "real_minus_constant_mae_improvement": constant_mae - real_mae,
        "real_minus_text_baseline_mae_improvement": text_mae - real_mae,
        "shuffled_history_mae_degradation": shuffled_mae - real_mae,
        "reset_history_mae_degradation": reset_mae - real_mae,
        "real_direction_accuracy": semantic.targeted_direction_accuracy(validation_real, real_predictions),
        "shuffled_direction_accuracy": semantic.targeted_direction_accuracy(validation_shuffled, shuffled_predictions),
        "reset_direction_accuracy": semantic.targeted_direction_accuracy(validation_reset, reset_predictions),
        "current_text_baseline_direction_accuracy": semantic.targeted_direction_accuracy(validation_real, text_predictions),
        "real_pair_order_accuracy": semantic.pair_order_accuracy(validation_real, real_predictions),
        "shuffled_pair_order_accuracy": semantic.pair_order_accuracy(validation_shuffled, shuffled_predictions),
        "reset_pair_order_accuracy": semantic.pair_order_accuracy(validation_reset, reset_predictions),
        "current_text_baseline_pair_order_accuracy": semantic.pair_order_accuracy(validation_real, text_predictions),
    }
    result.update(semantic.axis_metrics(validation_real, real_predictions, "real"))
    result.update(semantic.axis_metrics(validation_shuffled, shuffled_predictions, "shuffled"))
    result.update(semantic.axis_metrics(validation_reset, reset_predictions, "reset"))
    return result


def train_one(*, model_type: str, feedback_strength: float, seed: int, train_pairs, validation_pairs, episode_by_id, semantic_labels, text_encoder, args, device, output: Path, logger) -> dict[str, Any]:
    model = build_model(text_dim=text_encoder.output_dim, num_neurons=args.num_neurons, seed=seed, feedback_strength=feedback_strength, args=args, device=device)
    parameters = parameters_for(model)
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=1e-4)
    model_output = output / f"seed_{seed}" / model_type
    model_output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = model_output / "best_checkpoint.pt"
    best_validation = float("inf")
    best_epoch = -1
    history: list[dict[str, float | int]] = []

    logger.log("model.start", "Memory-threshold SNN 학습을 시작한다.", model_type=model_type, seed=seed, feedback_strength=feedback_strength, parameter_count=sum(parameter.numel() for parameter in parameters))
    for epoch in range(args.epochs):
        set_mode(model, training=True)
        epoch_rows: list[dict[str, float]] = []
        for pair in train_pairs:
            optimizer.zero_grad()
            left = run_sequence(model=model, episode=episode_by_id[pair.left_episode_id], step_index=pair.step_index, text_encoder=text_encoder, args=args, device=device)
            right = run_sequence(model=model, episode=episode_by_id[pair.right_episode_id], step_index=pair.step_index, text_encoder=text_encoder, args=args, device=device)
            total, metrics = pair_objective(left=left, right=right, args=args)
            total.backward()
            nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
            optimizer.step()
            epoch_rows.append(metrics)
        train_metrics = aggregate(epoch_rows)
        validation_metrics = evaluate_objective(model=model, pairs=validation_pairs, episode_by_id=episode_by_id, text_encoder=text_encoder, args=args, device=device)
        row = {"epoch": epoch, **{f"train_{key}": value for key, value in train_metrics.items()}, **{f"validation_{key}": value for key, value in validation_metrics.items()}}
        history.append(row)
        logger.log("epoch.done", "Memory-threshold epoch를 마쳤다.", model_type=model_type, seed=seed, **row)
        if validation_metrics["total"] < best_validation:
            best_validation = validation_metrics["total"]
            best_epoch = epoch
            torch.save({"model_type": model_type, "seed": seed, "feedback_strength": feedback_strength, "args": vars(args), "epoch": epoch, "validation": validation_metrics, **state_dict_for(model)}, checkpoint_path)

    pd.DataFrame(history).to_csv(model_output / "history.csv", index=False, encoding="utf-8-sig")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    load_state_dict_for(model, checkpoint)
    objective = evaluate_objective(model=model, pairs=validation_pairs, episode_by_id=episode_by_id, text_encoder=text_encoder, args=args, device=device)
    semantics = semantic_metrics(model=model, train_pairs=train_pairs, validation_pairs=validation_pairs, episode_by_id=episode_by_id, semantic_labels=semantic_labels, text_encoder=text_encoder, args=args, device=device)
    result = {"seed": seed, "model_type": model_type, "feedback_strength": feedback_strength, "best_epoch": best_epoch, "best_validation_total": best_validation, **{f"objective_{key}": value for key, value in objective.items()}, **semantics}
    (model_output / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.log("model.done", "Memory-threshold SNN 평가를 마쳤다.", **result)
    return result


def main() -> None:
    args = parse_args()
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if not args.seeds:
        raise ValueError("--seeds must not be empty")
    if not 0 <= args.stimulation_ticks <= args.event_ticks:
        raise ValueError("--stimulation-ticks must be between 0 and --event-ticks")
    if args.feedback_strength < 0:
        raise ValueError("--feedback-strength must be non-negative")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    logger = base.RunLogger(output_dir=output, verbose=not args.quiet)
    logger.section("memory-threshold semantic benchmark")
    logger.log("config", "Memory-threshold semantic benchmark 설정을 불러왔다.", **vars(args))

    device = torch.device(args.device)
    episodes = base.load_episodes(args.fixture)
    episode_by_id = {episode.episode_id: episode for episode in episodes}
    semantic_labels = semantic.load_semantic_labels(args.fixture)
    train_pairs = base.load_contrast_pairs(args.fixture, split="train")
    validation_pairs = base.load_contrast_pairs(args.fixture, split="validation")
    base.validate_contrast_pairs(episodes, train_pairs + validation_pairs)
    text_encoder = base.build_text_encoder(args, output)
    logger.log("embedding.ready", "Embedding encoder와 캐시가 준비됐다.", output_dim=text_encoder.output_dim)

    model_configs = {
        "snn_memory_readout_only": 0.0,
        "snn_memory_feedback": args.feedback_strength,
    }
    rows = [
        train_one(model_type=model_type, feedback_strength=feedback_strength, seed=seed, train_pairs=train_pairs, validation_pairs=validation_pairs, episode_by_id=episode_by_id, semantic_labels=semantic_labels, text_encoder=text_encoder, args=args, device=device, output=output, logger=logger)
        for seed in args.seeds
        for model_type, feedback_strength in model_configs.items()
    ]
    frame = pd.DataFrame(rows)
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
        "model_configs": model_configs,
        "semantic_labels_used_for_training": False,
        "raw_probe_representation": "concat temporal stats for spike, membrane, adaptation, accumulation, memory_strength plus final accumulation and memory_strength",
        "interpretation_boundary": INTERPRETATION_BOUNDARY,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.log("benchmark.done", "Memory-threshold semantic benchmark를 마쳤다.", files=["run_log.jsonl", "embedding_cache.json", "by_seed_model.csv", "summary_by_model.csv", "metadata.json"])
    print(summary.to_string())


if __name__ == "__main__":
    main()
