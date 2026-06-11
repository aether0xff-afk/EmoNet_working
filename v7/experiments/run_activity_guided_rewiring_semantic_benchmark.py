"""Train a memory-threshold SNN with activity-guided structural rewiring.

The experiment keeps the verified neuron-memory configuration fixed and rewires
only sparse adjacency. Rewiring uses train-episode neuron-local memory profiles;
semantic labels never affect SNN training or topology changes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

import run_context_objective_benchmark as base
import run_memory_threshold_semantic_benchmark as memory
import run_trace_semantic_alignment_benchmark as semantic

from emonet_v7.activity_guided_rewiring import (
    reset_optimizer_state_for_changed_edges,
    rewire_from_memory_profiles,
)


MODEL_TYPE = "snn_memory_feedback_rewired"
INTERPRETATION_BOUNDARY = (
    "This benchmark tests an activity-guided structural-plasticity ablation under a controlled fixture. "
    "It does not establish emotional ground truth, biological fidelity, final rewiring rules, or broad real-world generalization."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", default="fixtures/semantic_alignment_episodes.yaml")
    parser.add_argument("--output", default="runs/activity_guided_rewiring_semantic_benchmark_lmstudio")
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
    parser.add_argument("--feedback-strength", type=float, default=0.05)
    parser.add_argument("--memory-threshold", type=float, default=0.50)
    parser.add_argument("--accumulation-decay", type=float, default=0.85)
    parser.add_argument("--memory-decay", type=float, default=0.98)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--rewiring-start-epoch", type=int, default=10)
    parser.add_argument("--rewiring-interval", type=int, default=5)
    parser.add_argument("--rewiring-fraction", type=float, default=0.01)
    parser.add_argument("--new-weight-scale", type=float, default=0.05)
    parser.add_argument("--min-clusters", type=int, default=2)
    parser.add_argument("--max-clusters", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 13, 21, 42, 100])
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def unique_episode_steps(pairs) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for pair in pairs:
        mapping[pair.left_episode_id] = pair.step_index
        mapping[pair.right_episode_id] = pair.step_index
    return mapping


def collect_train_memory_profiles(*, model, episode_steps: dict[str, int], episode_by_id, text_encoder, args, device) -> np.ndarray:
    """Collect train-only final memory strengths without semantic labels."""

    profiles: list[np.ndarray] = []
    memory.set_mode(model, training=False)
    with torch.no_grad():
        for episode_id in sorted(episode_steps):
            output = memory.run_sequence(
                model=model,
                episode=episode_by_id[episode_id],
                step_index=episode_steps[episode_id],
                condition="real_history",
                swapped_history_source=None,
                text_encoder=text_encoder,
                args=args,
                device=device,
            )
            profiles.append(output.final_memory_strength.detach().cpu().reshape(-1).numpy())
    return np.stack(profiles)


def should_rewire(*, epoch: int, start_epoch: int, interval: int) -> bool:
    """Return whether a completed 1-indexed epoch reaches the rewiring schedule."""

    completed_epoch = epoch + 1
    return completed_epoch >= start_epoch and (completed_epoch - start_epoch) % interval == 0


def train_one(*, seed: int, train_pairs, validation_pairs, episode_by_id, semantic_labels, text_encoder, args, device, output: Path, logger) -> dict[str, Any]:
    model = memory.build_model(
        text_dim=text_encoder.output_dim,
        num_neurons=args.num_neurons,
        seed=seed,
        feedback_strength=args.feedback_strength,
        args=args,
        device=device,
    )
    parameters = memory.parameters_for(model)
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=1e-4)
    train_episode_steps = unique_episode_steps(train_pairs)
    model_output = output / f"seed_{seed}" / MODEL_TYPE
    model_output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = model_output / "best_checkpoint.pt"
    best_validation = float("inf")
    best_epoch = -1
    history: list[dict[str, Any]] = []
    rewiring_history: list[dict[str, Any]] = []

    logger.log("model.start", "Activity-guided rewiring SNN 학습을 시작한다.", model_type=MODEL_TYPE, seed=seed, parameter_count=sum(parameter.numel() for parameter in parameters))
    for epoch in range(args.epochs):
        memory.set_mode(model, training=True)
        epoch_rows: list[dict[str, float]] = []
        for pair in train_pairs:
            optimizer.zero_grad()
            left = memory.run_sequence(model=model, episode=episode_by_id[pair.left_episode_id], step_index=pair.step_index, text_encoder=text_encoder, args=args, device=device)
            right = memory.run_sequence(model=model, episode=episode_by_id[pair.right_episode_id], step_index=pair.step_index, text_encoder=text_encoder, args=args, device=device)
            total, metrics = memory.pair_objective(left=left, right=right, args=args)
            total.backward()
            nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
            optimizer.step()
            epoch_rows.append(metrics)

        rewiring_report = None
        if should_rewire(epoch=epoch, start_epoch=args.rewiring_start_epoch, interval=args.rewiring_interval):
            profiles = collect_train_memory_profiles(model=model, episode_steps=train_episode_steps, episode_by_id=episode_by_id, text_encoder=text_encoder, args=args, device=device)
            report, changed_mask = rewire_from_memory_profiles(
                snn=model.snn,
                response_by_episode=profiles,
                fraction=args.rewiring_fraction,
                seed=seed * 1000 + epoch,
                min_clusters=args.min_clusters,
                max_clusters=args.max_clusters,
                new_weight_scale=args.new_weight_scale,
            )
            reset_optimizer_state_for_changed_edges(
                optimizer=optimizer,
                parameter=model.snn.recurrent_weight,
                changed_mask=changed_mask,
            )
            rewiring_report = report.to_dict()
            rewiring_report["epoch"] = epoch
            rewiring_report["changed_weight_entry_count"] = int(changed_mask.sum().item())
            rewiring_history.append(rewiring_report)
            logger.log("rewiring.done", "Activity-guided adjacency rewiring을 수행했다.", **rewiring_report)

        train_metrics = memory.aggregate(epoch_rows)
        validation_metrics = memory.evaluate_objective(model=model, pairs=validation_pairs, episode_by_id=episode_by_id, text_encoder=text_encoder, args=args, device=device)
        row: dict[str, Any] = {
            "epoch": epoch,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"validation_{key}": value for key, value in validation_metrics.items()},
            "rewired_edge_count": int(rewiring_report["rewired_edge_count"]) if rewiring_report else 0,
        }
        history.append(row)
        logger.log("epoch.done", "Rewiring SNN epoch를 마쳤다.", seed=seed, **row)
        if validation_metrics["total"] < best_validation:
            best_validation = validation_metrics["total"]
            best_epoch = epoch
            torch.save(
                {
                    "model_type": MODEL_TYPE,
                    "seed": seed,
                    "args": vars(args),
                    "epoch": epoch,
                    "validation": validation_metrics,
                    "rewiring_history": rewiring_history,
                    **memory.state_dict_for(model),
                },
                checkpoint_path,
            )

    pd.DataFrame(history).to_csv(model_output / "history.csv", index=False, encoding="utf-8-sig")
    (model_output / "rewiring_history.json").write_text(json.dumps(rewiring_history, ensure_ascii=False, indent=2), encoding="utf-8")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    memory.load_state_dict_for(model, checkpoint)
    objective = memory.evaluate_objective(model=model, pairs=validation_pairs, episode_by_id=episode_by_id, text_encoder=text_encoder, args=args, device=device)
    semantics = memory.semantic_metrics(model=model, train_pairs=train_pairs, validation_pairs=validation_pairs, episode_by_id=episode_by_id, semantic_labels=semantic_labels, text_encoder=text_encoder, args=args, device=device)
    result = {
        "seed": seed,
        "model_type": MODEL_TYPE,
        "best_epoch": best_epoch,
        "best_validation_total": best_validation,
        "rewiring_event_count": len(checkpoint.get("rewiring_history", [])),
        "rewired_edges_total": int(sum(item["rewired_edge_count"] for item in checkpoint.get("rewiring_history", []))),
        **{f"objective_{key}": value for key, value in objective.items()},
        **semantics,
    }
    (model_output / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.log("model.done", "Activity-guided rewiring SNN 평가를 마쳤다.", **result)
    return result


def main() -> None:
    args = parse_args()
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if not args.seeds:
        raise ValueError("--seeds must not be empty")
    if args.rewiring_start_epoch <= 0:
        raise ValueError("--rewiring-start-epoch must be positive")
    if args.rewiring_interval <= 0:
        raise ValueError("--rewiring-interval must be positive")
    if not 0.0 <= args.rewiring_fraction <= 1.0:
        raise ValueError("--rewiring-fraction must remain in [0, 1]")
    if args.new_weight_scale <= 0.0:
        raise ValueError("--new-weight-scale must be positive")
    if not 0 <= args.stimulation_ticks <= args.event_ticks:
        raise ValueError("--stimulation-ticks must be between 0 and --event-ticks")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    logger = base.RunLogger(output_dir=output, verbose=not args.quiet)
    logger.section("activity-guided rewiring semantic benchmark")
    logger.log("config", "Activity-guided rewiring benchmark 설정을 불러왔다.", **vars(args))

    device = torch.device(args.device)
    episodes = base.load_episodes(args.fixture)
    episode_by_id = {episode.episode_id: episode for episode in episodes}
    semantic_labels = semantic.load_semantic_labels(args.fixture)
    train_pairs = base.load_contrast_pairs(args.fixture, split="train")
    validation_pairs = base.load_contrast_pairs(args.fixture, split="validation")
    base.validate_contrast_pairs(episodes, train_pairs + validation_pairs)
    text_encoder = base.build_text_encoder(args, output)
    logger.log("embedding.ready", "Embedding encoder와 캐시가 준비됐다.", output_dim=text_encoder.output_dim)

    rows = [
        train_one(seed=seed, train_pairs=train_pairs, validation_pairs=validation_pairs, episode_by_id=episode_by_id, semantic_labels=semantic_labels, text_encoder=text_encoder, args=args, device=device, output=output, logger=logger)
        for seed in args.seeds
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
        "epochs": args.epochs,
        "feedback_strength": args.feedback_strength,
        "memory_threshold": args.memory_threshold,
        "accumulation_decay": args.accumulation_decay,
        "memory_decay": args.memory_decay,
        "rewiring_start_epoch": args.rewiring_start_epoch,
        "rewiring_interval": args.rewiring_interval,
        "rewiring_fraction": args.rewiring_fraction,
        "new_weight_scale": args.new_weight_scale,
        "optimizer_state_reset_scope": "changed recurrent-weight entries only",
        "semantic_labels_used_for_training": False,
        "semantic_labels_used_for_rewiring": False,
        "rewiring_rule": "discover functional communities from train memory-strength profiles; prune weak inter-community edges; add high-coactivity intra-community edges; preserve directed edge budget",
        "interpretation_boundary": INTERPRETATION_BOUNDARY,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.log("benchmark.done", "Activity-guided rewiring benchmark를 마쳤다.", files=["run_log.jsonl", "embedding_cache.json", "by_seed_model.csv", "summary_by_model.csv", "metadata.json"])
    print(summary.to_string())


if __name__ == "__main__":
    main()
