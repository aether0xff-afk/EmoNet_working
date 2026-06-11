"""Run an OFAT stability sweep for activity-guided adjacency rewiring.

The sweep searches for a topology-plasticity regime that preserves semantic
readability before any cluster claim is evaluated. Semantic labels remain
probe-only and never affect SNN learning or rewiring.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch

import run_activity_guided_rewiring_semantic_benchmark as rewiring
import run_context_objective_benchmark as base
import run_trace_semantic_alignment_benchmark as semantic


BASELINE_CONFIG = {
    "rewiring_fraction": 0.01,
    "rewiring_start_epoch": 10,
    "rewiring_interval": 5,
}
SWEEP_VALUES = {
    "rewiring_fraction": (0.0, 0.0025, 0.005, 0.01, 0.02),
    "rewiring_start_epoch": (5, 10, 15),
    "rewiring_interval": (5, 10),
}
INTERPRETATION_BOUNDARY = (
    "This OFAT sweep searches for a semantic-preserving activity-guided rewiring regime under a controlled fixture. "
    "It does not establish emergent communities, final rewiring rules, emotional ground truth, or biological fidelity."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", default="fixtures/semantic_alignment_episodes.yaml")
    parser.add_argument("--output", default="runs/activity_guided_rewiring_stability_sweep_lmstudio")
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
    parser.add_argument("--new-weight-scale", type=float, default=0.05)
    parser.add_argument("--min-clusters", type=int, default=2)
    parser.add_argument("--max-clusters", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 13, 21, 42, 100])
    parser.add_argument("--families", nargs="+", choices=sorted(SWEEP_VALUES), default=sorted(SWEEP_VALUES))
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def config_key(config: dict[str, Any]) -> str:
    return (
        f"fraction_{float(config['rewiring_fraction']):.4f}__"
        f"start_{int(config['rewiring_start_epoch']):02d}__"
        f"interval_{int(config['rewiring_interval']):02d}"
    )


def build_trials(families: list[str]) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    unique: dict[str, dict[str, Any]] = {}
    references: list[dict[str, Any]] = []
    for family in families:
        for value in SWEEP_VALUES[family]:
            config = dict(BASELINE_CONFIG)
            config[family] = value
            key = config_key(config)
            unique.setdefault(key, config)
            references.append({"family": family, "value": value, "config_key": key, **config})
    return unique, references


def trial_args(args: argparse.Namespace, config: dict[str, Any]) -> argparse.Namespace:
    trial = deepcopy(args)
    trial.rewiring_fraction = float(config["rewiring_fraction"])
    trial.rewiring_start_epoch = int(config["rewiring_start_epoch"])
    trial.rewiring_interval = int(config["rewiring_interval"])
    return trial


def main() -> None:
    args = parse_args()
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if not args.seeds:
        raise ValueError("--seeds must not be empty")
    if not 0 <= args.stimulation_ticks <= args.event_ticks:
        raise ValueError("--stimulation-ticks must be between 0 and --event-ticks")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    logger = base.RunLogger(output_dir=output, verbose=not args.quiet)
    logger.section("activity-guided rewiring stability OFAT sweep")
    logger.log("config", "Rewiring stability OFAT sweep 설정을 불러왔다.", **vars(args))

    device = torch.device(args.device)
    episodes = base.load_episodes(args.fixture)
    episode_by_id = {episode.episode_id: episode for episode in episodes}
    semantic_labels = semantic.load_semantic_labels(args.fixture)
    train_pairs = base.load_contrast_pairs(args.fixture, split="train")
    validation_pairs = base.load_contrast_pairs(args.fixture, split="validation")
    base.validate_contrast_pairs(episodes, train_pairs + validation_pairs)
    text_encoder = base.build_text_encoder(args, output)
    logger.log("embedding.ready", "Embedding encoder와 sweep 공용 캐시가 준비됐다.", output_dim=text_encoder.output_dim)

    unique_configs, references = build_trials(args.families)
    rows: list[dict[str, Any]] = []
    for key, config in unique_configs.items():
        logger.section(key)
        current_args = trial_args(args, config)
        trial_output = output / "trials" / key
        for seed in args.seeds:
            result = rewiring.train_one(
                seed=seed,
                train_pairs=train_pairs,
                validation_pairs=validation_pairs,
                episode_by_id=episode_by_id,
                semantic_labels=semantic_labels,
                text_encoder=text_encoder,
                args=current_args,
                device=device,
                output=trial_output,
                logger=logger,
            )
            rows.append({"config_key": key, **config, **result})

    frame = pd.DataFrame(rows)
    frame.to_csv(output / "by_seed_config.csv", index=False, encoding="utf-8-sig")
    numeric_columns = [column for column in frame.columns if column not in {"config_key", "seed", "model_type"}]
    summary = frame.groupby("config_key")[numeric_columns].agg(["mean", "std", "min", "max"])
    summary.columns = ["_".join(column) for column in summary.columns]
    summary.reset_index().to_csv(output / "summary_by_config.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(references).to_csv(output / "sweep_references.csv", index=False, encoding="utf-8-sig")
    metadata = {
        "fixture": args.fixture,
        "encoder": args.encoder,
        "embedding_model": args.embedding_model,
        "seeds": args.seeds,
        "epochs": args.epochs,
        "families": args.families,
        "baseline_config": BASELINE_CONFIG,
        "sweep_values": {key: list(values) for key, values in SWEEP_VALUES.items()},
        "unique_config_count": len(unique_configs),
        "reference_count": len(references),
        "optimizer_state_reset_scope": "changed recurrent-weight entries only",
        "semantic_labels_used_for_training": False,
        "semantic_labels_used_for_rewiring": False,
        "interpretation_boundary": INTERPRETATION_BOUNDARY,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.log("benchmark.done", "Rewiring stability OFAT sweep를 마쳤다.", unique_config_count=len(unique_configs), files=["run_log.jsonl", "embedding_cache.json", "by_seed_config.csv", "summary_by_config.csv", "sweep_references.csv", "metadata.json"])
    print(summary.to_string())


if __name__ == "__main__":
    main()
