"""Run an OFAT sweep for neuron-memory-threshold SNN parameters.

The sweep keeps the verified memory-feedback architecture fixed and changes one
parameter at a time around the baseline configuration:

- feedback strength: 0.00, 0.01, 0.03, 0.05, 0.10
- memory threshold: 0.40, 0.50, 0.60, 0.70, 0.80
- accumulation decay: 0.70, 0.85, 0.95, 0.98

Duplicate configurations are trained once and referenced by multiple families.
Semantic labels remain evaluation-only and never update the SNN.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch

import run_context_objective_benchmark as base
import run_memory_threshold_semantic_benchmark as memory
import run_trace_semantic_alignment_benchmark as semantic


BASELINE_CONFIG = {
    "feedback_strength": 0.05,
    "memory_threshold": 0.60,
    "accumulation_decay": 0.85,
}
SWEEP_VALUES = {
    "feedback_strength": (0.00, 0.01, 0.03, 0.05, 0.10),
    "memory_threshold": (0.40, 0.50, 0.60, 0.70, 0.80),
    "accumulation_decay": (0.70, 0.85, 0.95, 0.98),
}
MODEL_TYPE = "snn_memory_feedback"
INTERPRETATION_BOUNDARY = (
    "This OFAT sweep identifies a stable parameter region for a controlled neuron-memory-threshold SNN fixture. "
    "It does not establish ground-truth emotions, biological fidelity, emergent clusters, or broad real-world generalization."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", default="fixtures/semantic_alignment_episodes.yaml")
    parser.add_argument("--output", default="runs/memory_threshold_parameter_sweep_lmstudio")
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
    parser.add_argument("--memory-decay", type=float, default=0.98)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 13, 21, 42, 100])
    parser.add_argument("--families", nargs="+", choices=sorted(SWEEP_VALUES), default=sorted(SWEEP_VALUES))
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def config_key(config: dict[str, float]) -> str:
    return (
        f"feedback_{config['feedback_strength']:.3f}__"
        f"threshold_{config['memory_threshold']:.3f}__"
        f"accumulation_decay_{config['accumulation_decay']:.3f}"
    )


def build_trials(families: list[str]) -> tuple[dict[str, dict[str, float]], list[dict[str, Any]]]:
    unique: dict[str, dict[str, float]] = {}
    references: list[dict[str, Any]] = []
    for family in families:
        for value in SWEEP_VALUES[family]:
            config = dict(BASELINE_CONFIG)
            config[family] = float(value)
            key = config_key(config)
            unique.setdefault(key, config)
            references.append(
                {
                    "family": family,
                    "value": float(value),
                    "config_key": key,
                    **config,
                }
            )
    return unique, references


def trial_args(args: argparse.Namespace, config: dict[str, float]) -> argparse.Namespace:
    trial = deepcopy(args)
    trial.feedback_strength = float(config["feedback_strength"])
    trial.memory_threshold = float(config["memory_threshold"])
    trial.accumulation_decay = float(config["accumulation_decay"])
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
    logger.section("memory-threshold OFAT parameter sweep")
    logger.log("config", "Memory-threshold OFAT sweep 설정을 불러왔다.", **vars(args))

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
            result = memory.train_one(
                model_type=MODEL_TYPE,
                feedback_strength=current_args.feedback_strength,
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
    numeric_columns = [
        column
        for column in frame.columns
        if column not in {"config_key", "seed", "model_type"}
    ]
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
        "semantic_labels_used_for_training": False,
        "interpretation_boundary": INTERPRETATION_BOUNDARY,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.log(
        "benchmark.done",
        "Memory-threshold OFAT sweep를 마쳤다.",
        unique_config_count=len(unique_configs),
        files=["run_log.jsonl", "embedding_cache.json", "by_seed_config.csv", "summary_by_config.csv", "sweep_references.csv", "metadata.json"],
    )
    print(summary.to_string())


if __name__ == "__main__":
    main()
