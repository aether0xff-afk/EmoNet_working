"""Re-check trace-context structure for a selected neuron-memory-threshold config.

This evaluation reuses sweep checkpoints and does not retrain the model. It
measures whether real history, shuffled history, reset history, and exact repeat
conditions remain distinguishable after adding neuron-local accumulation and a
separate memory threshold.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

import run_context_objective_benchmark as base
import run_memory_threshold_semantic_benchmark as memory
import run_trace_semantic_alignment_benchmark as semantic

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("scikit-learn is required for the evaluation-only linear probe") from exc


DISTANCE_NAMES = ("euclidean", "cosine")
DEFAULT_CONFIG_KEY = "feedback_0.050__threshold_0.500__accumulation_decay_0.850"
INTERPRETATION_BOUNDARY = (
    "This report evaluates whether traces from a selected neuron-memory-threshold SNN are stable and context-dependent. "
    "It does not establish emotional ground truth, emergent clusters, or biological fidelity."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="runs/memory_threshold_parameter_sweep_lmstudio")
    parser.add_argument("--output", default="runs/memory_threshold_context_structure_best_lmstudio")
    parser.add_argument("--fixture", default="fixtures/semantic_alignment_episodes.yaml")
    parser.add_argument("--config-key", default=DEFAULT_CONFIG_KEY)
    parser.add_argument("--feedback-strength", type=float, default=0.05)
    parser.add_argument("--memory-threshold", type=float, default=0.50)
    parser.add_argument("--accumulation-decay", type=float, default=0.85)
    parser.add_argument("--memory-decay", type=float, default=0.98)
    parser.add_argument("--encoder", choices=["hash", "lmstudio"], default="hash")
    parser.add_argument("--base-url")
    parser.add_argument("--embedding-model", default="text-embedding-nomic-embed-text-v1.5")
    parser.add_argument("--num-neurons", type=int, default=128)
    parser.add_argument("--event-ticks", type=int, default=16)
    parser.add_argument("--stimulation-ticks", type=int, default=6)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 13, 21, 42, 100])
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def checkpoint_path(input_dir: Path, config_key: str, seed: int) -> Path:
    return input_dir / "trials" / config_key / f"seed_{seed}" / "snn_memory_feedback" / "best_checkpoint.pt"


def set_mode(model, *, training: bool) -> None:
    memory.set_mode(model, training=training)


def extract_representation(*, model, episode, step_index: int, condition: str, swapped_history_source, text_encoder, args, device) -> torch.Tensor:
    set_mode(model, training=False)
    with torch.no_grad():
        output = memory.run_sequence(
            model=model,
            episode=episode,
            step_index=step_index,
            condition=condition,
            swapped_history_source=swapped_history_source,
            text_encoder=text_encoder,
            args=args,
            device=device,
        )
    return output.raw_representation.detach().cpu().reshape(-1)


def distance(left: torch.Tensor, right: torch.Tensor, name: str) -> float:
    if name == "euclidean":
        return float(torch.linalg.vector_norm(left - right))
    if name == "cosine":
        return float(1.0 - F.cosine_similarity(left.unsqueeze(0), right.unsqueeze(0), dim=-1).item())
    raise ValueError(f"unsupported distance: {name}")


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(sum(values) / max(1, len(values)))


def extract_pair_vectors(*, model, pair, episode_by_id, text_encoder, args, device):
    left_episode = episode_by_id[pair.left_episode_id]
    right_episode = episode_by_id[pair.right_episode_id]
    vectors: dict[str, dict[str, torch.Tensor]] = {"left": {}, "right": {}}
    for side, episode, swapped in (
        ("left", left_episode, right_episode),
        ("right", right_episode, left_episode),
    ):
        for condition in ("real_history", "shuffled_history", "reset_history", "same_context_repeat"):
            normalized_condition = "real_history" if condition == "same_context_repeat" else condition
            vectors[side][condition] = extract_representation(
                model=model,
                episode=episode,
                step_index=pair.step_index,
                condition=normalized_condition,
                swapped_history_source=swapped,
                text_encoder=text_encoder,
                args=args,
                device=device,
            )
    return vectors


def summarize_pair(pair, vectors: dict[str, dict[str, torch.Tensor]]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "relation": pair.relation,
        "left_episode_id": pair.left_episode_id,
        "right_episode_id": pair.right_episode_id,
        "step_index": pair.step_index,
        "trace_dim": int(vectors["left"]["real_history"].numel()),
    }
    for metric in DISTANCE_NAMES:
        same = mean(distance(vectors[side]["real_history"], vectors[side]["same_context_repeat"], metric) for side in ("left", "right"))
        shuffled = mean(distance(vectors[side]["real_history"], vectors[side]["shuffled_history"], metric) for side in ("left", "right"))
        reset = mean(distance(vectors[side]["real_history"], vectors[side]["reset_history"], metric) for side in ("left", "right"))
        retrieval = mean(
            [
                float(distance(vectors["left"]["shuffled_history"], vectors["right"]["real_history"], metric) < distance(vectors["left"]["shuffled_history"], vectors["left"]["real_history"], metric)),
                float(distance(vectors["right"]["shuffled_history"], vectors["left"]["real_history"], metric) < distance(vectors["right"]["shuffled_history"], vectors["right"]["real_history"], metric)),
            ]
        )
        row.update(
            {
                f"same_context_trace_distance_{metric}": same,
                f"real_vs_shuffled_trace_distance_{metric}": shuffled,
                f"real_vs_reset_trace_distance_{metric}": reset,
                f"trace_context_gap_{metric}": shuffled - same,
                f"trace_reset_gap_{metric}": reset - same,
                f"context_retrieval_accuracy_{metric}": retrieval,
            }
        )
    return row


def collect_real_probe_rows(*, model, pairs, episode_by_id, text_encoder, args, device):
    features: list[np.ndarray] = []
    labels: list[int] = []
    groups: list[str] = []
    for pair in pairs:
        for label, episode_id in ((0, pair.left_episode_id), (1, pair.right_episode_id)):
            episode = episode_by_id[episode_id]
            vector = extract_representation(
                model=model,
                episode=episode,
                step_index=pair.step_index,
                condition="real_history",
                swapped_history_source=None,
                text_encoder=text_encoder,
                args=args,
                device=device,
            )
            features.append(vector.numpy())
            labels.append(label)
            groups.append(episode_id)
    return np.stack(features), np.asarray(labels, dtype=np.int64), groups


def evaluate_linear_probe(*, model, train_pairs, validation_pairs, episode_by_id, text_encoder, args, device, seed: int) -> dict[str, Any]:
    train_x, train_y, train_groups = collect_real_probe_rows(model=model, pairs=train_pairs, episode_by_id=episode_by_id, text_encoder=text_encoder, args=args, device=device)
    validation_x, validation_y, validation_groups = collect_real_probe_rows(model=model, pairs=validation_pairs, episode_by_id=episode_by_id, text_encoder=text_encoder, args=args, device=device)
    overlap = sorted(set(train_groups).intersection(validation_groups))
    if overlap:
        raise ValueError(f"linear probe group leakage detected: {overlap}")
    probe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=seed, solver="liblinear"))
    probe.fit(train_x, train_y)
    return {
        "linear_probe_accuracy": float(probe.score(validation_x, validation_y)),
        "linear_probe_chance_level": 0.5,
        "linear_probe_train_group_count": len(set(train_groups)),
        "linear_probe_validation_group_count": len(set(validation_groups)),
        "linear_probe_group_overlap_count": 0,
    }


def aggregate_pair_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    numeric_keys = [key for key, value in rows[0].items() if key not in {"step_index", "trace_dim"} and isinstance(value, (int, float))]
    aggregated = {key: mean(float(row[key]) for row in rows) for key in numeric_keys}
    aggregated["trace_dim"] = float(rows[0]["trace_dim"])
    aggregated["validation_pair_count"] = float(len(rows))
    return aggregated


def evaluate_model(*, model, train_pairs, validation_pairs, episode_by_id, text_encoder, args, device, seed: int):
    pair_rows = []
    for pair in validation_pairs:
        vectors = extract_pair_vectors(model=model, pair=pair, episode_by_id=episode_by_id, text_encoder=text_encoder, args=args, device=device)
        pair_rows.append(summarize_pair(pair, vectors))
    metrics = aggregate_pair_rows(pair_rows)
    metrics.update(evaluate_linear_probe(model=model, train_pairs=train_pairs, validation_pairs=validation_pairs, episode_by_id=episode_by_id, text_encoder=text_encoder, args=args, device=device, seed=seed))
    return metrics, pair_rows


def main() -> None:
    args = parse_args()
    if not args.seeds:
        raise ValueError("--seeds must not be empty")
    if not 0 <= args.stimulation_ticks <= args.event_ticks:
        raise ValueError("--stimulation-ticks must be between 0 and --event-ticks")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    logger = base.RunLogger(output_dir=output, verbose=not args.quiet)
    logger.section("selected memory-threshold context-structure benchmark")
    logger.log("config", "선택 memory-threshold 설정의 context structure를 재검증한다.", **vars(args))

    device = torch.device(args.device)
    episodes = base.load_episodes(args.fixture)
    episode_by_id = {episode.episode_id: episode for episode in episodes}
    train_pairs = base.load_contrast_pairs(args.fixture, split="train")
    validation_pairs = base.load_contrast_pairs(args.fixture, split="validation")
    base.validate_contrast_pairs(episodes, train_pairs + validation_pairs)
    text_encoder = base.build_text_encoder(args, output)
    logger.log("embedding.ready", "Embedding encoder와 캐시가 준비됐다.", output_dim=text_encoder.output_dim)

    rows: list[dict[str, Any]] = []
    input_dir = Path(args.input)
    for seed in args.seeds:
        path = checkpoint_path(input_dir, args.config_key, seed)
        if not path.exists():
            raise FileNotFoundError(f"best checkpoint not found: {path}")
        model = memory.build_model(text_dim=text_encoder.output_dim, num_neurons=args.num_neurons, seed=seed, feedback_strength=args.feedback_strength, args=args, device=device)
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        memory.load_state_dict_for(model, checkpoint)
        metrics, pair_rows = evaluate_model(model=model, train_pairs=train_pairs, validation_pairs=validation_pairs, episode_by_id=episode_by_id, text_encoder=text_encoder, args=args, device=device, seed=seed)
        row = {"seed": seed, "model_type": "snn_memory_feedback", "config_key": args.config_key, **metrics}
        rows.append(row)
        model_output = output / f"seed_{seed}"
        model_output.mkdir(parents=True, exist_ok=True)
        (model_output / "trace_structure_summary.json").write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
        pd.DataFrame(pair_rows).to_csv(model_output / "validation_pair_metrics.csv", index=False, encoding="utf-8-sig")
        logger.log("model.done", "선택 설정 trace 구조 평가를 마쳤다.", **row)

    frame = pd.DataFrame(rows)
    frame.to_csv(output / "by_seed_model.csv", index=False, encoding="utf-8-sig")
    numeric_columns = [column for column in frame.columns if column not in {"seed", "model_type", "config_key"}]
    summary = frame.groupby("model_type")[numeric_columns].agg(["mean", "std", "min", "max"])
    summary.columns = ["_".join(column) for column in summary.columns]
    summary.reset_index().to_csv(output / "summary_by_model.csv", index=False, encoding="utf-8-sig")
    metadata = {
        "source_sweep": str(input_dir),
        "config_key": args.config_key,
        "feedback_strength": args.feedback_strength,
        "memory_threshold": args.memory_threshold,
        "accumulation_decay": args.accumulation_decay,
        "memory_decay": args.memory_decay,
        "fixture": args.fixture,
        "seeds": args.seeds,
        "interpretation_boundary": INTERPRETATION_BOUNDARY,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.log("benchmark.done", "선택 설정의 context-structure benchmark를 마쳤다.", files=["run_log.jsonl", "embedding_cache.json", "by_seed_model.csv", "summary_by_model.csv", "metadata.json"])
    print(summary.to_string())


if __name__ == "__main__":
    main()
