"""Evaluate whether learned internal representations change stably with prior context.

This benchmark reuses validation-best checkpoints produced by
``run_context_objective_benchmark_checked.py``.  It does not retrain the models.
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
import run_context_objective_benchmark_checked as checked

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
except ImportError as exc:  # pragma: no cover - dependency failure should be explicit
    raise RuntimeError("scikit-learn is required for the evaluation-only linear probe") from exc


MODEL_TYPES = base.MODEL_TYPES
DISTANCE_NAMES = ("euclidean", "cosine")
POOLING_SCHEMA = {
    "snn": "concat(mean, std, final, mean_abs_temporal_delta) for spike, membrane, adaptation of the final event window",
    "gru": "final GRU context hidden representation",
    "context_free_mlp": "current-event text embedding",
}
INTERPRETATION_BOUNDARY = (
    "This report evaluates whether internal traces are stable and context-dependent. "
    "It does not establish emotional semantics, interpretable neuron roles, emergent clusters, or biological fidelity."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="runs/context_objective_benchmark_lmstudio")
    parser.add_argument("--output", default="runs/trace_context_structure_benchmark_lmstudio")
    parser.add_argument("--fixture")
    parser.add_argument("--encoder", choices=["hash", "lmstudio"])
    parser.add_argument("--base-url")
    parser.add_argument("--embedding-model")
    parser.add_argument("--num-neurons", type=int)
    parser.add_argument("--event-ticks", type=int)
    parser.add_argument("--stimulation-ticks", type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def read_metadata(input_dir: Path) -> dict[str, Any]:
    path = input_dir / "metadata.json"
    if not path.exists():
        raise FileNotFoundError(f"context benchmark metadata not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"metadata must contain an object: {path}")
    return payload


def resolved_args(args: argparse.Namespace, metadata: dict[str, Any]) -> argparse.Namespace:
    def choose(name: str, fallback: Any) -> Any:
        value = getattr(args, name)
        return fallback if value is None else value

    return argparse.Namespace(
        input=args.input,
        output=args.output,
        fixture=choose("fixture", metadata.get("fixture", "fixtures/context_dependence_episodes.yaml")),
        encoder=choose("encoder", metadata.get("encoder", "hash")),
        base_url=args.base_url,
        embedding_model=choose("embedding_model", metadata.get("embedding_model", "text-embedding-nomic-embed-text-v1.5")),
        num_neurons=int(choose("num_neurons", metadata.get("num_neurons", 128))),
        event_ticks=int(choose("event_ticks", metadata.get("event_ticks", 16))),
        stimulation_ticks=int(choose("stimulation_ticks", metadata.get("stimulation_ticks", 6))),
        device=args.device,
        seeds=list(args.seeds if args.seeds is not None else metadata.get("seeds", [7, 13, 21, 42, 100])),
        quiet=args.quiet,
    )


def temporal_stats(sequence: torch.Tensor) -> torch.Tensor:
    """Pool one [batch, ticks, neurons] sequence without learning new parameters."""

    mean = sequence.mean(dim=1)
    std = sequence.std(dim=1, unbiased=False)
    final = sequence[:, -1, :]
    if sequence.shape[1] > 1:
        mean_abs_delta = (sequence[:, 1:, :] - sequence[:, :-1, :]).abs().mean(dim=1)
    else:
        mean_abs_delta = torch.zeros_like(final)
    return torch.cat([mean, std, final, mean_abs_delta], dim=-1)


def pool_window(window) -> torch.Tensor:
    return torch.cat(
        [
            temporal_stats(window.spike),
            temporal_stats(window.membrane),
            temporal_stats(window.adaptation),
        ],
        dim=-1,
    )


def events_for_condition(*, episode, step_index: int, condition: str, swapped_history_source=None):
    if condition in {"real_history", "same_context_repeat"}:
        return base.sequence_events(episode, step_index)
    if condition == "shuffled_history":
        if swapped_history_source is None:
            raise ValueError("shuffled_history requires swapped_history_source")
        return base.sequence_events(episode, step_index, swapped_history_source)
    if condition == "reset_history":
        return (episode.events[step_index],)
    raise ValueError(f"unsupported trace condition: {condition}")


def extract_representation(
    *,
    model_type: str,
    model,
    episode,
    step_index: int,
    condition: str,
    swapped_history_source,
    text_encoder,
    args: argparse.Namespace,
    device: torch.device,
) -> torch.Tensor:
    """Return one detached 1-D evaluation representation."""

    events = events_for_condition(
        episode=episode,
        step_index=step_index,
        condition=condition,
        swapped_history_source=swapped_history_source,
    )
    base.set_mode(model_type, model, training=False)
    with torch.no_grad():
        if model_type.startswith("snn_"):
            state = model.snn.initial_state(batch_size=1, device=device)
            last_window = None
            for event in events:
                embedding = text_encoder.encode([event.text]).to(device)
                current = model.event_encoder(embedding, [event])
                state, last_window = base.run_differentiable_window(
                    snn=model.snn,
                    event_current=current,
                    state=state,
                    event_ticks=args.event_ticks,
                    stimulation_ticks=args.stimulation_ticks,
                )
            if last_window is None:
                raise RuntimeError("SNN extraction requires at least one event")
            representation = pool_window(last_window)
        elif model_type == "gru_context_contrastive":
            texts = [event.text for event in events]
            sequence = text_encoder.encode(texts).unsqueeze(0).to(device)
            representation = model.encode_context(sequence)
        elif model_type == "context_free_mlp":
            representation = text_encoder.encode([episode.events[step_index].text]).to(device)
        else:
            raise ValueError(f"unsupported model type: {model_type}")
    return representation.detach().cpu().reshape(-1)


def distance(left: torch.Tensor, right: torch.Tensor, name: str) -> float:
    if name == "euclidean":
        return float(torch.linalg.vector_norm(left - right))
    if name == "cosine":
        return float(1.0 - F.cosine_similarity(left.unsqueeze(0), right.unsqueeze(0), dim=-1).item())
    raise ValueError(f"unsupported distance: {name}")


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(sum(values) / max(1, len(values)))


def extract_pair_vectors(*, model_type: str, model, pair, episode_by_id, text_encoder, args, device):
    left_episode = episode_by_id[pair.left_episode_id]
    right_episode = episode_by_id[pair.right_episode_id]
    vectors: dict[str, dict[str, torch.Tensor]] = {"left": {}, "right": {}}
    for side, episode, swapped in (
        ("left", left_episode, right_episode),
        ("right", right_episode, left_episode),
    ):
        for condition in ("real_history", "shuffled_history", "reset_history", "same_context_repeat"):
            vectors[side][condition] = extract_representation(
                model_type=model_type,
                model=model,
                episode=episode,
                step_index=pair.step_index,
                condition=condition,
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
        same = mean(
            distance(vectors[side]["real_history"], vectors[side]["same_context_repeat"], metric)
            for side in ("left", "right")
        )
        shuffled = mean(
            distance(vectors[side]["real_history"], vectors[side]["shuffled_history"], metric)
            for side in ("left", "right")
        )
        reset = mean(
            distance(vectors[side]["real_history"], vectors[side]["reset_history"], metric)
            for side in ("left", "right")
        )
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


def collect_real_probe_rows(*, model_type: str, model, pairs, episode_by_id, text_encoder, args, device):
    features: list[np.ndarray] = []
    labels: list[int] = []
    groups: list[str] = []
    for pair in pairs:
        for label, episode_id in ((0, pair.left_episode_id), (1, pair.right_episode_id)):
            episode = episode_by_id[episode_id]
            vector = extract_representation(
                model_type=model_type,
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


def evaluate_linear_probe(*, model_type: str, model, train_pairs, validation_pairs, episode_by_id, text_encoder, args, device, seed: int) -> dict[str, Any]:
    train_x, train_y, train_groups = collect_real_probe_rows(
        model_type=model_type,
        model=model,
        pairs=train_pairs,
        episode_by_id=episode_by_id,
        text_encoder=text_encoder,
        args=args,
        device=device,
    )
    validation_x, validation_y, validation_groups = collect_real_probe_rows(
        model_type=model_type,
        model=model,
        pairs=validation_pairs,
        episode_by_id=episode_by_id,
        text_encoder=text_encoder,
        args=args,
        device=device,
    )
    overlap = sorted(set(train_groups).intersection(validation_groups))
    if overlap:
        raise ValueError(f"linear probe group leakage detected: {overlap}")
    probe = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=seed, solver="liblinear"),
    )
    probe.fit(train_x, train_y)
    return {
        "linear_probe_accuracy": float(probe.score(validation_x, validation_y)),
        "linear_probe_chance_level": 0.5,
        "linear_probe_train_group_count": len(set(train_groups)),
        "linear_probe_validation_group_count": len(set(validation_groups)),
        "linear_probe_group_overlap_count": 0,
    }


def aggregate_pair_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    numeric_keys = [
        key
        for key, value in rows[0].items()
        if key not in {"step_index", "trace_dim"} and isinstance(value, (int, float))
    ]
    aggregated = {key: mean(float(row[key]) for row in rows) for key in numeric_keys}
    aggregated["trace_dim"] = float(rows[0]["trace_dim"])
    aggregated["validation_pair_count"] = float(len(rows))
    return aggregated


def evaluate_model(*, model_type: str, model, train_pairs, validation_pairs, episode_by_id, text_encoder, args, device, seed: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    pair_rows = []
    for pair in validation_pairs:
        vectors = extract_pair_vectors(
            model_type=model_type,
            model=model,
            pair=pair,
            episode_by_id=episode_by_id,
            text_encoder=text_encoder,
            args=args,
            device=device,
        )
        pair_rows.append(summarize_pair(pair, vectors))
    metrics = aggregate_pair_rows(pair_rows)
    metrics.update(
        evaluate_linear_probe(
            model_type=model_type,
            model=model,
            train_pairs=train_pairs,
            validation_pairs=validation_pairs,
            episode_by_id=episode_by_id,
            text_encoder=text_encoder,
            args=args,
            device=device,
            seed=seed,
        )
    )
    return metrics, pair_rows


def checkpoint_path(input_dir: Path, seed: int, model_type: str) -> Path:
    return input_dir / f"seed_{seed}" / model_type / "best_checkpoint.pt"


def main() -> None:
    cli_args = parse_args()
    input_dir = Path(cli_args.input)
    metadata = read_metadata(input_dir)
    args = resolved_args(cli_args, metadata)
    if not args.seeds:
        raise ValueError("--seeds must not be empty")
    if not 0 <= args.stimulation_ticks <= args.event_ticks:
        raise ValueError("--stimulation-ticks must be between 0 and --event-ticks")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    logger = base.RunLogger(output_dir=output, verbose=not args.quiet)
    logger.section("trace context structure benchmark")
    logger.log("config", "Trace 구조 benchmark 설정을 불러왔다.", **vars(args))

    device = torch.device(args.device)
    episodes = base.load_episodes(args.fixture)
    episode_by_id = {episode.episode_id: episode for episode in episodes}
    train_pairs = base.load_contrast_pairs(args.fixture, split="train")
    validation_pairs = base.load_contrast_pairs(args.fixture, split="validation")
    base.validate_contrast_pairs(episodes, train_pairs + validation_pairs)
    if not train_pairs or not validation_pairs:
        raise ValueError("fixture must contain train and validation contrast pairs")

    text_encoder = base.build_text_encoder(args, output)
    logger.log("embedding.ready", "Embedding encoder와 캐시가 준비됐다.", output_dim=text_encoder.output_dim)

    summary_rows: list[dict[str, Any]] = []
    for seed in args.seeds:
        logger.section(f"seed={seed}")
        for model_type in MODEL_TYPES:
            path = checkpoint_path(input_dir, seed, model_type)
            if not path.exists():
                raise FileNotFoundError(f"best checkpoint not found: {path}")
            model = base.build_model(
                model_type,
                text_dim=text_encoder.output_dim,
                num_neurons=args.num_neurons,
                seed=seed,
                device=device,
            )
            checkpoint = torch.load(path, map_location=device, weights_only=False)
            checked.load_state_dict_for(model_type, model, checkpoint)
            metrics, pair_rows = evaluate_model(
                model_type=model_type,
                model=model,
                train_pairs=train_pairs,
                validation_pairs=validation_pairs,
                episode_by_id=episode_by_id,
                text_encoder=text_encoder,
                args=args,
                device=device,
                seed=seed,
            )
            row = {"seed": seed, "model_type": model_type, **metrics}
            summary_rows.append(row)
            model_output = output / f"seed_{seed}" / model_type
            model_output.mkdir(parents=True, exist_ok=True)
            (model_output / "trace_structure_summary.json").write_text(
                json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            pd.DataFrame(pair_rows).to_csv(model_output / "validation_pair_metrics.csv", index=False, encoding="utf-8-sig")
            logger.log("model.done", "Trace 구조 평가를 마쳤다.", **row)

    frame = pd.DataFrame(summary_rows)
    frame.to_csv(output / "by_seed_model.csv", index=False, encoding="utf-8-sig")
    numeric_columns = [column for column in frame.columns if column not in {"seed", "model_type"}]
    summary = frame.groupby("model_type")[numeric_columns].agg(["mean", "std", "min", "max"])
    summary.columns = ["_".join(column) for column in summary.columns]
    summary.reset_index().to_csv(output / "summary_by_model.csv", index=False, encoding="utf-8-sig")
    output_metadata = {
        "source_context_benchmark": str(input_dir),
        "fixture": args.fixture,
        "encoder": args.encoder,
        "embedding_model": args.embedding_model,
        "seeds": args.seeds,
        "models": list(MODEL_TYPES),
        "event_ticks": args.event_ticks,
        "stimulation_ticks": args.stimulation_ticks,
        "num_neurons": args.num_neurons,
        "pooling_schema": POOLING_SCHEMA,
        "linear_probe": {
            "type": "evaluation-only logistic regression",
            "target": "contrast-pair side: left=0, right=1",
            "train_groups": "train episode ids",
            "validation_groups": "validation episode ids",
            "chance_level": 0.5,
        },
        "interpretation_boundary": INTERPRETATION_BOUNDARY,
    }
    (output / "metadata.json").write_text(json.dumps(output_metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.log("benchmark.done", "Trace context structure benchmark를 마쳤다.", files=["run_log.jsonl", "embedding_cache.json", "by_seed_model.csv", "summary_by_model.csv", "metadata.json"])
    print(summary.to_string())


if __name__ == "__main__":
    main()
