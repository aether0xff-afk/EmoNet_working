"""Evaluate whether internal representations align with coarse semantic axes.

The benchmark trains evaluation-only ridge probes on real-history traces from the
training split and evaluates unseen validation episodes.  It also measures
current-text leakage and degradation after history shuffling or reset.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

import run_context_objective_benchmark as base
import run_context_objective_benchmark_checked as checked
import run_trace_context_structure_benchmark as trace

try:
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("scikit-learn is required for the evaluation-only semantic probe") from exc


AXES = ("valence", "arousal", "certainty", "social_distance")
MODEL_TYPES = base.MODEL_TYPES
INTERPRETATION_BOUNDARY = (
    "This report evaluates coarse semantic alignment of internal traces under a controlled fixture. "
    "It does not establish ground-truth emotions, universal neuron meanings, emergent clusters, biological fidelity, or broad real-world generalization."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="runs/semantic_alignment_context_objective_lmstudio")
    parser.add_argument("--output", default="runs/trace_semantic_alignment_benchmark_lmstudio")
    parser.add_argument("--fixture", default="fixtures/semantic_alignment_episodes.yaml")
    parser.add_argument("--encoder", choices=["hash", "lmstudio"])
    parser.add_argument("--base-url")
    parser.add_argument("--embedding-model")
    parser.add_argument("--num-neurons", type=int)
    parser.add_argument("--event-ticks", type=int)
    parser.add_argument("--stimulation-ticks", type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def load_semantic_labels(path: str | Path) -> dict[str, np.ndarray]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    raw_labels = payload.get("semantic_labels") if isinstance(payload, dict) else None
    if not isinstance(raw_labels, dict) or not raw_labels:
        raise ValueError("semantic fixture must contain a non-empty semantic_labels mapping")
    labels: dict[str, np.ndarray] = {}
    for episode_id, raw in raw_labels.items():
        if not isinstance(raw, dict):
            raise ValueError(f"semantic label must be an object: {episode_id}")
        missing = [axis for axis in AXES if axis not in raw]
        if missing:
            raise ValueError(f"semantic label is missing axes for {episode_id}: {missing}")
        vector = np.asarray([float(raw[axis]) for axis in AXES], dtype=np.float64)
        if np.any(vector < 0.0) or np.any(vector > 1.0):
            raise ValueError(f"semantic labels must remain in [0, 1]: {episode_id}")
        labels[str(episode_id)] = vector
    return labels


def resolve_args(args: argparse.Namespace, metadata: dict[str, Any]) -> argparse.Namespace:
    def choose(value: Any, key: str, fallback: Any) -> Any:
        return metadata.get(key, fallback) if value is None else value

    return argparse.Namespace(
        input=args.input,
        output=args.output,
        fixture=args.fixture,
        encoder=choose(args.encoder, "encoder", "hash"),
        base_url=args.base_url,
        embedding_model=choose(args.embedding_model, "embedding_model", "text-embedding-nomic-embed-text-v1.5"),
        num_neurons=int(choose(args.num_neurons, "num_neurons", 128)),
        event_ticks=int(choose(args.event_ticks, "event_ticks", 16)),
        stimulation_ticks=int(choose(args.stimulation_ticks, "stimulation_ticks", 6)),
        device=args.device,
        seeds=list(args.seeds if args.seeds is not None else metadata.get("seeds", [7, 13, 21, 42, 100])),
        ridge_alpha=float(args.ridge_alpha),
        quiet=args.quiet,
    )


def targeted_axis_for(relation: str) -> str:
    for axis in AXES:
        if relation.startswith(axis):
            return axis
    raise ValueError(f"contrast relation must start with one semantic axis: {relation}")


def current_text_vector(*, episode, step_index: int, text_encoder) -> np.ndarray:
    return text_encoder.encode([episode.events[step_index].text]).detach().cpu().reshape(-1).numpy()


def collect_rows(*, model_type: str, model, pairs, episode_by_id, semantic_labels, text_encoder, args, device, condition: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pair in pairs:
        targeted_axis = targeted_axis_for(pair.relation)
        for side, episode_id, swapped_id in (
            ("left", pair.left_episode_id, pair.right_episode_id),
            ("right", pair.right_episode_id, pair.left_episode_id),
        ):
            episode = episode_by_id[episode_id]
            swapped = episode_by_id[swapped_id]
            representation = trace.extract_representation(
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
            rows.append(
                {
                    "episode_id": episode_id,
                    "relation": pair.relation,
                    "side": side,
                    "targeted_axis": targeted_axis,
                    "trace": representation.numpy(),
                    "current_text": current_text_vector(episode=episode, step_index=pair.step_index, text_encoder=text_encoder),
                    "label": semantic_labels[episode_id],
                }
            )
    return rows


def matrices(rows: list[dict[str, Any]], feature_key: str) -> tuple[np.ndarray, np.ndarray]:
    return np.stack([row[feature_key] for row in rows]), np.stack([row["label"] for row in rows])


def fit_probe(train_rows: list[dict[str, Any]], *, feature_key: str, alpha: float):
    x, y = matrices(train_rows, feature_key)
    probe = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    probe.fit(x, y)
    return probe


def predict_rows(probe, rows: list[dict[str, Any]], *, feature_key: str) -> np.ndarray:
    x, _ = matrices(rows, feature_key)
    return np.clip(probe.predict(x), 0.0, 1.0)


def targeted_values(rows: list[dict[str, Any]], predictions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    truth: list[float] = []
    predicted: list[float] = []
    for row, prediction in zip(rows, predictions, strict=True):
        index = AXES.index(row["targeted_axis"])
        truth.append(float(row["label"][index]))
        predicted.append(float(prediction[index]))
    return np.asarray(truth), np.asarray(predicted)


def targeted_mae(rows: list[dict[str, Any]], predictions: np.ndarray) -> float:
    truth, predicted = targeted_values(rows, predictions)
    return float(np.abs(predicted - truth).mean())


def targeted_direction_accuracy(rows: list[dict[str, Any]], predictions: np.ndarray) -> float:
    truth, predicted = targeted_values(rows, predictions)
    return float(((predicted >= 0.5) == (truth >= 0.5)).mean())


def pair_order_accuracy(rows: list[dict[str, Any]], predictions: np.ndarray) -> float:
    prediction_by_episode = {row["episode_id"]: prediction for row, prediction in zip(rows, predictions, strict=True)}
    label_by_episode = {row["episode_id"]: row["label"] for row in rows}
    grouped: dict[str, dict[str, str]] = {}
    for row in rows:
        grouped.setdefault(row["relation"], {})[row["side"]] = row["episode_id"]
    correct: list[float] = []
    for relation, sides in grouped.items():
        left_id = sides["left"]
        right_id = sides["right"]
        axis = targeted_axis_for(relation)
        index = AXES.index(axis)
        truth_delta = float(label_by_episode[left_id][index] - label_by_episode[right_id][index])
        prediction_delta = float(prediction_by_episode[left_id][index] - prediction_by_episode[right_id][index])
        correct.append(float(truth_delta * prediction_delta > 0.0))
    return float(np.mean(correct))


def axis_metrics(rows: list[dict[str, Any]], predictions: np.ndarray, prefix: str) -> dict[str, float]:
    result: dict[str, float] = {}
    for axis in AXES:
        selected = [index for index, row in enumerate(rows) if row["targeted_axis"] == axis]
        if not selected:
            raise ValueError(f"no evaluation rows for semantic axis: {axis}")
        axis_rows = [rows[index] for index in selected]
        axis_predictions = predictions[selected]
        result[f"{prefix}_{axis}_targeted_mae"] = targeted_mae(axis_rows, axis_predictions)
        result[f"{prefix}_{axis}_direction_accuracy"] = targeted_direction_accuracy(axis_rows, axis_predictions)
    return result


def evaluate_model(*, model_type: str, model, train_pairs, validation_pairs, episode_by_id, semantic_labels, text_encoder, args, device) -> dict[str, Any]:
    train_real = collect_rows(
        model_type=model_type,
        model=model,
        pairs=train_pairs,
        episode_by_id=episode_by_id,
        semantic_labels=semantic_labels,
        text_encoder=text_encoder,
        args=args,
        device=device,
        condition="real_history",
    )
    validation_real = collect_rows(
        model_type=model_type,
        model=model,
        pairs=validation_pairs,
        episode_by_id=episode_by_id,
        semantic_labels=semantic_labels,
        text_encoder=text_encoder,
        args=args,
        device=device,
        condition="real_history",
    )
    validation_shuffled = collect_rows(
        model_type=model_type,
        model=model,
        pairs=validation_pairs,
        episode_by_id=episode_by_id,
        semantic_labels=semantic_labels,
        text_encoder=text_encoder,
        args=args,
        device=device,
        condition="shuffled_history",
    )
    validation_reset = collect_rows(
        model_type=model_type,
        model=model,
        pairs=validation_pairs,
        episode_by_id=episode_by_id,
        semantic_labels=semantic_labels,
        text_encoder=text_encoder,
        args=args,
        device=device,
        condition="reset_history",
    )

    train_ids = {row["episode_id"] for row in train_real}
    validation_ids = {row["episode_id"] for row in validation_real}
    overlap = sorted(train_ids.intersection(validation_ids))
    if overlap:
        raise ValueError(f"semantic probe group leakage detected: {overlap}")

    trace_probe = fit_probe(train_real, feature_key="trace", alpha=args.ridge_alpha)
    text_probe = fit_probe(train_real, feature_key="current_text", alpha=args.ridge_alpha)
    real_predictions = predict_rows(trace_probe, validation_real, feature_key="trace")
    shuffled_predictions = predict_rows(trace_probe, validation_shuffled, feature_key="trace")
    reset_predictions = predict_rows(trace_probe, validation_reset, feature_key="trace")
    text_predictions = predict_rows(text_probe, validation_real, feature_key="current_text")

    train_labels = np.stack([row["label"] for row in train_real])
    constant_vector = train_labels.mean(axis=0)
    constant_predictions = np.repeat(constant_vector[None, :], len(validation_real), axis=0)

    real_mae = targeted_mae(validation_real, real_predictions)
    shuffled_mae = targeted_mae(validation_shuffled, shuffled_predictions)
    reset_mae = targeted_mae(validation_reset, reset_predictions)
    text_mae = targeted_mae(validation_real, text_predictions)
    constant_mae = targeted_mae(validation_real, constant_predictions)

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
        "real_direction_accuracy": targeted_direction_accuracy(validation_real, real_predictions),
        "shuffled_direction_accuracy": targeted_direction_accuracy(validation_shuffled, shuffled_predictions),
        "reset_direction_accuracy": targeted_direction_accuracy(validation_reset, reset_predictions),
        "current_text_baseline_direction_accuracy": targeted_direction_accuracy(validation_real, text_predictions),
        "real_pair_order_accuracy": pair_order_accuracy(validation_real, real_predictions),
        "shuffled_pair_order_accuracy": pair_order_accuracy(validation_shuffled, shuffled_predictions),
        "reset_pair_order_accuracy": pair_order_accuracy(validation_reset, reset_predictions),
        "current_text_baseline_pair_order_accuracy": pair_order_accuracy(validation_real, text_predictions),
    }
    result.update(axis_metrics(validation_real, real_predictions, "real"))
    result.update(axis_metrics(validation_shuffled, shuffled_predictions, "shuffled"))
    result.update(axis_metrics(validation_reset, reset_predictions, "reset"))
    result.update(axis_metrics(validation_real, text_predictions, "current_text_baseline"))
    return result


def main() -> None:
    cli_args = parse_args()
    input_dir = Path(cli_args.input)
    metadata = trace.read_metadata(input_dir)
    args = resolve_args(cli_args, metadata)
    if not args.seeds:
        raise ValueError("--seeds must not be empty")
    if args.ridge_alpha < 0:
        raise ValueError("--ridge-alpha must be non-negative")
    if not 0 <= args.stimulation_ticks <= args.event_ticks:
        raise ValueError("--stimulation-ticks must be between 0 and --event-ticks")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    logger = base.RunLogger(output_dir=output, verbose=not args.quiet)
    logger.section("trace semantic alignment benchmark")
    logger.log("config", "Semantic alignment benchmark 설정을 불러왔다.", **vars(args))

    device = torch.device(args.device)
    episodes = base.load_episodes(args.fixture)
    episode_by_id = {episode.episode_id: episode for episode in episodes}
    semantic_labels = load_semantic_labels(args.fixture)
    missing_labels = sorted(set(episode_by_id) - set(semantic_labels))
    if missing_labels:
        raise ValueError(f"semantic labels are missing for episodes: {missing_labels}")
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
            path = trace.checkpoint_path(input_dir, seed, model_type)
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
            metrics = evaluate_model(
                model_type=model_type,
                model=model,
                train_pairs=train_pairs,
                validation_pairs=validation_pairs,
                episode_by_id=episode_by_id,
                semantic_labels=semantic_labels,
                text_encoder=text_encoder,
                args=args,
                device=device,
            )
            row = {"seed": seed, "model_type": model_type, **metrics}
            summary_rows.append(row)
            model_output = output / f"seed_{seed}" / model_type
            model_output.mkdir(parents=True, exist_ok=True)
            (model_output / "semantic_alignment_summary.json").write_text(
                json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            logger.log("model.done", "Semantic alignment 평가를 마쳤다.", **row)

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
        "axes": list(AXES),
        "ridge_alpha": args.ridge_alpha,
        "probe": "evaluation-only StandardScaler + Ridge trained on real-history train episodes",
        "leakage_control": "current-text-only ridge baseline; current text is identical inside each contrast pair",
        "interpretation_boundary": INTERPRETATION_BOUNDARY,
    }
    (output / "metadata.json").write_text(json.dumps(output_metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.log("benchmark.done", "Trace semantic alignment benchmark를 마쳤다.", files=["run_log.jsonl", "embedding_cache.json", "by_seed_model.csv", "summary_by_model.csv", "metadata.json"])
    print(summary.to_string())


if __name__ == "__main__":
    main()
