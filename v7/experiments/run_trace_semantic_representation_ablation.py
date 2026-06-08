"""Diagnose where coarse semantic information is readable inside trained models.

This evaluation-only ablation reuses semantic-alignment checkpoints.  It compares
raw pooled SNN traces, final SNN state, learned latent z, history-only deltas,
GRU hidden states, and the context-free current-text embedding.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

import run_context_objective_benchmark as base
import run_context_objective_benchmark_checked as checked
import run_trace_context_structure_benchmark as trace
import run_trace_semantic_alignment_benchmark as semantic

try:
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("scikit-learn is required for representation ablation") from exc


MODEL_TYPES = base.MODEL_TYPES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="runs/semantic_alignment_context_objective_lmstudio")
    parser.add_argument("--output", default="runs/trace_semantic_representation_ablation_lmstudio")
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


def modes_for(model_type: str) -> tuple[str, ...]:
    if model_type.startswith("snn_"):
        return ("raw_pool", "final_state", "latent_z", "history_delta_raw_pool", "history_delta_latent_z")
    if model_type == "gru_context_contrastive":
        return ("gru_hidden", "history_delta_gru_hidden")
    if model_type == "context_free_mlp":
        return ("current_text",)
    raise ValueError(f"unsupported model type: {model_type}")


def run_snn_window(*, model, events, text_encoder, args, device):
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
        raise RuntimeError("SNN feature extraction requires at least one event")
    return last_window


def feature_for_condition(*, model_type: str, model, episode, step_index: int, condition: str, swapped, text_encoder, args, device, mode: str) -> np.ndarray:
    events = trace.events_for_condition(
        episode=episode,
        step_index=step_index,
        condition=condition,
        swapped_history_source=swapped,
    )
    base.set_mode(model_type, model, training=False)
    with torch.no_grad():
        if model_type.startswith("snn_"):
            window = run_snn_window(model=model, events=events, text_encoder=text_encoder, args=args, device=device)
            raw_pool = trace.pool_window(window)
            final_state = torch.cat(
                [window.spike[:, -1, :], window.membrane[:, -1, :], window.adaptation[:, -1, :]], dim=-1
            )
            latent_z = model.trace_encoder(window.spike, window.membrane, window.adaptation)
            features = {"raw_pool": raw_pool, "final_state": final_state, "latent_z": latent_z}
            base_mode = mode.removeprefix("history_delta_")
            feature = features[base_mode]
            if mode.startswith("history_delta_"):
                reset_events = trace.events_for_condition(
                    episode=episode,
                    step_index=step_index,
                    condition="reset_history",
                    swapped_history_source=swapped,
                )
                reset_window = run_snn_window(model=model, events=reset_events, text_encoder=text_encoder, args=args, device=device)
                reset_features = {
                    "raw_pool": trace.pool_window(reset_window),
                    "latent_z": model.trace_encoder(reset_window.spike, reset_window.membrane, reset_window.adaptation),
                }
                feature = feature - reset_features[base_mode]
        elif model_type == "gru_context_contrastive":
            sequence = text_encoder.encode([event.text for event in events]).unsqueeze(0).to(device)
            feature = model.encode_context(sequence)
            if mode == "history_delta_gru_hidden":
                reset_events = trace.events_for_condition(
                    episode=episode,
                    step_index=step_index,
                    condition="reset_history",
                    swapped_history_source=swapped,
                )
                reset_sequence = text_encoder.encode([event.text for event in reset_events]).unsqueeze(0).to(device)
                feature = feature - model.encode_context(reset_sequence)
        elif model_type == "context_free_mlp":
            feature = text_encoder.encode([episode.events[step_index].text]).to(device)
        else:
            raise ValueError(f"unsupported model type: {model_type}")
    return feature.detach().cpu().reshape(-1).numpy()


def collect_rows(*, model_type: str, model, mode: str, pairs, episode_by_id, semantic_labels, text_encoder, args, device, condition: str):
    rows: list[dict[str, Any]] = []
    for pair in pairs:
        targeted_axis = semantic.targeted_axis_for(pair.relation)
        for side, episode_id, swapped_id in (
            ("left", pair.left_episode_id, pair.right_episode_id),
            ("right", pair.right_episode_id, pair.left_episode_id),
        ):
            episode = episode_by_id[episode_id]
            swapped = episode_by_id[swapped_id]
            rows.append(
                {
                    "episode_id": episode_id,
                    "relation": pair.relation,
                    "side": side,
                    "targeted_axis": targeted_axis,
                    "feature": feature_for_condition(
                        model_type=model_type,
                        model=model,
                        episode=episode,
                        step_index=pair.step_index,
                        condition=condition,
                        swapped=swapped,
                        text_encoder=text_encoder,
                        args=args,
                        device=device,
                        mode=mode,
                    ),
                    "label": semantic_labels[episode_id],
                }
            )
    return rows


def fit_probe(rows, alpha: float):
    x = np.stack([row["feature"] for row in rows])
    y = np.stack([row["label"] for row in rows])
    probe = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    probe.fit(x, y)
    return probe


def predict(probe, rows) -> np.ndarray:
    x = np.stack([row["feature"] for row in rows])
    return np.clip(probe.predict(x), 0.0, 1.0)


def evaluate_mode(*, model_type: str, model, mode: str, train_pairs, validation_pairs, episode_by_id, semantic_labels, text_encoder, args, device) -> dict[str, Any]:
    train_real = collect_rows(model_type=model_type, model=model, mode=mode, pairs=train_pairs, episode_by_id=episode_by_id, semantic_labels=semantic_labels, text_encoder=text_encoder, args=args, device=device, condition="real_history")
    validation_real = collect_rows(model_type=model_type, model=model, mode=mode, pairs=validation_pairs, episode_by_id=episode_by_id, semantic_labels=semantic_labels, text_encoder=text_encoder, args=args, device=device, condition="real_history")
    validation_shuffled = collect_rows(model_type=model_type, model=model, mode=mode, pairs=validation_pairs, episode_by_id=episode_by_id, semantic_labels=semantic_labels, text_encoder=text_encoder, args=args, device=device, condition="shuffled_history")
    validation_reset = collect_rows(model_type=model_type, model=model, mode=mode, pairs=validation_pairs, episode_by_id=episode_by_id, semantic_labels=semantic_labels, text_encoder=text_encoder, args=args, device=device, condition="reset_history")
    train_ids = {row["episode_id"] for row in train_real}
    validation_ids = {row["episode_id"] for row in validation_real}
    if train_ids.intersection(validation_ids):
        raise ValueError("representation ablation group leakage detected")
    probe = fit_probe(train_real, args.ridge_alpha)
    real_predictions = predict(probe, validation_real)
    shuffled_predictions = predict(probe, validation_shuffled)
    reset_predictions = predict(probe, validation_reset)
    real_mae = semantic.targeted_mae(validation_real, real_predictions)
    shuffled_mae = semantic.targeted_mae(validation_shuffled, shuffled_predictions)
    reset_mae = semantic.targeted_mae(validation_reset, reset_predictions)
    return {
        "feature_dim": int(train_real[0]["feature"].shape[0]),
        "real_targeted_mae": real_mae,
        "shuffled_targeted_mae": shuffled_mae,
        "reset_targeted_mae": reset_mae,
        "shuffled_history_mae_degradation": shuffled_mae - real_mae,
        "reset_history_mae_degradation": reset_mae - real_mae,
        "real_direction_accuracy": semantic.targeted_direction_accuracy(validation_real, real_predictions),
        "real_pair_order_accuracy": semantic.pair_order_accuracy(validation_real, real_predictions),
    }


def main() -> None:
    cli_args = parse_args()
    input_dir = Path(cli_args.input)
    metadata = trace.read_metadata(input_dir)
    args = resolve_args(cli_args, metadata)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    logger = base.RunLogger(output_dir=output, verbose=not args.quiet)
    logger.section("semantic representation ablation")
    logger.log("config", "Representation ablation 설정을 불러왔다.", **vars(args))

    device = torch.device(args.device)
    episodes = base.load_episodes(args.fixture)
    episode_by_id = {episode.episode_id: episode for episode in episodes}
    semantic_labels = semantic.load_semantic_labels(args.fixture)
    train_pairs = base.load_contrast_pairs(args.fixture, split="train")
    validation_pairs = base.load_contrast_pairs(args.fixture, split="validation")
    base.validate_contrast_pairs(episodes, train_pairs + validation_pairs)
    text_encoder = base.build_text_encoder(args, output)

    rows: list[dict[str, Any]] = []
    for seed in args.seeds:
        for model_type in MODEL_TYPES:
            path = trace.checkpoint_path(input_dir, seed, model_type)
            if not path.exists():
                raise FileNotFoundError(f"best checkpoint not found: {path}")
            model = base.build_model(model_type, text_dim=text_encoder.output_dim, num_neurons=args.num_neurons, seed=seed, device=device)
            checkpoint = torch.load(path, map_location=device, weights_only=False)
            checked.load_state_dict_for(model_type, model, checkpoint)
            for mode in modes_for(model_type):
                metrics = evaluate_mode(model_type=model_type, model=model, mode=mode, train_pairs=train_pairs, validation_pairs=validation_pairs, episode_by_id=episode_by_id, semantic_labels=semantic_labels, text_encoder=text_encoder, args=args, device=device)
                row = {"seed": seed, "model_type": model_type, "representation_mode": mode, **metrics}
                rows.append(row)
                logger.log("mode.done", "Representation mode 평가를 마쳤다.", **row)

    frame = pd.DataFrame(rows)
    frame.to_csv(output / "by_seed_representation.csv", index=False, encoding="utf-8-sig")
    numeric_columns = [column for column in frame.columns if column not in {"seed", "model_type", "representation_mode"}]
    summary = frame.groupby(["model_type", "representation_mode"])[numeric_columns].agg(["mean", "std", "min", "max"])
    summary.columns = ["_".join(column) for column in summary.columns]
    summary.reset_index().to_csv(output / "summary_by_representation.csv", index=False, encoding="utf-8-sig")
    (output / "metadata.json").write_text(json.dumps({"source_context_benchmark": str(input_dir), "fixture": args.fixture, "seeds": args.seeds, "ridge_alpha": args.ridge_alpha, "note": "Evaluation-only representation ablation. Semantic labels are never used to update the SNN, GRU, or text encoder."}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary.to_string())


if __name__ == "__main__":
    main()
