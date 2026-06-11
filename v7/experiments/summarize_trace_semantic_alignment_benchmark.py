"""Summarize trace semantic-alignment benchmark outputs into a conservative decision report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


REQUIRED_MODELS = {
    "snn_next_only",
    "snn_context_contrastive",
    "context_free_mlp",
    "gru_context_contrastive",
}
AXES = ("valence", "arousal", "certainty", "social_distance")
FLOAT_TOLERANCE = 1e-6
MIN_POSITIVE_RATE = 0.8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="runs/trace_semantic_alignment_benchmark_lmstudio")
    parser.add_argument("--output")
    return parser.parse_args()


def mean_for(frame: pd.DataFrame, model_type: str, column: str) -> float:
    selected = frame.loc[frame["model_type"] == model_type, column]
    if selected.empty:
        raise ValueError(f"missing model rows for {model_type}")
    return float(selected.mean())


def positive_rate(frame: pd.DataFrame, model_type: str, column: str) -> float:
    selected = frame.loc[frame["model_type"] == model_type, column]
    if selected.empty:
        raise ValueError(f"missing model rows for {model_type}")
    return float((selected > 0).mean())


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input)
    csv_path = input_dir / "by_seed_model.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"benchmark result not found: {csv_path}")
    frame = pd.read_csv(csv_path)
    present_models = set(frame["model_type"].unique())
    missing = REQUIRED_MODELS - present_models
    if missing:
        raise ValueError(f"missing benchmark models: {sorted(missing)}")

    snn_real_mae = mean_for(frame, "snn_context_contrastive", "real_targeted_mae")
    snn_shuffled_mae = mean_for(frame, "snn_context_contrastive", "shuffled_targeted_mae")
    snn_reset_mae = mean_for(frame, "snn_context_contrastive", "reset_targeted_mae")
    snn_text_mae = mean_for(frame, "snn_context_contrastive", "current_text_baseline_targeted_mae")
    snn_constant_mae = mean_for(frame, "snn_context_contrastive", "constant_baseline_targeted_mae")
    snn_pair_order = mean_for(frame, "snn_context_contrastive", "real_pair_order_accuracy")
    snn_direction = mean_for(frame, "snn_context_contrastive", "real_direction_accuracy")
    next_only_mae = mean_for(frame, "snn_next_only", "real_targeted_mae")
    gru_mae = mean_for(frame, "gru_context_contrastive", "real_targeted_mae")
    mlp_mae = mean_for(frame, "context_free_mlp", "real_targeted_mae")
    mlp_shuffle_degradation = mean_for(frame, "context_free_mlp", "shuffled_history_mae_degradation")
    mlp_reset_degradation = mean_for(frame, "context_free_mlp", "reset_history_mae_degradation")

    rate_real_vs_constant = positive_rate(frame, "snn_context_contrastive", "real_minus_constant_mae_improvement")
    rate_real_vs_text = positive_rate(frame, "snn_context_contrastive", "real_minus_text_baseline_mae_improvement")
    rate_shuffled_degradation = positive_rate(frame, "snn_context_contrastive", "shuffled_history_mae_degradation")
    rate_reset_degradation = positive_rate(frame, "snn_context_contrastive", "reset_history_mae_degradation")

    axis_means = {
        axis: {
            "real_targeted_mae": mean_for(frame, "snn_context_contrastive", f"real_{axis}_targeted_mae"),
            "real_direction_accuracy": mean_for(frame, "snn_context_contrastive", f"real_{axis}_direction_accuracy"),
            "shuffled_targeted_mae": mean_for(frame, "snn_context_contrastive", f"shuffled_{axis}_targeted_mae"),
            "current_text_baseline_targeted_mae": mean_for(frame, "snn_context_contrastive", f"current_text_baseline_{axis}_targeted_mae"),
        }
        for axis in AXES
    }

    checks = {
        "contrastive_snn_semantic_mae_beats_constant_baseline": snn_real_mae < snn_constant_mae,
        "contrastive_snn_semantic_mae_beats_current_text_baseline": snn_real_mae < snn_text_mae,
        "contrastive_snn_semantics_degrade_when_history_is_shuffled": snn_shuffled_mae > snn_real_mae,
        "contrastive_snn_semantics_degrade_when_history_is_reset": snn_reset_mae > snn_real_mae,
        "contrastive_snn_pair_order_is_above_chance": snn_pair_order > 0.5,
        "contrastive_snn_direction_accuracy_is_above_chance": snn_direction > 0.5,
        "contrastive_snn_beats_next_only_mae": snn_real_mae < next_only_mae,
        "contrastive_snn_is_not_worse_than_gru_mae": snn_real_mae <= gru_mae,
        "contrastive_snn_seed_stability_is_sufficient": (
            rate_real_vs_constant >= MIN_POSITIVE_RATE
            and rate_real_vs_text >= MIN_POSITIVE_RATE
            and rate_shuffled_degradation >= MIN_POSITIVE_RATE
            and rate_reset_degradation >= MIN_POSITIVE_RATE
        ),
        "context_free_mlp_is_history_invariant": abs(mlp_shuffle_degradation) <= FLOAT_TOLERANCE and abs(mlp_reset_degradation) <= FLOAT_TOLERANCE,
    }
    semantic_alignment_established = all(
        checks[key]
        for key in (
            "contrastive_snn_semantic_mae_beats_current_text_baseline",
            "contrastive_snn_semantics_degrade_when_history_is_shuffled",
            "contrastive_snn_semantics_degrade_when_history_is_reset",
            "contrastive_snn_pair_order_is_above_chance",
            "contrastive_snn_direction_accuracy_is_above_chance",
            "contrastive_snn_beats_next_only_mae",
            "contrastive_snn_seed_stability_is_sufficient",
            "context_free_mlp_is_history_invariant",
        )
    )
    report = {
        "input": str(input_dir),
        "seed_count": int(frame["seed"].nunique()),
        "stage_verdict": "established" if semantic_alignment_established else "not_established",
        "means": {
            "snn_context_contrastive_real_targeted_mae": snn_real_mae,
            "snn_context_contrastive_shuffled_targeted_mae": snn_shuffled_mae,
            "snn_context_contrastive_reset_targeted_mae": snn_reset_mae,
            "snn_context_contrastive_current_text_baseline_targeted_mae": snn_text_mae,
            "snn_context_contrastive_constant_baseline_targeted_mae": snn_constant_mae,
            "snn_context_contrastive_real_pair_order_accuracy": snn_pair_order,
            "snn_context_contrastive_real_direction_accuracy": snn_direction,
            "snn_next_only_real_targeted_mae": next_only_mae,
            "gru_context_contrastive_real_targeted_mae": gru_mae,
            "context_free_mlp_real_targeted_mae": mlp_mae,
        },
        "axis_means_for_snn_context_contrastive": axis_means,
        "positive_rates": {
            "snn_real_minus_constant_mae_improvement": rate_real_vs_constant,
            "snn_real_minus_text_baseline_mae_improvement": rate_real_vs_text,
            "snn_shuffled_history_mae_degradation": rate_shuffled_degradation,
            "snn_reset_history_mae_degradation": rate_reset_degradation,
        },
        "comparisons": {
            "snn_constant_baseline_minus_real_mae": snn_constant_mae - snn_real_mae,
            "snn_current_text_baseline_minus_real_mae": snn_text_mae - snn_real_mae,
            "snn_shuffled_minus_real_mae": snn_shuffled_mae - snn_real_mae,
            "snn_reset_minus_real_mae": snn_reset_mae - snn_real_mae,
            "snn_next_only_minus_contrastive_real_mae": next_only_mae - snn_real_mae,
            "gru_minus_snn_contrastive_real_mae": gru_mae - snn_real_mae,
            "context_free_mlp_minus_snn_contrastive_real_mae": mlp_mae - snn_real_mae,
        },
        "checks": checks,
        "interpretation_boundary": (
            "This report evaluates coarse semantic alignment of internal traces under a controlled fixture. "
            "It does not establish ground-truth emotions, universal neuron meanings, emergent clusters, biological fidelity, or broad real-world generalization."
        ),
    }
    output_path = Path(args.output) if args.output else input_dir / "decision_report.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
