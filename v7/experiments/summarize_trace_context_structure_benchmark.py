"""Summarize trace-context structure benchmark outputs into a decision report."""

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
STABILITY_TOLERANCE = 1e-8
INVARIANCE_TOLERANCE = 1e-8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="runs/trace_context_structure_benchmark_lmstudio")
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

    metric = "cosine"
    snn_next_gap = mean_for(frame, "snn_next_only", f"trace_context_gap_{metric}")
    snn_contrast_gap = mean_for(frame, "snn_context_contrastive", f"trace_context_gap_{metric}")
    gru_gap = mean_for(frame, "gru_context_contrastive", f"trace_context_gap_{metric}")
    mlp_gap = mean_for(frame, "context_free_mlp", f"trace_context_gap_{metric}")
    snn_repeat = mean_for(frame, "snn_context_contrastive", f"same_context_trace_distance_{metric}")
    mlp_shuffled = mean_for(frame, "context_free_mlp", f"real_vs_shuffled_trace_distance_{metric}")
    mlp_reset = mean_for(frame, "context_free_mlp", f"real_vs_reset_trace_distance_{metric}")
    snn_retrieval = mean_for(frame, "snn_context_contrastive", f"context_retrieval_accuracy_{metric}")
    gru_retrieval = mean_for(frame, "gru_context_contrastive", f"context_retrieval_accuracy_{metric}")
    snn_probe = mean_for(frame, "snn_context_contrastive", "linear_probe_accuracy")
    chance = mean_for(frame, "snn_context_contrastive", "linear_probe_chance_level")

    report = {
        "input": str(input_dir),
        "seed_count": int(frame["seed"].nunique()),
        "primary_distance": metric,
        "means": {
            "snn_next_only_trace_context_gap": snn_next_gap,
            "snn_context_contrastive_trace_context_gap": snn_contrast_gap,
            "context_free_mlp_trace_context_gap": mlp_gap,
            "gru_context_contrastive_trace_context_gap": gru_gap,
            "snn_context_contrastive_same_context_repeat_distance": snn_repeat,
            "context_free_mlp_real_vs_shuffled_distance": mlp_shuffled,
            "context_free_mlp_real_vs_reset_distance": mlp_reset,
            "snn_context_contrastive_retrieval_accuracy": snn_retrieval,
            "gru_context_contrastive_retrieval_accuracy": gru_retrieval,
            "snn_context_contrastive_linear_probe_accuracy": snn_probe,
            "linear_probe_chance_level": chance,
        },
        "positive_rates": {
            "snn_context_contrastive_trace_context_gap": positive_rate(frame, "snn_context_contrastive", f"trace_context_gap_{metric}"),
            "snn_context_contrastive_trace_reset_gap": positive_rate(frame, "snn_context_contrastive", f"trace_reset_gap_{metric}"),
            "snn_context_contrastive_probe_above_chance": float(
                (
                    frame.loc[frame["model_type"] == "snn_context_contrastive", "linear_probe_accuracy"]
                    > frame.loc[frame["model_type"] == "snn_context_contrastive", "linear_probe_chance_level"]
                ).mean()
            ),
        },
        "comparisons": {
            "snn_contrastive_minus_next_only_trace_gap": snn_contrast_gap - snn_next_gap,
            "snn_contrastive_minus_gru_trace_gap": snn_contrast_gap - gru_gap,
            "snn_probe_minus_chance": snn_probe - chance,
        },
        "checks": {
            "contrastive_snn_trace_changes_with_correct_history": snn_contrast_gap > 0,
            "contrastive_snn_trace_is_stable_under_repeat": snn_repeat <= STABILITY_TOLERANCE,
            "contrastive_snn_beats_next_only_trace_gap": snn_contrast_gap > snn_next_gap,
            "context_free_mlp_is_history_invariant": abs(mlp_shuffled) <= INVARIANCE_TOLERANCE and abs(mlp_reset) <= INVARIANCE_TOLERANCE,
            "trace_retrieval_is_above_chance": snn_retrieval > 0.5,
            "linear_probe_is_above_chance": snn_probe > chance,
        },
        "interpretation_boundary": (
            "This report evaluates whether internal traces are stable and context-dependent. "
            "It does not establish emotional semantics, interpretable neuron roles, emergent clusters, or biological fidelity."
        ),
    }
    output_path = Path(args.output) if args.output else input_dir / "decision_report.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
