"""Compare history-reconstructive SNN results with the previous semantic benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


MIN_POSITIVE_RATE = 0.8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="runs/history_reconstructive_snn_benchmark_lmstudio")
    parser.add_argument("--baseline", default="runs/trace_semantic_alignment_benchmark_lmstudio")
    parser.add_argument("--output")
    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"benchmark result not found: {path}")
    return pd.read_csv(path)


def mean(frame: pd.DataFrame, column: str) -> float:
    if column not in frame.columns:
        raise ValueError(f"missing benchmark column: {column}")
    return float(frame[column].mean())


def positive_rate(frame: pd.DataFrame, column: str) -> float:
    if column not in frame.columns:
        raise ValueError(f"missing benchmark column: {column}")
    return float((frame[column] > 0).mean())


def baseline_mean(frame: pd.DataFrame, model_type: str, column: str) -> float:
    selected = frame.loc[frame["model_type"] == model_type]
    if selected.empty:
        raise ValueError(f"missing baseline model rows: {model_type}")
    return mean(selected, column)


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input)
    baseline_dir = Path(args.baseline)
    frame = read_csv(input_dir / "by_seed_model.csv")
    baseline = read_csv(baseline_dir / "by_seed_model.csv")

    real_mae = mean(frame, "real_targeted_mae")
    shuffled_mae = mean(frame, "shuffled_targeted_mae")
    reset_mae = mean(frame, "reset_targeted_mae")
    text_mae = mean(frame, "current_text_baseline_targeted_mae")
    constant_mae = mean(frame, "constant_baseline_targeted_mae")
    direction = mean(frame, "real_direction_accuracy")
    pair_order = mean(frame, "real_pair_order_accuracy")

    baseline_snn_mae = baseline_mean(baseline, "snn_context_contrastive", "real_targeted_mae")
    baseline_next_only_mae = baseline_mean(baseline, "snn_next_only", "real_targeted_mae")
    baseline_gru_mae = baseline_mean(baseline, "gru_context_contrastive", "real_targeted_mae")

    rate_vs_text = positive_rate(frame, "real_minus_text_baseline_mae_improvement")
    rate_vs_constant = positive_rate(frame, "real_minus_constant_mae_improvement")
    rate_shuffle = positive_rate(frame, "shuffled_history_mae_degradation")
    rate_reset = positive_rate(frame, "reset_history_mae_degradation")

    checks = {
        "history_reconstructive_snn_beats_previous_contrastive_snn_mae": real_mae < baseline_snn_mae,
        "history_reconstructive_snn_beats_next_only_snn_mae": real_mae < baseline_next_only_mae,
        "history_reconstructive_snn_beats_current_text_baseline": real_mae < text_mae,
        "history_reconstructive_snn_beats_constant_baseline": real_mae < constant_mae,
        "history_reconstructive_snn_semantics_degrade_when_history_is_shuffled": shuffled_mae > real_mae,
        "history_reconstructive_snn_semantics_degrade_when_history_is_reset": reset_mae > real_mae,
        "history_reconstructive_snn_direction_accuracy_is_above_chance": direction > 0.5,
        "history_reconstructive_snn_pair_order_is_above_chance": pair_order > 0.5,
        "history_reconstructive_snn_seed_stability_is_sufficient": (
            rate_vs_text >= MIN_POSITIVE_RATE
            and rate_vs_constant >= MIN_POSITIVE_RATE
            and rate_shuffle >= MIN_POSITIVE_RATE
            and rate_reset >= MIN_POSITIVE_RATE
        ),
        "history_reconstructive_snn_is_not_worse_than_gru_mae": real_mae <= baseline_gru_mae,
    }
    established = all(
        checks[key]
        for key in (
            "history_reconstructive_snn_beats_previous_contrastive_snn_mae",
            "history_reconstructive_snn_beats_current_text_baseline",
            "history_reconstructive_snn_semantics_degrade_when_history_is_shuffled",
            "history_reconstructive_snn_semantics_degrade_when_history_is_reset",
            "history_reconstructive_snn_direction_accuracy_is_above_chance",
            "history_reconstructive_snn_pair_order_is_above_chance",
            "history_reconstructive_snn_seed_stability_is_sufficient",
        )
    )

    report = {
        "input": str(input_dir),
        "baseline": str(baseline_dir),
        "seed_count": int(frame["seed"].nunique()),
        "stage_verdict": "established" if established else "not_established",
        "means": {
            "history_reconstructive_snn_real_targeted_mae": real_mae,
            "history_reconstructive_snn_shuffled_targeted_mae": shuffled_mae,
            "history_reconstructive_snn_reset_targeted_mae": reset_mae,
            "history_reconstructive_snn_current_text_baseline_targeted_mae": text_mae,
            "history_reconstructive_snn_constant_baseline_targeted_mae": constant_mae,
            "history_reconstructive_snn_real_direction_accuracy": direction,
            "history_reconstructive_snn_real_pair_order_accuracy": pair_order,
            "previous_snn_context_contrastive_real_targeted_mae": baseline_snn_mae,
            "previous_snn_next_only_real_targeted_mae": baseline_next_only_mae,
            "previous_gru_context_contrastive_real_targeted_mae": baseline_gru_mae,
        },
        "positive_rates": {
            "real_minus_text_baseline_mae_improvement": rate_vs_text,
            "real_minus_constant_mae_improvement": rate_vs_constant,
            "shuffled_history_mae_degradation": rate_shuffle,
            "reset_history_mae_degradation": rate_reset,
        },
        "comparisons": {
            "previous_snn_minus_history_reconstructive_snn_mae": baseline_snn_mae - real_mae,
            "next_only_snn_minus_history_reconstructive_snn_mae": baseline_next_only_mae - real_mae,
            "gru_minus_history_reconstructive_snn_mae": baseline_gru_mae - real_mae,
            "text_baseline_minus_history_reconstructive_snn_mae": text_mae - real_mae,
            "constant_baseline_minus_history_reconstructive_snn_mae": constant_mae - real_mae,
            "shuffled_minus_real_mae": shuffled_mae - real_mae,
            "reset_minus_real_mae": reset_mae - real_mae,
        },
        "checks": checks,
        "interpretation_boundary": (
            "This experiment tests whether a label-free prior-event reconstruction objective improves semantic readability of SNN traces. "
            "It does not establish ground-truth emotions, biological fidelity, emergent clusters, or broad real-world generalization."
        ),
    }
    output_path = Path(args.output) if args.output else input_dir / "decision_report.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
