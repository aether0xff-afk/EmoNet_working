"""Summarize neuron-memory-threshold semantic benchmark outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


MIN_POSITIVE_RATE = 0.8
MODEL_TYPES = ("snn_memory_readout_only", "snn_memory_feedback")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="runs/memory_threshold_semantic_benchmark_lmstudio")
    parser.add_argument("--baseline", default="runs/trace_semantic_alignment_benchmark_lmstudio")
    parser.add_argument("--output")
    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"benchmark result not found: {path}")
    return pd.read_csv(path)


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


def baseline_mean(frame: pd.DataFrame, model_type: str, column: str) -> float:
    selected = frame.loc[frame["model_type"] == model_type, column]
    if selected.empty:
        raise ValueError(f"missing baseline rows for {model_type}")
    return float(selected.mean())


def metrics_for(frame: pd.DataFrame, model_type: str) -> dict[str, float]:
    return {
        "real_targeted_mae": mean_for(frame, model_type, "real_targeted_mae"),
        "shuffled_targeted_mae": mean_for(frame, model_type, "shuffled_targeted_mae"),
        "reset_targeted_mae": mean_for(frame, model_type, "reset_targeted_mae"),
        "current_text_baseline_targeted_mae": mean_for(frame, model_type, "current_text_baseline_targeted_mae"),
        "constant_baseline_targeted_mae": mean_for(frame, model_type, "constant_baseline_targeted_mae"),
        "real_direction_accuracy": mean_for(frame, model_type, "real_direction_accuracy"),
        "real_pair_order_accuracy": mean_for(frame, model_type, "real_pair_order_accuracy"),
        "objective_context_margin": mean_for(frame, model_type, "objective_context_margin"),
        "objective_memory_strength_mean_abs": mean_for(frame, model_type, "objective_memory_strength_mean_abs"),
        "real_minus_text_baseline_positive_rate": positive_rate(frame, model_type, "real_minus_text_baseline_mae_improvement"),
        "shuffled_degradation_positive_rate": positive_rate(frame, model_type, "shuffled_history_mae_degradation"),
        "reset_degradation_positive_rate": positive_rate(frame, model_type, "reset_history_mae_degradation"),
    }


def checks_for(metrics: dict[str, float], *, baseline_snn_mae: float, gru_mae: float) -> dict[str, bool]:
    real = metrics["real_targeted_mae"]
    return {
        "beats_previous_contrastive_snn_mae": real < baseline_snn_mae,
        "beats_current_text_baseline": real < metrics["current_text_baseline_targeted_mae"],
        "beats_constant_baseline": real < metrics["constant_baseline_targeted_mae"],
        "semantics_degrade_when_history_is_shuffled": metrics["shuffled_targeted_mae"] > real,
        "semantics_degrade_when_history_is_reset": metrics["reset_targeted_mae"] > real,
        "direction_accuracy_is_above_chance": metrics["real_direction_accuracy"] > 0.5,
        "pair_order_accuracy_is_above_chance": metrics["real_pair_order_accuracy"] > 0.5,
        "seed_stability_is_sufficient": (
            metrics["real_minus_text_baseline_positive_rate"] >= MIN_POSITIVE_RATE
            and metrics["shuffled_degradation_positive_rate"] >= MIN_POSITIVE_RATE
            and metrics["reset_degradation_positive_rate"] >= MIN_POSITIVE_RATE
        ),
        "is_not_worse_than_gru_mae": real <= gru_mae,
    }


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input)
    baseline_dir = Path(args.baseline)
    frame = read_csv(input_dir / "by_seed_model.csv")
    baseline = read_csv(baseline_dir / "by_seed_model.csv")
    missing = sorted(set(MODEL_TYPES) - set(frame["model_type"].unique()))
    if missing:
        raise ValueError(f"missing memory-threshold model rows: {missing}")

    baseline_snn_mae = baseline_mean(baseline, "snn_context_contrastive", "real_targeted_mae")
    baseline_next_only_mae = baseline_mean(baseline, "snn_next_only", "real_targeted_mae")
    baseline_gru_mae = baseline_mean(baseline, "gru_context_contrastive", "real_targeted_mae")
    model_metrics = {model_type: metrics_for(frame, model_type) for model_type in MODEL_TYPES}
    checks = {
        model_type: checks_for(metrics, baseline_snn_mae=baseline_snn_mae, gru_mae=baseline_gru_mae)
        for model_type, metrics in model_metrics.items()
    }
    best_model = min(MODEL_TYPES, key=lambda model_type: model_metrics[model_type]["real_targeted_mae"])
    best_metrics = model_metrics[best_model]
    best_checks = checks[best_model]
    stage_established = all(
        best_checks[key]
        for key in (
            "beats_previous_contrastive_snn_mae",
            "beats_current_text_baseline",
            "semantics_degrade_when_history_is_shuffled",
            "semantics_degrade_when_history_is_reset",
            "direction_accuracy_is_above_chance",
            "pair_order_accuracy_is_above_chance",
            "seed_stability_is_sufficient",
        )
    )

    report = {
        "input": str(input_dir),
        "baseline": str(baseline_dir),
        "seed_count": int(frame["seed"].nunique()),
        "stage_verdict": "established" if stage_established else "not_established",
        "best_memory_threshold_model": best_model,
        "baseline_means": {
            "snn_next_only_real_targeted_mae": baseline_next_only_mae,
            "snn_context_contrastive_real_targeted_mae": baseline_snn_mae,
            "gru_context_contrastive_real_targeted_mae": baseline_gru_mae,
        },
        "model_means": model_metrics,
        "comparisons_for_best_model": {
            "previous_snn_minus_best_memory_model_mae": baseline_snn_mae - best_metrics["real_targeted_mae"],
            "next_only_snn_minus_best_memory_model_mae": baseline_next_only_mae - best_metrics["real_targeted_mae"],
            "gru_minus_best_memory_model_mae": baseline_gru_mae - best_metrics["real_targeted_mae"],
            "text_baseline_minus_best_memory_model_mae": best_metrics["current_text_baseline_targeted_mae"] - best_metrics["real_targeted_mae"],
            "shuffled_minus_real_mae": best_metrics["shuffled_targeted_mae"] - best_metrics["real_targeted_mae"],
            "reset_minus_real_mae": best_metrics["reset_targeted_mae"] - best_metrics["real_targeted_mae"],
        },
        "checks_by_model": checks,
        "interpretation_boundary": (
            "This benchmark evaluates whether neuron-local accumulation and a separate memory threshold improve controlled semantic readability. "
            "It does not establish ground-truth emotions, biological fidelity, emergent clusters, or broad real-world generalization."
        ),
    }
    output_path = Path(args.output) if args.output else input_dir / "decision_report.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
