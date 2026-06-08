"""Summarize activity-guided rewiring semantic benchmark outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


BASELINE_CONFIG_KEY = "feedback_0.050__threshold_0.500__accumulation_decay_0.850"
MIN_POSITIVE_RATE = 0.8
MAE_REGRESSION_TOLERANCE = 0.005


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="runs/activity_guided_rewiring_semantic_benchmark_lmstudio")
    parser.add_argument("--baseline", default="runs/memory_threshold_parameter_sweep_lmstudio")
    parser.add_argument("--baseline-config-key", default=BASELINE_CONFIG_KEY)
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


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input)
    baseline_dir = Path(args.baseline)
    frame = read_csv(input_dir / "by_seed_model.csv")
    baseline = read_csv(baseline_dir / "by_seed_config.csv")
    selected = baseline.loc[baseline["config_key"] == args.baseline_config_key]
    if selected.empty:
        raise ValueError(f"missing baseline config rows: {args.baseline_config_key}")

    rewired_real = mean(frame, "real_targeted_mae")
    rewired_shuffled = mean(frame, "shuffled_targeted_mae")
    rewired_reset = mean(frame, "reset_targeted_mae")
    rewired_direction = mean(frame, "real_direction_accuracy")
    rewired_pair_order = mean(frame, "real_pair_order_accuracy")
    rewired_memory = mean(frame, "objective_memory_strength_mean_abs")
    rewiring_events = mean(frame, "rewiring_event_count")
    rewired_edges = mean(frame, "rewired_edges_total")
    baseline_real = mean(selected, "real_targeted_mae")
    baseline_direction = mean(selected, "real_direction_accuracy")
    baseline_pair_order = mean(selected, "real_pair_order_accuracy")

    rates = {
        "real_minus_text_baseline_mae_improvement": positive_rate(frame, "real_minus_text_baseline_mae_improvement"),
        "shuffled_history_mae_degradation": positive_rate(frame, "shuffled_history_mae_degradation"),
        "reset_history_mae_degradation": positive_rate(frame, "reset_history_mae_degradation"),
    }
    checks = {
        "rewiring_was_applied": rewiring_events > 0.0 and rewired_edges > 0.0,
        "semantic_mae_does_not_regress_materially": rewired_real <= baseline_real + MAE_REGRESSION_TOLERANCE,
        "semantic_mae_improves_over_non_rewired_baseline": rewired_real < baseline_real,
        "semantics_degrade_when_history_is_shuffled": rewired_shuffled > rewired_real,
        "semantics_degrade_when_history_is_reset": rewired_reset > rewired_real,
        "direction_accuracy_is_above_chance": rewired_direction > 0.5,
        "pair_order_accuracy_is_above_chance": rewired_pair_order > 0.5,
        "seed_stability_is_sufficient": all(value >= MIN_POSITIVE_RATE for value in rates.values()),
        "memory_strength_is_not_saturated": rewired_memory < 0.90,
    }
    preservation_keys = (
        "rewiring_was_applied",
        "semantic_mae_does_not_regress_materially",
        "semantics_degrade_when_history_is_shuffled",
        "semantics_degrade_when_history_is_reset",
        "direction_accuracy_is_above_chance",
        "pair_order_accuracy_is_above_chance",
        "seed_stability_is_sufficient",
        "memory_strength_is_not_saturated",
    )
    verdict = "rewiring_semantics_preserved" if all(checks[key] for key in preservation_keys) else "rewiring_semantics_regressed"

    report = {
        "input": str(input_dir),
        "baseline": str(baseline_dir),
        "baseline_config_key": args.baseline_config_key,
        "seed_count": int(frame["seed"].nunique()),
        "stage_verdict": verdict,
        "means": {
            "rewired_real_targeted_mae": rewired_real,
            "rewired_shuffled_targeted_mae": rewired_shuffled,
            "rewired_reset_targeted_mae": rewired_reset,
            "rewired_real_direction_accuracy": rewired_direction,
            "rewired_real_pair_order_accuracy": rewired_pair_order,
            "rewired_memory_strength_mean_abs": rewired_memory,
            "rewiring_event_count": rewiring_events,
            "rewired_edges_total": rewired_edges,
            "baseline_real_targeted_mae": baseline_real,
            "baseline_real_direction_accuracy": baseline_direction,
            "baseline_real_pair_order_accuracy": baseline_pair_order,
        },
        "comparisons": {
            "baseline_minus_rewired_mae": baseline_real - rewired_real,
            "rewired_minus_baseline_direction_accuracy": rewired_direction - baseline_direction,
            "rewired_minus_baseline_pair_order_accuracy": rewired_pair_order - baseline_pair_order,
            "shuffled_minus_real_mae": rewired_shuffled - rewired_real,
            "reset_minus_real_mae": rewired_reset - rewired_real,
        },
        "positive_rates": rates,
        "checks": checks,
        "next_step": (
            "If stage_verdict is rewiring_semantics_preserved, run the rewired adjacency-community diagnostic. "
            "If semantic MAE materially regresses, reduce rewiring fraction or interval before testing clusters."
        ),
        "interpretation_boundary": (
            "This benchmark tests whether activity-guided topology changes preserve controlled semantic readability. "
            "It does not establish emergent clusters, final rewiring rules, emotional ground truth, or biological fidelity."
        ),
    }
    output_path = Path(args.output) if args.output else input_dir / "decision_report.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
