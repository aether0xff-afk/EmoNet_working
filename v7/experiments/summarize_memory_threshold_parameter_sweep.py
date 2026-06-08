"""Summarize OFAT memory-threshold parameter sweep outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


MIN_POSITIVE_RATE = 0.8
MAX_MEMORY_STRENGTH_MEAN_ABS = 0.90
AXES = ("valence", "arousal", "certainty", "social_distance")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="runs/memory_threshold_parameter_sweep_lmstudio")
    parser.add_argument("--baseline", default="runs/trace_semantic_alignment_benchmark_lmstudio")
    parser.add_argument("--output")
    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"sweep result not found: {path}")
    return pd.read_csv(path)


def baseline_mean(frame: pd.DataFrame, model_type: str, column: str) -> float:
    selected = frame.loc[frame["model_type"] == model_type, column]
    if selected.empty:
        raise ValueError(f"missing baseline model rows: {model_type}")
    return float(selected.mean())


def positive_rate(frame: pd.DataFrame, column: str) -> float:
    return float((frame[column] > 0).mean())


def metrics_for(group: pd.DataFrame) -> dict[str, Any]:
    first = group.iloc[0]
    metrics: dict[str, Any] = {
        "config_key": str(first["config_key"]),
        "feedback_strength": float(first["feedback_strength"]),
        "memory_threshold": float(first["memory_threshold"]),
        "accumulation_decay": float(first["accumulation_decay"]),
        "seed_count": int(group["seed"].nunique()),
        "real_targeted_mae": float(group["real_targeted_mae"].mean()),
        "real_targeted_mae_std": float(group["real_targeted_mae"].std(ddof=0)),
        "shuffled_targeted_mae": float(group["shuffled_targeted_mae"].mean()),
        "reset_targeted_mae": float(group["reset_targeted_mae"].mean()),
        "real_direction_accuracy": float(group["real_direction_accuracy"].mean()),
        "real_pair_order_accuracy": float(group["real_pair_order_accuracy"].mean()),
        "objective_context_margin": float(group["objective_context_margin"].mean()),
        "objective_memory_strength_mean_abs": float(group["objective_memory_strength_mean_abs"].mean()),
        "real_minus_text_baseline_positive_rate": positive_rate(group, "real_minus_text_baseline_mae_improvement"),
        "shuffled_degradation_positive_rate": positive_rate(group, "shuffled_history_mae_degradation"),
        "reset_degradation_positive_rate": positive_rate(group, "reset_history_mae_degradation"),
    }
    for axis in AXES:
        metrics[f"real_{axis}_targeted_mae"] = float(group[f"real_{axis}_targeted_mae"].mean())
        metrics[f"real_{axis}_direction_accuracy"] = float(group[f"real_{axis}_direction_accuracy"].mean())
    metrics["shuffled_minus_real_mae"] = metrics["shuffled_targeted_mae"] - metrics["real_targeted_mae"]
    metrics["reset_minus_real_mae"] = metrics["reset_targeted_mae"] - metrics["real_targeted_mae"]
    metrics["seed_stability_is_sufficient"] = (
        metrics["real_minus_text_baseline_positive_rate"] >= MIN_POSITIVE_RATE
        and metrics["shuffled_degradation_positive_rate"] >= MIN_POSITIVE_RATE
        and metrics["reset_degradation_positive_rate"] >= MIN_POSITIVE_RATE
    )
    metrics["memory_strength_is_not_saturated"] = metrics["objective_memory_strength_mean_abs"] < MAX_MEMORY_STRENGTH_MEAN_ABS
    metrics["semantic_checks_pass"] = (
        metrics["real_direction_accuracy"] > 0.5
        and metrics["real_pair_order_accuracy"] > 0.5
        and metrics["shuffled_minus_real_mae"] > 0.0
        and metrics["reset_minus_real_mae"] > 0.0
        and metrics["seed_stability_is_sufficient"]
        and metrics["memory_strength_is_not_saturated"]
    )
    return metrics


def summarize_families(config_metrics: dict[str, dict[str, Any]], references: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
    families: dict[str, list[dict[str, Any]]] = {}
    for family, rows in references.groupby("family"):
        records: list[dict[str, Any]] = []
        for _, row in rows.sort_values("value").iterrows():
            metrics = dict(config_metrics[str(row["config_key"])])
            metrics["swept_parameter"] = str(family)
            metrics["swept_value"] = float(row["value"])
            records.append(metrics)
        families[str(family)] = records
    return families


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input)
    baseline_dir = Path(args.baseline)
    frame = read_csv(input_dir / "by_seed_config.csv")
    references = read_csv(input_dir / "sweep_references.csv")
    baseline = read_csv(baseline_dir / "by_seed_model.csv")

    config_metrics = {
        str(config_key): metrics_for(group)
        for config_key, group in frame.groupby("config_key")
    }
    baseline_snn_mae = baseline_mean(baseline, "snn_context_contrastive", "real_targeted_mae")
    baseline_gru_mae = baseline_mean(baseline, "gru_context_contrastive", "real_targeted_mae")
    for metrics in config_metrics.values():
        metrics["beats_previous_contrastive_snn_mae"] = metrics["real_targeted_mae"] < baseline_snn_mae
        metrics["is_not_worse_than_gru_mae"] = metrics["real_targeted_mae"] <= baseline_gru_mae

    ranked = sorted(config_metrics.values(), key=lambda item: (item["real_targeted_mae"], item["real_targeted_mae_std"]))
    eligible = [item for item in ranked if item["semantic_checks_pass"]]
    best = eligible[0] if eligible else ranked[0]
    families = summarize_families(config_metrics, references)

    stable_regions: dict[str, list[float]] = {}
    for family, records in families.items():
        stable_regions[family] = [
            float(record["swept_value"])
            for record in records
            if record["semantic_checks_pass"] and record["beats_previous_contrastive_snn_mae"]
        ]

    report = {
        "input": str(input_dir),
        "baseline": str(baseline_dir),
        "seed_count": int(frame["seed"].nunique()),
        "unique_config_count": int(frame["config_key"].nunique()),
        "baseline_means": {
            "snn_context_contrastive_real_targeted_mae": baseline_snn_mae,
            "gru_context_contrastive_real_targeted_mae": baseline_gru_mae,
        },
        "best_config": best,
        "best_config_comparisons": {
            "previous_snn_minus_best_mae": baseline_snn_mae - float(best["real_targeted_mae"]),
            "gru_minus_best_mae": baseline_gru_mae - float(best["real_targeted_mae"]),
        },
        "stable_regions": stable_regions,
        "families": families,
        "configs_ranked_by_real_targeted_mae": ranked,
        "interpretation_guide": {
            "stable_region": "Values listed here pass semantic direction, pair-order, shuffled/reset degradation, seed-stability, and non-saturation checks while improving over the previous contrastive SNN.",
            "best_config": "Lowest-MAE eligible configuration. Treat this as a candidate operating point, not a universal optimum.",
            "next_step": "Re-run context-structure validation using the selected configuration, then begin the adjacency-based emergent-cluster benchmark."
        },
        "interpretation_boundary": (
            "This OFAT sweep identifies a stable parameter region for a controlled neuron-memory-threshold SNN fixture. "
            "It does not establish ground-truth emotions, biological fidelity, emergent clusters, or broad real-world generalization."
        ),
    }
    output_path = Path(args.output) if args.output else input_dir / "decision_report.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
