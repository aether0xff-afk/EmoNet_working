"""Summarize activity-guided rewiring stability OFAT sweep outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


BASELINE_MEMORY_CONFIG_KEY = "feedback_0.050__threshold_0.500__accumulation_decay_0.850"
MIN_POSITIVE_RATE = 0.8
MAE_REGRESSION_TOLERANCE = 0.005


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="runs/activity_guided_rewiring_stability_sweep_lmstudio")
    parser.add_argument("--baseline", default="runs/memory_threshold_parameter_sweep_lmstudio")
    parser.add_argument("--baseline-config-key", default=BASELINE_MEMORY_CONFIG_KEY)
    parser.add_argument("--output")
    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"sweep result not found: {path}")
    return pd.read_csv(path)


def positive_rate(frame: pd.DataFrame, column: str) -> float:
    return float((frame[column] > 0).mean())


def metrics_for(group: pd.DataFrame, *, baseline_mae: float) -> dict[str, Any]:
    first = group.iloc[0]
    real_mae = float(group["real_targeted_mae"].mean())
    shuffled_mae = float(group["shuffled_targeted_mae"].mean())
    reset_mae = float(group["reset_targeted_mae"].mean())
    rewiring_events = float(group["rewiring_event_count"].mean())
    rewired_edges = float(group["rewired_edges_total"].mean())
    rates = {
        "real_minus_text_baseline_mae_improvement": positive_rate(group, "real_minus_text_baseline_mae_improvement"),
        "shuffled_history_mae_degradation": positive_rate(group, "shuffled_history_mae_degradation"),
        "reset_history_mae_degradation": positive_rate(group, "reset_history_mae_degradation"),
    }
    metrics: dict[str, Any] = {
        "config_key": str(first["config_key"]),
        "rewiring_fraction": float(first["rewiring_fraction"]),
        "rewiring_start_epoch": int(first["rewiring_start_epoch"]),
        "rewiring_interval": int(first["rewiring_interval"]),
        "seed_count": int(group["seed"].nunique()),
        "real_targeted_mae": real_mae,
        "real_targeted_mae_std": float(group["real_targeted_mae"].std(ddof=0)),
        "shuffled_targeted_mae": shuffled_mae,
        "reset_targeted_mae": reset_mae,
        "real_direction_accuracy": float(group["real_direction_accuracy"].mean()),
        "real_pair_order_accuracy": float(group["real_pair_order_accuracy"].mean()),
        "objective_memory_strength_mean_abs": float(group["objective_memory_strength_mean_abs"].mean()),
        "rewiring_event_count": rewiring_events,
        "rewired_edges_total": rewired_edges,
        "shuffled_minus_real_mae": shuffled_mae - real_mae,
        "reset_minus_real_mae": reset_mae - real_mae,
        "baseline_minus_rewired_mae": baseline_mae - real_mae,
        **rates,
    }
    metrics["rewiring_was_applied"] = rewiring_events > 0.0 and rewired_edges > 0.0
    metrics["semantic_mae_does_not_regress_materially"] = real_mae <= baseline_mae + MAE_REGRESSION_TOLERANCE
    metrics["seed_stability_is_sufficient"] = all(value >= MIN_POSITIVE_RATE for value in rates.values())
    metrics["memory_strength_is_not_saturated"] = metrics["objective_memory_strength_mean_abs"] < 0.90
    metrics["semantic_preservation_checks_pass"] = (
        metrics["rewiring_was_applied"]
        and metrics["semantic_mae_does_not_regress_materially"]
        and metrics["shuffled_minus_real_mae"] > 0.0
        and metrics["reset_minus_real_mae"] > 0.0
        and metrics["real_direction_accuracy"] > 0.5
        and metrics["real_pair_order_accuracy"] > 0.5
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
    baseline = read_csv(baseline_dir / "by_seed_config.csv")
    selected = baseline.loc[baseline["config_key"] == args.baseline_config_key]
    if selected.empty:
        raise ValueError(f"missing baseline config rows: {args.baseline_config_key}")
    baseline_mae = float(selected["real_targeted_mae"].mean())

    config_metrics = {
        str(config_key): metrics_for(group, baseline_mae=baseline_mae)
        for config_key, group in frame.groupby("config_key")
    }
    ranked = sorted(config_metrics.values(), key=lambda item: (item["real_targeted_mae"], item["real_targeted_mae_std"]))
    eligible = [item for item in ranked if item["semantic_preservation_checks_pass"]]
    best = eligible[0] if eligible else None
    families = summarize_families(config_metrics, references)
    stable_regions = {
        family: [record["swept_value"] for record in records if record["semantic_preservation_checks_pass"]]
        for family, records in families.items()
    }

    report = {
        "input": str(input_dir),
        "baseline": str(baseline_dir),
        "baseline_config_key": args.baseline_config_key,
        "baseline_memory_model_real_targeted_mae": baseline_mae,
        "seed_count": int(frame["seed"].nunique()),
        "unique_config_count": int(frame["config_key"].nunique()),
        "stage_verdict": "semantic_preserving_rewiring_region_found" if best is not None else "semantic_preserving_rewiring_region_not_found",
        "best_semantic_preserving_rewiring_config": best,
        "stable_regions": stable_regions,
        "families": families,
        "configs_ranked_by_real_targeted_mae": ranked,
        "next_step": (
            "Use the best semantic-preserving rewiring config for the rewired emergent-cluster diagnostic. "
            "If no eligible config exists, reduce topology changes further or revise the rewiring rule before testing communities."
        ),
        "interpretation_boundary": (
            "This OFAT sweep searches for a semantic-preserving activity-guided rewiring regime under a controlled fixture. "
            "It does not establish emergent communities, final rewiring rules, emotional ground truth, or biological fidelity."
        ),
    }
    output_path = Path(args.output) if args.output else input_dir / "decision_report.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
