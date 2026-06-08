"""Summarize selected memory-threshold context-structure benchmark outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


FLOAT_TOLERANCE = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="runs/memory_threshold_context_structure_best_lmstudio")
    parser.add_argument("--output")
    return parser.parse_args()


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
    csv_path = input_dir / "by_seed_model.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"benchmark result not found: {csv_path}")
    frame = pd.read_csv(csv_path)
    metric = "cosine"
    context_gap = mean(frame, f"trace_context_gap_{metric}")
    reset_gap = mean(frame, f"trace_reset_gap_{metric}")
    repeat_distance = mean(frame, f"same_context_trace_distance_{metric}")
    shuffled_distance = mean(frame, f"real_vs_shuffled_trace_distance_{metric}")
    reset_distance = mean(frame, f"real_vs_reset_trace_distance_{metric}")
    retrieval = mean(frame, f"context_retrieval_accuracy_{metric}")
    probe = mean(frame, "linear_probe_accuracy")
    chance = mean(frame, "linear_probe_chance_level")

    checks = {
        "memory_threshold_trace_changes_when_history_is_shuffled": context_gap > 0.0,
        "memory_threshold_trace_changes_when_history_is_reset": reset_gap > 0.0,
        "memory_threshold_trace_is_stable_under_exact_repeat": repeat_distance <= FLOAT_TOLERANCE,
        "memory_threshold_context_retrieval_is_above_chance": retrieval > 0.5,
        "memory_threshold_linear_probe_is_above_chance": probe > chance,
        "memory_threshold_context_gap_is_positive_for_all_seeds": positive_rate(frame, f"trace_context_gap_{metric}") == 1.0,
        "memory_threshold_reset_gap_is_positive_for_all_seeds": positive_rate(frame, f"trace_reset_gap_{metric}") == 1.0,
    }
    report = {
        "input": str(input_dir),
        "seed_count": int(frame["seed"].nunique()),
        "stage_verdict": "established" if all(checks.values()) else "not_established",
        "primary_distance": metric,
        "means": {
            "trace_context_gap": context_gap,
            "trace_reset_gap": reset_gap,
            "same_context_repeat_distance": repeat_distance,
            "real_vs_shuffled_distance": shuffled_distance,
            "real_vs_reset_distance": reset_distance,
            "context_retrieval_accuracy": retrieval,
            "linear_probe_accuracy": probe,
            "linear_probe_chance_level": chance,
        },
        "positive_rates": {
            "trace_context_gap": positive_rate(frame, f"trace_context_gap_{metric}"),
            "trace_reset_gap": positive_rate(frame, f"trace_reset_gap_{metric}"),
        },
        "checks": checks,
        "interpretation_boundary": (
            "This report evaluates whether traces from a selected neuron-memory-threshold SNN are stable and context-dependent. "
            "It does not establish emotional ground truth, emergent clusters, or biological fidelity."
        ),
    }
    output_path = Path(args.output) if args.output else input_dir / "decision_report.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
