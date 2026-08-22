#!/usr/bin/env python3
"""Run one v3.1 TRACE mechanism condition across all frozen full80 samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import analyze_trace_mechanism as base
from analyze_trace_mechanism_fast import run_sample_fast


base.run_sample = run_sample_fast


def find_condition(name: str):
    if name == "shuffled_stimulus":
        return next(c for c in base.CONDITIONS if c.name == "baseline")
    for condition in base.CONDITIONS:
        if condition.name == name:
            return condition
    raise ValueError(f"unknown condition: {name}")


def run(args: argparse.Namespace) -> dict:
    rows = base.read_rows(args.summary_csv)
    dynamics = base.load_dynamics(args.config)
    condition = find_condition(args.condition)

    donor_indices = np.arange(len(rows))
    if args.condition == "shuffled_stimulus":
        donor_indices = np.random.default_rng(args.seed + 991).permutation(len(rows))

    features = []
    tick_counts = []
    densities = []
    for idx, row in enumerate(rows):
        run_row = dict(row)
        if args.condition == "shuffled_stimulus":
            donor = base.stimulus(rows[int(donor_indices[idx])])
            for name, value in zip(base.STIM_COLUMNS, donor, strict=False):
                run_row[name] = str(float(value))
        feature, summary, _ = run_sample_fast(run_row, condition, dynamics, args.seed)
        features.append(feature)
        tick_counts.append(summary["ticks"])
        densities.append(summary["mean_density"])
        if (idx + 1) % 20 == 0:
            print(f"[{args.condition}] {idx + 1}/{len(rows)}")

    matrix = np.stack(features, axis=0)
    distance = base.standardized_distance_matrix(matrix)
    metrics = {axis: base.axis_metric(rows, distance, axis) for axis in base.LABEL_AXES}
    result = {
        "condition": args.condition,
        "n_samples": len(rows),
        "seed": args.seed,
        "mean_ticks": float(np.mean(tick_counts)),
        "mean_density": float(np.mean(densities)),
        "metrics": metrics,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / f"condition_{args.condition}.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", required=True)
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("v3.1/outputs/neural_trace_final_candidate_thr064_topk2_clip15_inh018_full80/neural_trace_summary.csv"),
    )
    parser.add_argument("--config", type=Path, default=Path("v3.1/configs/final_dynamics_v1.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("v3.1/outputs/trace_mechanism_matrix"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
