#!/usr/bin/env python3
"""Sparsity-preserving dynamics sweep for neural trace stabilization.

This sweep starts after the first stabilization pass showed that collapse can
be eliminated, but at the cost of excessive activation density.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import export_neural_activation_traces as exporter  # noqa: E402
import probe_neural_trace_geometry as geometry  # noqa: E402


TRACKED_AXES = ["valence", "social_orientation", "action_tendency_class", "appraisal_family", "control_state"]


BASE_PARAMS = {
    "k_threshold_base": 0.66,
    "k_remem_base": 0.84,
    "k_decay": 0.99,
    "refractory_ticks": 1,
    "input_topk": 3,
    "input_signal_clip": 1.50,
    "intrinsic_alignment_gain": 0.24,
    "fatigue_gain": 0.18,
    "fatigue_threshold_gain": 0.10,
    "fatigue_k_leak": 0.05,
    "inhibitory_suppression_gain": 0.16,
    "ne_thresh_reduce_gain": 0.25,
    "ne_remem_reduce_gain": 0.25,
    "activity_churn_eps": 0.02,
}


def grid_configs() -> list[dict[str, Any]]:
    thresholds = [0.64, 0.67, 0.70]
    input_topks = [2, 3]
    input_clips = [1.20, 1.50, 1.80]
    inhibitions = [0.12, 0.18]
    fatigue_profiles = [
        {"name": "midfat", "fatigue_gain": 0.18, "fatigue_threshold_gain": 0.10, "fatigue_k_leak": 0.05},
        {"name": "highfat", "fatigue_gain": 0.24, "fatigue_threshold_gain": 0.14, "fatigue_k_leak": 0.07},
    ]
    configs: list[dict[str, Any]] = []
    for threshold, topk, clip, inhibition, fatigue in itertools.product(
        thresholds,
        input_topks,
        input_clips,
        inhibitions,
        fatigue_profiles,
    ):
        params = dict(BASE_PARAMS)
        params.update(
            {
                "k_threshold_base": threshold,
                "k_remem_base": threshold + 0.18,
                "input_topk": topk,
                "input_signal_clip": clip,
                "inhibitory_suppression_gain": inhibition,
                "fatigue_gain": fatigue["fatigue_gain"],
                "fatigue_threshold_gain": fatigue["fatigue_threshold_gain"],
                "fatigue_k_leak": fatigue["fatigue_k_leak"],
            }
        )
        configs.append(
            {
                "name": f"thr{threshold:.2f}_topk{topk}_clip{clip:.1f}_inh{inhibition:.2f}_{fatigue['name']}",
                "params": params,
            }
        )
    return configs


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def export_args(args: argparse.Namespace, config: dict[str, Any], out_dir: Path) -> SimpleNamespace:
    params = config["params"]
    return SimpleNamespace(
        input=args.input,
        output_dir=out_dir,
        limit=args.limit,
        n_neurons=args.n_neurons,
        seed=args.seed,
        z_encoder_mode="stat",
        stim_source=args.stim_source,
        max_ticks=args.max_ticks,
        min_ticks_before_converged=args.min_ticks_before_converged,
        convergence_patience=args.convergence_patience,
        progress_every=0,
        **params,
    )


def geometry_args(out_dir: Path) -> SimpleNamespace:
    return SimpleNamespace(
        summary_csv=out_dir / "neural_trace_summary.csv",
        trace_dir=out_dir / "traces_npz",
        feature_kind="branch_mean",
        output=out_dir / "neural_trace_geometry_branch.json",
        limit=None,
    )


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def density_penalty(density: float, low: float = 0.55, high: float = 0.80) -> float:
    if low <= density <= high:
        return 0.0
    if density < low:
        return (low - density) * 1.2
    return (density - high) * 3.0


def score_candidate(report: dict[str, Any]) -> dict[str, Any]:
    branch = report["branch_health"]
    nn = report["nearest_neighbor"]
    gd = report["group_distances"]
    lifts = [float(nn[axis]["lift"]) for axis in TRACKED_AXES]
    separations = [float(gd[axis]["separation"]) for axis in TRACKED_AXES]
    len1 = float(branch["len1_ratio"])
    density = float(branch["mean_activation_density"])
    branch_len = float(branch["mean_dominant_branch_len"])
    sep = mean(separations)
    lift = mean(lifts)

    branch_len_bonus = min(branch_len, 32.0) / 32.0
    objective = (
        3.5 * sep
        + 1.0 * lift
        + 0.35 * branch_len_bonus
        - 5.0 * len1
        - density_penalty(density)
    )
    return {
        "mean_branch_len": round(branch_len, 6),
        "len1_ratio": round(len1, 6),
        "mean_activation_density": round(density, 6),
        "tracked_lift_mean": round(lift, 6),
        "tracked_separation_mean": round(sep, 6),
        "objective": round(objective, 6),
        **{f"{axis}_lift": float(nn[axis]["lift"]) for axis in TRACKED_AXES},
        **{f"{axis}_separation": float(gd[axis]["separation"]) for axis in TRACKED_AXES},
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    configs = grid_configs()
    for idx, config in enumerate(configs, start=1):
        print(f"sparse-sweep: {idx}/{len(configs)} {config['name']}")
        out_dir = args.output_dir / config["name"]
        manifest = exporter.run(export_args(args, config, out_dir))
        report = geometry.run(geometry_args(out_dir))
        row = {
            "config": config["name"],
            "ok_rows": manifest["ok_rows"],
            "error_rows": manifest["error_rows"],
            **score_candidate(report),
            "params_json": json.dumps(config["params"], ensure_ascii=False, sort_keys=True),
        }
        rows.append(row)

    rows.sort(key=lambda row: float(row["objective"]), reverse=True)
    fieldnames = [
        "config",
        "ok_rows",
        "error_rows",
        "objective",
        "mean_branch_len",
        "len1_ratio",
        "mean_activation_density",
        "tracked_lift_mean",
        "tracked_separation_mean",
        *[f"{axis}_lift" for axis in TRACKED_AXES],
        *[f"{axis}_separation" for axis in TRACKED_AXES],
        "params_json",
    ]
    summary_csv = args.output_dir / "sparse_dynamics_sweep_summary.csv"
    write_csv(summary_csv, rows, fieldnames)
    payload = {
        "output_dir": str(args.output_dir),
        "summary_csv": str(summary_csv),
        "limit": args.limit,
        "n_neurons": args.n_neurons,
        "tracked_axes": TRACKED_AXES,
        "density_target": [0.55, 0.80],
        "configs": len(configs),
        "top_configs": rows[: min(10, len(rows))],
    }
    (args.output_dir / "sparse_dynamics_sweep_manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("outputs/targeted_records_trace_normalized.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/neural_trace_dynamics_sparse_sweep_v1"))
    parser.add_argument("--limit", type=int, default=24)
    parser.add_argument("--n-neurons", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stim-source", choices=["auto", "text", "proxy"], default="auto")
    parser.add_argument("--max-ticks", type=int, default=64)
    parser.add_argument("--min-ticks-before-converged", type=int, default=6)
    parser.add_argument("--convergence-patience", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run(args)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

