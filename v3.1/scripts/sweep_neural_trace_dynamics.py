#!/usr/bin/env python3
"""Small dynamics sweep for stabilizing neural activation traces."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import export_neural_activation_traces as exporter  # noqa: E402
import probe_neural_trace_geometry as geometry  # noqa: E402


BASE_DYNAMICS = {
    "k_threshold_base": 0.72,
    "k_remem_base": 0.95,
    "k_decay": 0.99,
    "refractory_ticks": 1,
    "input_topk": 2,
    "input_signal_clip": 1.50,
    "intrinsic_alignment_gain": 0.24,
    "fatigue_gain": 0.30,
    "fatigue_threshold_gain": 0.18,
    "fatigue_k_leak": 0.08,
    "inhibitory_suppression_gain": 0.18,
    "ne_thresh_reduce_gain": 0.25,
    "ne_remem_reduce_gain": 0.25,
    "activity_churn_eps": 0.02,
}


SWEEP_CONFIGS: list[dict[str, Any]] = [
    {
        "name": "baseline",
        "hypothesis": "current reference",
        "params": {},
    },
    {
        "name": "lower_threshold",
        "hypothesis": "help weak stimuli cross initial activation threshold",
        "params": {"k_threshold_base": 0.58, "k_remem_base": 0.80},
    },
    {
        "name": "stronger_input",
        "hypothesis": "let more upstream signal reach downstream neurons",
        "params": {"input_topk": 4, "input_signal_clip": 2.40},
    },
    {
        "name": "less_inhibition",
        "hypothesis": "reduce early suppression of competing active nodes",
        "params": {"inhibitory_suppression_gain": 0.06},
    },
    {
        "name": "less_fatigue",
        "hypothesis": "avoid killing active routes before a trace unfolds",
        "params": {"fatigue_gain": 0.10, "fatigue_threshold_gain": 0.05, "fatigue_k_leak": 0.02},
    },
    {
        "name": "persistent_flow",
        "hypothesis": "combine lower threshold, stronger input, and less fatigue",
        "params": {
            "k_threshold_base": 0.58,
            "k_remem_base": 0.80,
            "input_topk": 4,
            "input_signal_clip": 2.40,
            "fatigue_gain": 0.10,
            "fatigue_threshold_gain": 0.05,
            "fatigue_k_leak": 0.02,
        },
    },
    {
        "name": "persistent_less_inhibition",
        "hypothesis": "persistent flow with weaker lateral suppression",
        "params": {
            "k_threshold_base": 0.58,
            "k_remem_base": 0.80,
            "input_topk": 4,
            "input_signal_clip": 2.40,
            "fatigue_gain": 0.10,
            "fatigue_threshold_gain": 0.05,
            "fatigue_k_leak": 0.02,
            "inhibitory_suppression_gain": 0.06,
        },
    },
    {
        "name": "high_ne_modulation",
        "hypothesis": "norepinephrine-like modulation lowers thresholds during active traces",
        "params": {"ne_thresh_reduce_gain": 0.45, "ne_remem_reduce_gain": 0.40},
    },
]


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def export_args(args: argparse.Namespace, config: dict[str, Any], out_dir: Path) -> SimpleNamespace:
    params = {**BASE_DYNAMICS, **config["params"]}
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


def geometry_args(out_dir: Path, feature_kind: str) -> SimpleNamespace:
    return SimpleNamespace(
        summary_csv=out_dir / "neural_trace_summary.csv",
        trace_dir=out_dir / "traces_npz",
        feature_kind=feature_kind,
        output=out_dir / f"neural_trace_geometry_{feature_kind}.json",
        limit=None,
    )


def score_report(config_name: str, report: dict[str, Any]) -> dict[str, Any]:
    branch = report["branch_health"]
    nn = report["nearest_neighbor"]
    gd = report["group_distances"]
    tracked_axes = ["valence", "social_orientation", "action_tendency_class", "appraisal_family", "control_state"]
    lift_mean = sum(float(nn[axis]["lift"]) for axis in tracked_axes) / len(tracked_axes)
    sep_mean = sum(float(gd[axis]["separation"]) for axis in tracked_axes) / len(tracked_axes)
    return {
        "config": config_name,
        "feature_kind": report["feature_kind"],
        "n": report["n"],
        "mean_branch_len": branch["mean_dominant_branch_len"],
        "len1_ratio": branch["len1_ratio"],
        "mean_activation_density": branch["mean_activation_density"],
        "tracked_lift_mean": round(lift_mean, 6),
        "tracked_separation_mean": round(sep_mean, 6),
        "valence_lift": nn["valence"]["lift"],
        "social_orientation_lift": nn["social_orientation"]["lift"],
        "action_tendency_class_lift": nn["action_tendency_class"]["lift"],
        "appraisal_family_lift": nn["appraisal_family"]["lift"],
        "control_state_lift": nn["control_state"]["lift"],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    for config in SWEEP_CONFIGS:
        print(f"sweep: {config['name']}")
        out_dir = args.output_dir / config["name"]
        manifest = exporter.run(export_args(args, config, out_dir))
        manifests.append({"name": config["name"], "hypothesis": config["hypothesis"], "manifest": manifest})
        for feature_kind in args.feature_kinds:
            report = geometry.run(geometry_args(out_dir, feature_kind))
            row = score_report(config["name"], report)
            row["hypothesis"] = config["hypothesis"]
            row["params_json"] = json.dumps({**BASE_DYNAMICS, **config["params"]}, ensure_ascii=False, sort_keys=True)
            rows.append(row)

    fieldnames = [
        "config",
        "feature_kind",
        "n",
        "mean_branch_len",
        "len1_ratio",
        "mean_activation_density",
        "tracked_lift_mean",
        "tracked_separation_mean",
        "valence_lift",
        "social_orientation_lift",
        "action_tendency_class_lift",
        "appraisal_family_lift",
        "control_state_lift",
        "hypothesis",
        "params_json",
    ]
    summary_csv = args.output_dir / "dynamics_sweep_summary.csv"
    write_csv(summary_csv, rows, fieldnames)
    payload = {
        "output_dir": str(args.output_dir),
        "summary_csv": str(summary_csv),
        "limit": args.limit,
        "n_neurons": args.n_neurons,
        "feature_kinds": args.feature_kinds,
        "configs": manifests,
    }
    (args.output_dir / "dynamics_sweep_manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("outputs/targeted_records_trace_normalized.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/neural_trace_dynamics_sweep_v1"))
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--n-neurons", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stim-source", choices=["auto", "text", "proxy"], default="auto")
    parser.add_argument("--max-ticks", type=int, default=64)
    parser.add_argument("--min-ticks-before-converged", type=int, default=6)
    parser.add_argument("--convergence-patience", type=int, default=4)
    parser.add_argument("--feature-kinds", nargs="+", default=["branch_mean"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run(args)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

