#!/usr/bin/env python3
"""Fine-tune neural trace dynamics around the first stable configuration."""

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
    "k_threshold_base": 0.58,
    "k_remem_base": 0.80,
    "k_decay": 0.99,
    "refractory_ticks": 1,
    "input_topk": 4,
    "input_signal_clip": 2.40,
    "intrinsic_alignment_gain": 0.24,
    "fatigue_gain": 0.10,
    "fatigue_threshold_gain": 0.05,
    "fatigue_k_leak": 0.02,
    "inhibitory_suppression_gain": 0.06,
    "ne_thresh_reduce_gain": 0.25,
    "ne_remem_reduce_gain": 0.25,
    "activity_churn_eps": 0.02,
}


def fine_grid_configs() -> list[dict[str, Any]]:
    thresholds = [0.60, 0.63, 0.66]
    input_clips = [1.60, 1.90, 2.20]
    inhibitions = [0.08, 0.12]
    fatigue_profiles = [
        {"name": "low_fatigue", "fatigue_gain": 0.12, "fatigue_threshold_gain": 0.06, "fatigue_k_leak": 0.025},
        {"name": "mid_fatigue", "fatigue_gain": 0.16, "fatigue_threshold_gain": 0.08, "fatigue_k_leak": 0.035},
    ]
    configs: list[dict[str, Any]] = []
    for threshold, clip, inhibition, fatigue in itertools.product(thresholds, input_clips, inhibitions, fatigue_profiles):
        params = dict(BASE_PARAMS)
        params.update(
            {
                "k_threshold_base": threshold,
                "k_remem_base": max(0.72, threshold + 0.18),
                "input_signal_clip": clip,
                "inhibitory_suppression_gain": inhibition,
                "fatigue_gain": fatigue["fatigue_gain"],
                "fatigue_threshold_gain": fatigue["fatigue_threshold_gain"],
                "fatigue_k_leak": fatigue["fatigue_k_leak"],
            }
        )
        configs.append(
            {
                "name": f"thr{threshold:.2f}_clip{clip:.1f}_inh{inhibition:.2f}_{fatigue['name']}",
                "params": params,
            }
        )
    return configs


def conservative_grid_configs() -> list[dict[str, Any]]:
    grid = [
        {
            "name": "thr0.70_clip1.2_inh0.16_high_fatigue",
            "k_threshold_base": 0.70,
            "input_signal_clip": 1.20,
            "inhibitory_suppression_gain": 0.16,
            "fatigue_gain": 0.22,
            "fatigue_threshold_gain": 0.12,
            "fatigue_k_leak": 0.05,
        },
        {
            "name": "thr0.70_clip1.4_inh0.16_high_fatigue",
            "k_threshold_base": 0.70,
            "input_signal_clip": 1.40,
            "inhibitory_suppression_gain": 0.16,
            "fatigue_gain": 0.22,
            "fatigue_threshold_gain": 0.12,
            "fatigue_k_leak": 0.05,
        },
        {
            "name": "thr0.74_clip1.2_inh0.20_high_fatigue",
            "k_threshold_base": 0.74,
            "input_signal_clip": 1.20,
            "inhibitory_suppression_gain": 0.20,
            "fatigue_gain": 0.24,
            "fatigue_threshold_gain": 0.14,
            "fatigue_k_leak": 0.06,
        },
        {
            "name": "thr0.74_clip1.4_inh0.20_high_fatigue",
            "k_threshold_base": 0.74,
            "input_signal_clip": 1.40,
            "inhibitory_suppression_gain": 0.20,
            "fatigue_gain": 0.24,
            "fatigue_threshold_gain": 0.14,
            "fatigue_k_leak": 0.06,
        },
    ]
    configs: list[dict[str, Any]] = []
    for item in grid:
        params = dict(BASE_PARAMS)
        threshold = float(item["k_threshold_base"])
        params.update(
            {
                "k_threshold_base": threshold,
                "k_remem_base": threshold + 0.18,
                "input_signal_clip": item["input_signal_clip"],
                "inhibitory_suppression_gain": item["inhibitory_suppression_gain"],
                "fatigue_gain": item["fatigue_gain"],
                "fatigue_threshold_gain": item["fatigue_threshold_gain"],
                "fatigue_k_leak": item["fatigue_k_leak"],
            }
        )
        configs.append({"name": str(item["name"]), "params": params})
    return configs


def adaptive_grid_configs() -> list[dict[str, Any]]:
    base_candidates = [
        {
            "name": "adaptive_thr0.60_clip1.6_inh0.10_start8_cap0.78",
            "k_threshold_base": 0.60,
            "input_signal_clip": 1.60,
            "inhibitory_suppression_gain": 0.10,
            "fatigue_gain": 0.14,
            "fatigue_threshold_gain": 0.07,
            "fatigue_k_leak": 0.030,
            "density_control_start_tick": 8,
            "density_target_high": 0.72,
            "density_soft_k_leak_gain": 1.25,
            "density_hard_cap": 0.78,
            "density_pruned_fatigue_gain": 0.06,
        },
        {
            "name": "adaptive_thr0.63_clip1.6_inh0.10_start8_cap0.76",
            "k_threshold_base": 0.63,
            "input_signal_clip": 1.60,
            "inhibitory_suppression_gain": 0.10,
            "fatigue_gain": 0.15,
            "fatigue_threshold_gain": 0.08,
            "fatigue_k_leak": 0.035,
            "density_control_start_tick": 8,
            "density_target_high": 0.70,
            "density_soft_k_leak_gain": 1.50,
            "density_hard_cap": 0.76,
            "density_pruned_fatigue_gain": 0.08,
        },
        {
            "name": "adaptive_thr0.63_clip1.8_inh0.12_start10_cap0.80",
            "k_threshold_base": 0.63,
            "input_signal_clip": 1.80,
            "inhibitory_suppression_gain": 0.12,
            "fatigue_gain": 0.14,
            "fatigue_threshold_gain": 0.08,
            "fatigue_k_leak": 0.035,
            "density_control_start_tick": 10,
            "density_target_high": 0.74,
            "density_soft_k_leak_gain": 1.25,
            "density_hard_cap": 0.80,
            "density_pruned_fatigue_gain": 0.06,
        },
        {
            "name": "adaptive_thr0.66_clip1.6_inh0.12_start8_cap0.76",
            "k_threshold_base": 0.66,
            "input_signal_clip": 1.60,
            "inhibitory_suppression_gain": 0.12,
            "fatigue_gain": 0.16,
            "fatigue_threshold_gain": 0.09,
            "fatigue_k_leak": 0.040,
            "density_control_start_tick": 8,
            "density_target_high": 0.70,
            "density_soft_k_leak_gain": 1.50,
            "density_hard_cap": 0.76,
            "density_pruned_fatigue_gain": 0.08,
        },
    ]
    configs: list[dict[str, Any]] = []
    for item in base_candidates:
        params = dict(BASE_PARAMS)
        threshold = float(item["k_threshold_base"])
        params.update(
            {
                "k_threshold_base": threshold,
                "k_remem_base": max(0.72, threshold + 0.18),
                **{key: value for key, value in item.items() if key != "name"},
            }
        )
        configs.append({"name": str(item["name"]), "params": params})
    return configs


def grid_configs(mode: str = "fine") -> list[dict[str, Any]]:
    if mode == "fine":
        return fine_grid_configs()
    if mode == "conservative":
        return conservative_grid_configs()
    if mode == "adaptive":
        return adaptive_grid_configs()
    raise ValueError(f"unknown grid mode: {mode}")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_existing_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def upsert_row(rows: list[dict[str, Any]], row: dict[str, Any]) -> list[dict[str, Any]]:
    keyed = {str(existing.get("config", "")): existing for existing in rows}
    keyed[str(row.get("config", ""))] = row
    merged = list(keyed.values())
    merged.sort(key=lambda item: float(item.get("objective", 0.0)), reverse=True)
    return merged


def export_args(args: argparse.Namespace, config: dict[str, Any], out_dir: Path) -> SimpleNamespace:
    params = {
        "density_control_start_tick": 0,
        "density_target_high": 1.0,
        "density_soft_k_leak_gain": 0.0,
        "density_hard_cap": 1.0,
        "density_pruned_fatigue_gain": 0.0,
        **config["params"],
    }
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


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def density_penalty(density: float, low: float = 0.55, high: float = 0.80) -> float:
    if low <= density <= high:
        return 0.0
    if density < low:
        return low - density
    return density - high


def score_feature_report(report: dict[str, Any]) -> dict[str, float]:
    branch = report["branch_health"]
    nn = report["nearest_neighbor"]
    balanced_nn = report.get("balanced_nearest_neighbor", {})
    gd = report["group_distances"]
    lifts = [float(nn[axis]["lift"]) for axis in TRACKED_AXES]
    balanced_lifts = [
        float(balanced_nn.get(axis, {}).get("balanced_lift", 0.0))
        for axis in TRACKED_AXES
    ]
    separations = [float(gd[axis]["separation"]) for axis in TRACKED_AXES]
    len1 = float(branch["len1_ratio"])
    density = float(branch["mean_activation_density"])
    branch_len = float(branch["mean_dominant_branch_len"])
    sep = mean(separations)
    lift = mean(lifts)

    return {
        "mean_branch_len": branch_len,
        "len1_ratio": len1,
        "mean_activation_density": density,
        "tracked_lift_mean": lift,
        "tracked_balanced_lift_mean": mean(balanced_lifts),
        "tracked_separation_mean": sep,
        **{f"{axis}_lift": float(nn[axis]["lift"]) for axis in TRACKED_AXES},
        **{
            f"{axis}_balanced_lift": float(balanced_nn.get(axis, {}).get("balanced_lift", 0.0))
            for axis in TRACKED_AXES
        },
        **{f"{axis}_separation": float(gd[axis]["separation"]) for axis in TRACKED_AXES},
    }


def score_candidate(branch_mean_report: dict[str, Any], branch_temporal_report: dict[str, Any]) -> dict[str, Any]:
    mean_score = score_feature_report(branch_mean_report)
    temporal_score = score_feature_report(branch_temporal_report)
    len1 = mean_score["len1_ratio"]
    density = mean_score["mean_activation_density"]
    branch_len = mean_score["mean_branch_len"]
    combined_lift = 0.7 * mean_score["tracked_lift_mean"] + 0.3 * temporal_score["tracked_lift_mean"]
    combined_balanced_lift = (
        0.7 * mean_score["tracked_balanced_lift_mean"]
        + 0.3 * temporal_score["tracked_balanced_lift_mean"]
    )
    combined_sep = 0.7 * mean_score["tracked_separation_mean"] + 0.3 * temporal_score["tracked_separation_mean"]

    # Prefer low collapse, controlled density, and stable branch geometry.
    # Over-activation is now penalized strongly because the first no-collapse
    # candidate reached density ~0.95 and weakened several emotion axes.
    objective = (
        3.0 * combined_sep
        + 0.5 * combined_lift
        + 0.5 * combined_balanced_lift
        + 0.01 * min(branch_len, 40.0)
        - 4.0 * len1
        - 4.0 * density_penalty(density)
    )
    return {
        "mean_branch_len": round(branch_len, 6),
        "len1_ratio": round(len1, 6),
        "mean_activation_density": round(density, 6),
        "tracked_lift_mean": round(mean_score["tracked_lift_mean"], 6),
        "tracked_balanced_lift_mean": round(mean_score["tracked_balanced_lift_mean"], 6),
        "tracked_separation_mean": round(mean_score["tracked_separation_mean"], 6),
        "branch_temporal_lift_mean": round(temporal_score["tracked_lift_mean"], 6),
        "branch_temporal_balanced_lift_mean": round(temporal_score["tracked_balanced_lift_mean"], 6),
        "branch_temporal_separation_mean": round(temporal_score["tracked_separation_mean"], 6),
        "combined_lift_mean": round(combined_lift, 6),
        "combined_balanced_lift_mean": round(combined_balanced_lift, 6),
        "combined_separation_mean": round(combined_sep, 6),
        "objective": round(objective, 6),
        **{f"{axis}_lift": round(mean_score[f"{axis}_lift"], 6) for axis in TRACKED_AXES},
        **{f"{axis}_balanced_lift": round(mean_score[f"{axis}_balanced_lift"], 6) for axis in TRACKED_AXES},
        **{f"{axis}_separation": round(mean_score[f"{axis}_separation"], 6) for axis in TRACKED_AXES},
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    configs = grid_configs(args.grid_mode)
    if args.start_index and args.start_index > 1:
        configs = configs[args.start_index - 1 :]
    if args.max_configs and args.max_configs > 0:
        configs = configs[: args.max_configs]
    fieldnames = [
        "config",
        "ok_rows",
        "error_rows",
        "objective",
        "mean_branch_len",
        "len1_ratio",
        "mean_activation_density",
        "tracked_lift_mean",
        "tracked_balanced_lift_mean",
        "tracked_separation_mean",
        "branch_temporal_lift_mean",
        "branch_temporal_balanced_lift_mean",
        "branch_temporal_separation_mean",
        "combined_lift_mean",
        "combined_balanced_lift_mean",
        "combined_separation_mean",
        *[f"{axis}_lift" for axis in TRACKED_AXES],
        *[f"{axis}_balanced_lift" for axis in TRACKED_AXES],
        *[f"{axis}_separation" for axis in TRACKED_AXES],
        "params_json",
    ]
    summary_csv = args.output_dir / "fine_dynamics_sweep_summary.csv"
    if args.resume:
        rows = read_existing_csv(summary_csv)
    for idx, config in enumerate(configs, start=1):
        print(f"fine-sweep: {idx}/{len(configs)} {config['name']}")
        out_dir = args.output_dir / config["name"]
        manifest_path = out_dir / "neural_trace_manifest.json"
        branch_mean_path = out_dir / "neural_trace_geometry_branch_mean.json"
        branch_temporal_path = out_dir / "neural_trace_geometry_branch_temporal.json"
        if args.resume and manifest_path.exists() and branch_mean_path.exists() and branch_temporal_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            branch_mean_report = json.loads(branch_mean_path.read_text(encoding="utf-8"))
            branch_temporal_report = json.loads(branch_temporal_path.read_text(encoding="utf-8"))
            print(f"fine-sweep: reusing {config['name']}")
        else:
            manifest = exporter.run(export_args(args, config, out_dir))
            branch_mean_report = geometry.run(geometry_args(out_dir, "branch_mean"))
            branch_temporal_report = geometry.run(geometry_args(out_dir, "branch_temporal"))
        row = {
            "config": config["name"],
            "ok_rows": manifest["ok_rows"],
            "error_rows": manifest["error_rows"],
            **score_candidate(branch_mean_report, branch_temporal_report),
            "params_json": json.dumps(config["params"], ensure_ascii=False, sort_keys=True),
        }
        rows = upsert_row(rows, row)
        write_csv(summary_csv, rows, fieldnames)

    rows.sort(key=lambda row: float(row["objective"]), reverse=True)
    write_csv(summary_csv, rows, fieldnames)
    payload = {
        "output_dir": str(args.output_dir),
        "summary_csv": str(summary_csv),
        "limit": args.limit,
        "n_neurons": args.n_neurons,
        "tracked_axes": TRACKED_AXES,
        "density_target": [0.55, 0.80],
        "configs": len(configs),
        "top_configs": rows[: min(8, len(rows))],
    }
    (args.output_dir / "fine_dynamics_sweep_manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("outputs/targeted_records_trace_normalized.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/neural_trace_dynamics_fine_sweep_v1"))
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--n-neurons", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stim-source", choices=["auto", "text", "proxy"], default="auto")
    parser.add_argument("--max-ticks", type=int, default=64)
    parser.add_argument("--min-ticks-before-converged", type=int, default=6)
    parser.add_argument("--convergence-patience", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--grid-mode", choices=["fine", "conservative", "adaptive"], default="fine")
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--max-configs", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run(args)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

