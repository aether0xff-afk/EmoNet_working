from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from emonet.cli import (
    DEFAULT_Z_ENCODER_MODEL_PATH,
    build_model,
    load_training_json_as_dataframe,
    maybe_print_progress,
    resolve_num_workers,
    resolve_text_column,
)
from emonet.core import EmoNetConfig

import analyze_branch_dynamics as ANALYSIS
import optimize_branch_dynamics as OPT


DEFAULT_CALIBRATION_SPACE: dict[str, list[Any]] = {
    "k_threshold_base": [0.68, 0.70, 0.72, 0.74],
    "k_remem_base": [1.00, 1.05, 1.10],
    "k_decay": [0.91, 0.93, 0.95],
    "intrinsic_alignment_gain": [0.20, 0.24, 0.28],
    "fatigue_gain": [0.15, 0.20, 0.25],
    "inhibitory_suppression_gain": [0.12, 0.18, 0.24],
    "convergence_patience": [3, 4, 6],
    "activity_count_delta_eps": [1.0, 2.0, 3.0],
}


def coerce_scalar(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def load_input_dataframe(args: argparse.Namespace) -> pd.DataFrame:
    if bool(args.input_csv) == bool(args.input_json):
        raise ValueError("provide exactly one of --input-csv or --input-json")
    if args.input_json:
        return load_training_json_as_dataframe(Path(args.input_json))
    return pd.read_csv(Path(args.input_csv))


def sample_input_rows(df: pd.DataFrame, sample_size: int | None, sample_mode: str, seed: int) -> pd.DataFrame:
    if sample_size is None or sample_size <= 0 or len(df) <= sample_size:
        return df.reset_index(drop=True).copy()
    if sample_mode == "random":
        return df.sample(n=sample_size, random_state=seed).reset_index(drop=True).copy()
    return df.head(sample_size).reset_index(drop=True).copy()


def parse_calibration_space(args: argparse.Namespace, center: dict[str, Any]) -> dict[str, list[Any]]:
    search_space = {key: list(values) for key, values in DEFAULT_CALIBRATION_SPACE.items()}
    for raw in args.space:
        key, values = OPT.parse_space_assignment(raw)
        search_space[key] = values

    if args.calibrate_params:
        requested = [token.strip() for token in args.calibrate_params.split(",") if token.strip()]
        search_space = {key: search_space[key] for key in requested if key in search_space}
    if not search_space:
        raise ValueError("calibration search space is empty")

    normalized: dict[str, list[Any]] = {}
    for key, values in search_space.items():
        merged = [coerce_scalar(value) for value in values]
        center_value = coerce_scalar(center[key])
        if center_value not in merged:
            merged = [center_value] + merged
        normalized[key] = merged
    return normalized


def build_model_namespace(args: argparse.Namespace, overrides: dict[str, Any]) -> argparse.Namespace:
    payload = {
        "dataset_csv": args.dataset_csv,
        "benchmark_csv": args.benchmark_csv,
        "model_cache_path": args.model_cache_path,
        "max_samples": args.max_samples,
        "force_refit": args.force_refit,
        "seed": args.model_seed,
        "z_dim": args.z_dim,
        "z_encoder_mode": "stat",
        "z_encoder_path": str(DEFAULT_Z_ENCODER_MODEL_PATH),
    }
    for field_name in OPT.SWEEPABLE_FIELDS:
        payload[field_name] = None
    payload.update(overrides)
    return argparse.Namespace(**payload)


def evaluate_params(
    *,
    run_id: str,
    parameter_name: str,
    candidate_value: Any,
    is_center: bool,
    params: dict[str, Any],
    texts: list[str],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    model_args = build_model_namespace(args, params)
    worker_count = resolve_num_workers(args.num_workers)
    model = build_model(model_args) if worker_count <= 1 else None
    sample_df, tick_df, sample_summary = ANALYSIS.analyze_sample_runs(
        model=model,
        texts=texts,
        progress_every=args.progress_every,
        num_workers=args.num_workers,
        model_args=model_args,
    )
    ignition_df, ignition_summary = ANALYSIS.build_ignition_metrics(tick_df)
    sample_df = sample_df.copy()
    tick_df = tick_df.copy()
    ignition_df = ignition_df.copy()
    sample_df = sample_df.merge(ignition_df, on="sample_index", how="left")

    rows = int(sample_summary["rows"])
    no_activity_rows = int(ignition_summary["no_activity_rows"])
    no_activity_ratio = no_activity_rows / rows if rows else 0.0
    len1_ratio = float((sample_df["dominant_branch_len"] == 1).mean()) if rows else 0.0
    hit_max_ticks_ratio = float(sample_summary["hit_max_ticks_ratio"])
    mean_branch_len = float(sample_summary["mean_branch_len"])
    max_ticks = int(params.get("max_ticks", EmoNetConfig().max_ticks))
    mean_first_active_tick = float(ignition_summary["mean_first_active_tick"]) if ignition_summary["mean_first_active_tick"] is not None else float(max_ticks)
    late_ignition_ratio = float(ignition_summary["late_ignition_ratio_ge_15"])
    mean_active_window_ticks = float(ignition_summary["mean_active_window_ticks"])
    active_window_ratio = mean_active_window_ticks / max_ticks if max_ticks > 0 else 0.0
    branch_ratio = mean_branch_len / max_ticks if max_ticks > 0 else 0.0
    mean_active_nodes = float(tick_df["active_nodes"].mean()) if not tick_df.empty else 0.0
    mean_edges_fired = float(tick_df["edges_fired"].mean()) if not tick_df.empty else 0.0

    constraint_failures: list[str] = []
    constraint_penalty = 0.0

    def apply_upper(metric_name: str, metric_value: float, limit: float | None) -> None:
        nonlocal constraint_penalty
        if limit is None:
            return
        if metric_value > limit:
            scale = max(abs(limit), 1e-6)
            constraint_penalty += (metric_value - limit) / scale
            constraint_failures.append(f"{metric_name}>{limit}")

    def apply_lower(metric_name: str, metric_value: float, limit: float | None) -> None:
        nonlocal constraint_penalty
        if limit is None:
            return
        if metric_value < limit:
            scale = max(abs(limit), 1e-6)
            constraint_penalty += (limit - metric_value) / scale
            constraint_failures.append(f"{metric_name}<{limit}")

    apply_upper("no_activity_ratio", no_activity_ratio, args.max_no_activity_ratio)
    apply_upper("len1_ratio", len1_ratio, args.max_len1_ratio)
    apply_upper("hit_max_ticks_ratio", hit_max_ticks_ratio, args.max_hit_max_ticks_ratio)
    apply_upper("mean_first_active_tick", mean_first_active_tick, args.max_first_active_tick)
    apply_upper("late_ignition_ratio_ge_15", late_ignition_ratio, args.max_late_ignition_ratio)
    apply_lower("mean_branch_len", mean_branch_len, args.min_mean_branch_len)
    is_feasible = len(constraint_failures) == 0

    first_active_target_delta = abs(mean_first_active_tick - args.target_first_active_tick)
    branch_target_delta = abs(branch_ratio - args.target_branch_ratio)
    active_window_target_delta = abs(active_window_ratio - args.target_active_window_ratio)
    no_activity_component = (1.0 - no_activity_ratio) * 20.0
    len1_component = (1.0 - len1_ratio) * 20.0
    hitmax_component = (1.0 - hit_max_ticks_ratio) * 20.0
    late_component = (1.0 - late_ignition_ratio) * 10.0
    first_active_component = OPT.closeness_score(
        mean_first_active_tick,
        args.target_first_active_tick,
        args.target_first_active_tolerance,
    ) * 15.0
    branch_component = OPT.closeness_score(
        branch_ratio,
        args.target_branch_ratio,
        args.target_branch_tolerance,
    ) * 15.0
    evidence_score = round(
        no_activity_component
        + len1_component
        + hitmax_component
        + late_component
        + first_active_component
        + branch_component,
        4,
    )

    sample_df["run_id"] = run_id
    sample_df["parameter_name"] = parameter_name
    sample_df["candidate_value"] = candidate_value
    sample_df["is_center"] = is_center
    tick_df["run_id"] = run_id
    tick_df["parameter_name"] = parameter_name
    tick_df["candidate_value"] = candidate_value
    tick_df["is_center"] = is_center

    row = {
        "run_id": run_id,
        "parameter_name": parameter_name,
        "candidate_value": coerce_scalar(candidate_value),
        "is_center": bool(is_center),
        "rows": rows,
        "max_ticks": max_ticks,
        "mean_branch_len": mean_branch_len,
        "p95_branch_len": float(sample_summary["p95_branch_len"]),
        "mean_ticks_run": float(sample_summary["mean_ticks_run"]),
        "hit_max_ticks_ratio": hit_max_ticks_ratio,
        "len1_ratio": len1_ratio,
        "no_activity_ratio": no_activity_ratio,
        "no_activity_rows": no_activity_rows,
        "mean_first_active_tick": mean_first_active_tick,
        "late_ignition_ratio_ge_15": late_ignition_ratio,
        "mean_active_window_ticks": mean_active_window_ticks,
        "active_window_ratio": active_window_ratio,
        "branch_ratio": branch_ratio,
        "mean_active_nodes": mean_active_nodes,
        "mean_edges_fired": mean_edges_fired,
        "constraint_penalty": round(constraint_penalty, 6),
        "constraint_failures": ";".join(constraint_failures),
        "is_feasible": bool(is_feasible),
        "first_active_target_delta": first_active_target_delta,
        "branch_target_delta": branch_target_delta,
        "active_window_target_delta": active_window_target_delta,
        "evidence_score": evidence_score,
        "params_json": json.dumps(params, ensure_ascii=False, sort_keys=True),
    }
    return row, sample_df, tick_df


def load_resume_state(output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    evidence_csv = output_dir / "parameter_evidence.csv"
    sample_details_csv = output_dir / "sample_details.csv"
    tick_details_csv = output_dir / "tick_details.csv"
    evidence_df = pd.read_csv(evidence_csv) if evidence_csv.exists() else pd.DataFrame()
    sample_df = pd.read_csv(sample_details_csv) if sample_details_csv.exists() else pd.DataFrame()
    tick_df = pd.read_csv(tick_details_csv) if tick_details_csv.exists() else pd.DataFrame()
    return evidence_df, sample_df, tick_df


def rank_parameter_group(group: pd.DataFrame) -> pd.DataFrame:
    ranked = group.sort_values(
        by=[
            "is_feasible",
            "constraint_penalty",
            "no_activity_ratio",
            "len1_ratio",
            "hit_max_ticks_ratio",
            "first_active_target_delta",
            "branch_target_delta",
            "mean_branch_len",
        ],
        ascending=[False, True, True, True, True, True, True, False],
    ).reset_index(drop=True)
    ranked["parameter_rank"] = np.arange(1, len(ranked) + 1)
    ranked["is_recommended"] = False
    if not ranked.empty:
        ranked.loc[0, "is_recommended"] = True
    return ranked


def render_parameter_figure(group: pd.DataFrame, output_path: Path) -> None:
    chart_df = group.sort_values("candidate_value").copy()
    x = np.arange(len(chart_df))
    labels = [str(value) for value in chart_df["candidate_value"].tolist()]

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.2))
    plots = [
        ("no_activity_ratio", "No-Activity Ratio", "#dc2626"),
        ("len1_ratio", "L1 Ratio", "#ea580c"),
        ("hit_max_ticks_ratio", "Hit Max-Ticks Ratio", "#b45309"),
        ("mean_first_active_tick", "Mean First Active Tick", "#2563eb"),
        ("mean_branch_len", "Mean Branch Length", "#16a34a"),
        ("evidence_score", "Evidence Score", "#7c3aed"),
    ]
    for ax, (column, title, color) in zip(axes.flat, plots, strict=True):
        ax.plot(x, chart_df[column], marker="o", color=color, linewidth=1.8)
        recommended = chart_df[chart_df["is_recommended"]]
        if not recommended.empty:
            rec_x = chart_df.index[chart_df["run_id"] == recommended.iloc[0]["run_id"]].tolist()
            if rec_x:
                ax.scatter(rec_x, recommended[column], color="#111827", s=40, zorder=3)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30)
    fig.suptitle(f"Calibration Evidence: {chart_df.iloc[0]['parameter_name']}", fontsize=12)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="svg")
    plt.close(fig)


def write_artifacts(
    *,
    output_dir: Path,
    evidence_df: pd.DataFrame,
    sample_df: pd.DataFrame,
    tick_df: pd.DataFrame,
    center_config: dict[str, Any],
    calibrated_config: dict[str, Any] | None,
    combined_validation: dict[str, Any] | None,
    args: argparse.Namespace,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    if evidence_df.empty:
        ranked_df = evidence_df.copy()
        recommendation_df = evidence_df.copy()
    else:
        ranked_groups = [rank_parameter_group(group) for _, group in evidence_df.groupby("parameter_name", sort=True)]
        ranked_df = pd.concat(ranked_groups, ignore_index=True)
        ranked_df = ranked_df.sort_values(["parameter_name", "parameter_rank"]).reset_index(drop=True)
        recommendation_df = ranked_df[ranked_df["is_recommended"]].copy().reset_index(drop=True)

    evidence_csv = output_dir / "parameter_evidence.csv"
    recommendations_csv = output_dir / "parameter_recommendations.csv"
    sample_details_csv = output_dir / "sample_details.csv"
    tick_details_csv = output_dir / "tick_details.csv"
    ranked_df.to_csv(evidence_csv, index=False, encoding="utf-8-sig")
    recommendation_df.to_csv(recommendations_csv, index=False, encoding="utf-8-sig")
    sample_df.to_csv(sample_details_csv, index=False, encoding="utf-8-sig")
    tick_df.to_csv(tick_details_csv, index=False, encoding="utf-8-sig")

    (output_dir / "center_config.json").write_text(
        json.dumps(center_config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if calibrated_config is not None:
        (output_dir / "calibrated_reference_config.json").write_text(
            json.dumps(calibrated_config, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    if combined_validation is not None:
        (output_dir / "combined_validation.json").write_text(
            json.dumps(combined_validation, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    figure_paths: list[str] = []
    figures_dir = output_dir / "figures"
    if not ranked_df.empty:
        for parameter_name, group in ranked_df.groupby("parameter_name", sort=True):
            safe_name = str(parameter_name).replace("/", "_")
            figure_path = figures_dir / f"{safe_name}_calibration.svg"
            render_parameter_figure(group, figure_path)
            figure_paths.append(str(figure_path))

    progress_payload = {
        "completed_runs": int(len(ranked_df)),
        "recommended_rows": int(len(recommendation_df)),
        "parameter_evidence_csv": str(evidence_csv),
        "parameter_recommendations_csv": str(recommendations_csv),
        "sample_details_csv": str(sample_details_csv),
        "tick_details_csv": str(tick_details_csv),
        "figure_paths": figure_paths,
    }
    (output_dir / "progress.json").write_text(
        json.dumps(progress_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    write_report(
        output_dir / "CALIBRATION_REPORT.md",
        ranked_df=ranked_df,
        recommendation_df=recommendation_df,
        center_config=center_config,
        calibrated_config=calibrated_config,
        combined_validation=combined_validation,
        args=args,
        figure_paths=figure_paths,
    )


def write_report(
    output_path: Path,
    *,
    ranked_df: pd.DataFrame,
    recommendation_df: pd.DataFrame,
    center_config: dict[str, Any],
    calibrated_config: dict[str, Any] | None,
    combined_validation: dict[str, Any] | None,
    args: argparse.Namespace,
    figure_paths: list[str],
) -> None:
    lines = [
        "# Reference Config Calibration Report",
        "",
        "## Goal",
        "",
        "Calibrate a reference configuration with experimentally justified parameter values rather than arbitrary defaults.",
        "",
        "## Target Constraints",
        "",
        f"- max_no_activity_ratio: `{args.max_no_activity_ratio}`",
        f"- max_len1_ratio: `{args.max_len1_ratio}`",
        f"- max_hit_max_ticks_ratio: `{args.max_hit_max_ticks_ratio}`",
        f"- max_first_active_tick: `{args.max_first_active_tick}`",
        f"- max_late_ignition_ratio: `{args.max_late_ignition_ratio}`",
        f"- min_mean_branch_len: `{args.min_mean_branch_len}`",
        "",
        "## Target Operating Point",
        "",
        f"- target_first_active_tick: `{args.target_first_active_tick}`",
        f"- target_branch_ratio: `{args.target_branch_ratio}`",
        f"- target_active_window_ratio: `{args.target_active_window_ratio}`",
        "",
        "## Center Config",
        "",
        "```json",
        json.dumps(center_config, ensure_ascii=False, indent=2),
        "```",
        "",
    ]

    if not recommendation_df.empty:
        lines.extend(["## Parameter Recommendations", ""])
        for _, row in recommendation_df.iterrows():
            lines.extend(
                [
                    f"### {row['parameter_name']}",
                    "",
                    f"- recommended_value: `{row['candidate_value']}`",
                    f"- feasible: `{bool(row['is_feasible'])}`",
                    f"- constraint_penalty: `{row['constraint_penalty']}`",
                    f"- no_activity_ratio: `{row['no_activity_ratio']:.4f}`",
                    f"- len1_ratio: `{row['len1_ratio']:.4f}`",
                    f"- hit_max_ticks_ratio: `{row['hit_max_ticks_ratio']:.4f}`",
                    f"- mean_first_active_tick: `{row['mean_first_active_tick']:.4f}`",
                    f"- mean_branch_len: `{row['mean_branch_len']:.4f}`",
                    f"- evidence_score: `{row['evidence_score']:.4f}`",
                    "",
                ]
            )

    if calibrated_config is not None:
        lines.extend(
            [
                "## Calibrated Reference Config",
                "",
                "```json",
                json.dumps(calibrated_config, ensure_ascii=False, indent=2),
                "```",
                "",
            ]
        )

    if combined_validation is not None:
        lines.extend(
            [
                "## Combined Validation",
                "",
                "```json",
                json.dumps(combined_validation, ensure_ascii=False, indent=2),
                "```",
                "",
            ]
        )

    if not ranked_df.empty:
        lines.extend(["## Evidence Table Preview", ""])
        preview_cols = [
            "parameter_name",
            "candidate_value",
            "is_center",
            "is_recommended",
            "is_feasible",
            "no_activity_ratio",
            "len1_ratio",
            "hit_max_ticks_ratio",
            "mean_first_active_tick",
            "mean_branch_len",
            "evidence_score",
        ]
        preview = ranked_df[preview_cols].head(20)
        lines.extend(["```csv", preview.to_csv(index=False).strip(), "```", ""])

    if figure_paths:
        lines.extend(["## Figures", ""])
        for figure_path in figure_paths:
            lines.append(f"- `{Path(figure_path).name}`")
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def build_calibrated_config(center: dict[str, Any], recommendation_df: pd.DataFrame) -> dict[str, Any]:
    calibrated = dict(center)
    for _, row in recommendation_df.iterrows():
        calibrated[str(row["parameter_name"])] = coerce_scalar(row["candidate_value"])
    return calibrated


def build_combined_validation_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "config_name": row["run_id"],
        "is_feasible": bool(row["is_feasible"]),
        "constraint_penalty": float(row["constraint_penalty"]),
        "constraint_failures": row["constraint_failures"],
        "no_activity_ratio": float(row["no_activity_ratio"]),
        "len1_ratio": float(row["len1_ratio"]),
        "hit_max_ticks_ratio": float(row["hit_max_ticks_ratio"]),
        "mean_first_active_tick": float(row["mean_first_active_tick"]),
        "late_ignition_ratio_ge_15": float(row["late_ignition_ratio_ge_15"]),
        "mean_branch_len": float(row["mean_branch_len"]),
        "mean_active_window_ticks": float(row["mean_active_window_ticks"]),
        "evidence_score": float(row["evidence_score"]),
        "params_json": row["params_json"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate a branch-dynamics reference config from experimental evidence.")
    parser.add_argument("--input-csv", default=None)
    parser.add_argument("--input-json", default=None)
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--sample-size", type=int, default=60)
    parser.add_argument("--sample-mode", choices=["head", "random"], default="random")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-seed", type=int, default=42)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--dataset-csv", default=None)
    parser.add_argument("--benchmark-csv", default=None)
    parser.add_argument("--model-cache-path", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--force-refit", action="store_true")
    parser.add_argument("--z-dim", type=int, default=64)
    parser.add_argument("--fixed", action="append", default=[], help="fixed key=value override")
    parser.add_argument("--space", action="append", default=[], help="parameter space override key=v1,v2,...")
    parser.add_argument("--calibrate-params", default="", help="comma-separated parameter subset")
    parser.add_argument("--target-first-active-tick", type=float, default=4.0)
    parser.add_argument("--target-first-active-tolerance", type=float, default=8.0)
    parser.add_argument("--target-branch-ratio", type=float, default=0.60)
    parser.add_argument("--target-branch-tolerance", type=float, default=0.30)
    parser.add_argument("--target-active-window-ratio", type=float, default=0.60)
    parser.add_argument("--max-no-activity-ratio", type=float, default=0.10)
    parser.add_argument("--max-len1-ratio", type=float, default=0.15)
    parser.add_argument("--max-hit-max-ticks-ratio", type=float, default=0.35)
    parser.add_argument("--max-first-active-tick", type=float, default=10.0)
    parser.add_argument("--max-late-ignition-ratio", type=float, default=0.10)
    parser.add_argument("--min-mean-branch-len", type=float, default=40.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-dir", default=str(Path("outputs") / "branch_calibration" / "latest"))
    args = parser.parse_args()

    defaults = OPT.get_default_param_values()
    fixed = OPT.parse_assignment_map(args.fixed)
    center = dict(defaults)
    center.update(fixed)
    calibration_space = parse_calibration_space(args, center)

    input_df = load_input_dataframe(args)
    text_column = resolve_text_column(input_df, args.text_column)
    sampled_df = sample_input_rows(input_df, args.sample_size, args.sample_mode, args.seed)
    texts = sampled_df[text_column].fillna("").astype(str).tolist()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sampled_df.to_csv(output_dir / "sampled_inputs.csv", index=False, encoding="utf-8-sig")
    (output_dir / "calibration_space.json").write_text(json.dumps(calibration_space, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.resume:
        evidence_df, sample_df_all, tick_df_all = load_resume_state(output_dir)
    else:
        evidence_df, sample_df_all, tick_df_all = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    completed_runs = set(evidence_df["run_id"].astype(str).tolist()) if not evidence_df.empty else set()
    total_runs = 1
    for _, values in calibration_space.items():
        total_runs += len(values) - 1
    overall_start = time.perf_counter()

    center_run_id = "center"
    if center_run_id not in completed_runs:
        center_row, center_sample_df, center_tick_df = evaluate_params(
            run_id=center_run_id,
            parameter_name="center",
            candidate_value="center",
            is_center=True,
            params=center,
            texts=texts,
            args=args,
        )
        evidence_df = pd.concat([evidence_df, pd.DataFrame([center_row])], ignore_index=True)
        sample_df_all = pd.concat([sample_df_all, center_sample_df], ignore_index=True)
        tick_df_all = pd.concat([tick_df_all, center_tick_df], ignore_index=True)
        completed_runs.add(center_run_id)
        maybe_print_progress("config-calibration", 1, total_runs, overall_start, every=1, unit="runs", extra=center_run_id)
    else:
        maybe_print_progress("config-calibration", 1, total_runs, overall_start, every=1, unit="runs", extra=f"resume-skip {center_run_id}")

    run_index = 1
    center_row_dict = (
        evidence_df.loc[evidence_df["run_id"] == center_run_id].iloc[0].to_dict()
        if not evidence_df.empty and (evidence_df["run_id"] == center_run_id).any()
        else None
    )

    for parameter_name, values in calibration_space.items():
        for value in values:
            if coerce_scalar(value) == coerce_scalar(center[parameter_name]):
                continue
            run_index += 1
            run_id = f"{parameter_name}={value}"
            if run_id in completed_runs:
                maybe_print_progress("config-calibration", run_index, total_runs, overall_start, every=1, unit="runs", extra=f"resume-skip {run_id}")
                continue

            params = dict(center)
            params[parameter_name] = value
            row, sample_df, tick_df = evaluate_params(
                run_id=run_id,
                parameter_name=parameter_name,
                candidate_value=value,
                is_center=False,
                params=params,
                texts=texts,
                args=args,
            )
            evidence_df = pd.concat([evidence_df, pd.DataFrame([row])], ignore_index=True)
            sample_df_all = pd.concat([sample_df_all, sample_df], ignore_index=True)
            tick_df_all = pd.concat([tick_df_all, tick_df], ignore_index=True)
            completed_runs.add(run_id)
            maybe_print_progress("config-calibration", run_index, total_runs, overall_start, every=1, unit="runs", extra=run_id)

            provisional_ranked = pd.concat(
                [rank_parameter_group(group) for _, group in evidence_df[evidence_df["parameter_name"] != "center"].groupby("parameter_name", sort=True)],
                ignore_index=True,
            ) if not evidence_df[evidence_df["parameter_name"] != "center"].empty else pd.DataFrame()
            provisional_reco = provisional_ranked[provisional_ranked["is_recommended"]].copy().reset_index(drop=True) if not provisional_ranked.empty else pd.DataFrame()
            provisional_calibrated = build_calibrated_config(center, provisional_reco) if not provisional_reco.empty else dict(center)
            write_artifacts(
                output_dir=output_dir,
                evidence_df=evidence_df[evidence_df["parameter_name"] != "center"].copy(),
                sample_df=sample_df_all,
                tick_df=tick_df_all,
                center_config=center,
                calibrated_config=provisional_calibrated,
                combined_validation=None,
                args=args,
            )

    parameter_evidence = evidence_df[evidence_df["parameter_name"] != "center"].copy()
    if parameter_evidence.empty:
        recommendation_df = pd.DataFrame()
    else:
        ranked_groups = [rank_parameter_group(group) for _, group in parameter_evidence.groupby("parameter_name", sort=True)]
        ranked_df = pd.concat(ranked_groups, ignore_index=True)
        recommendation_df = ranked_df[ranked_df["is_recommended"]].copy().reset_index(drop=True)

    calibrated_config = build_calibrated_config(center, recommendation_df) if not recommendation_df.empty else dict(center)
    combined_validation = None
    if calibrated_config == center and center_row_dict is not None:
        combined_validation = build_combined_validation_payload(center_row_dict)
        combined_validation["run_mode"] = "center_reused"
    else:
        combined_row, combined_sample_df, combined_tick_df = evaluate_params(
            run_id="combined_validation",
            parameter_name="combined_validation",
            candidate_value="combined_validation",
            is_center=False,
            params=calibrated_config,
            texts=texts,
            args=args,
        )
        combined_validation = build_combined_validation_payload(combined_row)
        sample_df_all = pd.concat([sample_df_all, combined_sample_df], ignore_index=True)
        tick_df_all = pd.concat([tick_df_all, combined_tick_df], ignore_index=True)

    write_artifacts(
        output_dir=output_dir,
        evidence_df=parameter_evidence,
        sample_df=sample_df_all,
        tick_df=tick_df_all,
        center_config=center,
        calibrated_config=calibrated_config,
        combined_validation=combined_validation,
        args=args,
    )

    payload = {
        "input_rows": int(len(input_df)),
        "sample_rows": int(len(sampled_df)),
        "parameter_count": int(len(calibration_space)),
        "evidence_rows": int(len(parameter_evidence)),
        "recommended_rows": int(len(recommendation_df)),
        "center_config": center,
        "calibrated_reference_config_path": str(output_dir / "calibrated_reference_config.json"),
        "combined_validation_path": str(output_dir / "combined_validation.json"),
        "parameter_evidence_csv": str(output_dir / "parameter_evidence.csv"),
        "parameter_recommendations_csv": str(output_dir / "parameter_recommendations.csv"),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
