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

from emonet.cli import maybe_print_progress


REQUIRED_SUMMARY_COLUMNS = {
    "config_name",
    "balanced_score",
    "constraint_penalty",
    "len1_ratio",
    "hit_max_ticks_ratio",
    "mean_first_active_tick",
    "late_ignition_ratio_ge_15",
    "mean_branch_len",
}
REQUIRED_DETAILS_COLUMNS = {
    "sample_index",
    "text",
    "dominant_branch_len",
    "first_active_tick",
    "last_active_tick",
    "active_window_ticks",
    "mean_active_nodes",
    "mean_edges_fired",
    "max_active_nodes",
    "config_name",
}
REQUIRED_TICK_COLUMNS = {"sample_index", "tick", "active_nodes", "edges_fired", "has_activity", "config_name"}


def validate_columns(df: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {', '.join(missing)}")


def choose_configs(
    summary_df: pd.DataFrame,
    baseline_name: str,
    explicit_configs: list[str],
    top_k_candidates: int,
) -> list[str]:
    available = set(summary_df["config_name"].astype(str).tolist())
    selected: list[str] = []
    if baseline_name in available:
        selected.append(baseline_name)

    if explicit_configs:
        for config_name in explicit_configs:
            if config_name in available and config_name not in selected:
                selected.append(config_name)
        return selected

    nonbaseline = summary_df.loc[summary_df["config_name"] != baseline_name].copy()
    if nonbaseline.empty:
        return selected

    candidate_frames = [
        nonbaseline.sort_values(by=["constraint_penalty", "balanced_score"], ascending=[True, False]),
        nonbaseline.sort_values(by=["balanced_score"], ascending=[False]),
        nonbaseline.sort_values(by=["hit_max_ticks_ratio", "constraint_penalty"], ascending=[True, True]),
        nonbaseline.sort_values(by=["mean_first_active_tick", "constraint_penalty"], ascending=[True, True]),
        nonbaseline.sort_values(by=["late_ignition_ratio_ge_15", "constraint_penalty"], ascending=[True, True]),
    ]
    for frame in candidate_frames:
        for config_name in frame["config_name"].astype(str).tolist():
            if config_name not in selected:
                selected.append(config_name)
            if len(selected) >= 1 + top_k_candidates:
                return selected
    return selected


def compute_config_comparison(summary_df: pd.DataFrame, details_df: pd.DataFrame, tick_df: pd.DataFrame, config_names: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for config_name in config_names:
        summary_row = summary_df.loc[summary_df["config_name"] == config_name].iloc[0]
        detail_slice = details_df.loc[details_df["config_name"] == config_name]
        tick_slice = tick_df.loc[tick_df["config_name"] == config_name]
        tick_group = tick_slice.groupby("tick", as_index=False).agg(
            mean_active_nodes=("active_nodes", "mean"),
            mean_edges_fired=("edges_fired", "mean"),
        )

        total_activity = float(tick_group["mean_active_nodes"].sum()) if not tick_group.empty else 0.0
        cumulative = tick_group["mean_active_nodes"].cumsum() / total_activity if total_activity > 1e-8 else pd.Series(np.zeros(len(tick_group)))

        def reach_tick(threshold: float) -> float | None:
            if tick_group.empty or total_activity <= 1e-8:
                return None
            reached = tick_group.loc[cumulative >= threshold, "tick"]
            return float(reached.iloc[0]) if not reached.empty else None

        rows.append(
            {
                "config_name": config_name,
                "balanced_score": float(summary_row["balanced_score"]),
                "constraint_penalty": float(summary_row.get("constraint_penalty", 0.0)),
                "constraint_failures": str(summary_row.get("constraint_failures", "")),
                "len1_ratio": float(summary_row["len1_ratio"]),
                "hit_max_ticks_ratio": float(summary_row["hit_max_ticks_ratio"]),
                "mean_first_active_tick": float(summary_row["mean_first_active_tick"]),
                "late_ignition_ratio_ge_15": float(summary_row["late_ignition_ratio_ge_15"]),
                "mean_branch_len": float(summary_row["mean_branch_len"]),
                "mean_active_nodes": float(detail_slice["mean_active_nodes"].mean()),
                "mean_peak_active_nodes": float(detail_slice["max_active_nodes"].mean()),
                "mean_edges_fired": float(detail_slice["mean_edges_fired"].mean()),
                "mean_peak_edges_fired": float(detail_slice["max_edges_fired"].mean()) if "max_edges_fired" in detail_slice.columns else float("nan"),
                "mean_active_window_ticks": float(detail_slice["active_window_ticks"].mean()),
                "p10_activity_tick": reach_tick(0.10),
                "p50_activity_tick": reach_tick(0.50),
                "p90_activity_tick": reach_tick(0.90),
            }
        )
    return pd.DataFrame(rows)


def choose_representative_samples(details_df: pd.DataFrame, baseline_name: str, top_k_samples: int) -> pd.DataFrame:
    baseline = details_df.loc[details_df["config_name"] == baseline_name].copy()
    if baseline.empty:
        return pd.DataFrame(columns=["sample_index", "text", "category", "rank"])

    picks: list[dict[str, Any]] = []
    seen: set[int] = set()
    category_specs = [
        ("delayed", baseline.sort_values(by=["first_active_tick", "dominant_branch_len"], ascending=[False, False])),
        ("saturated", baseline.sort_values(by=["mean_active_nodes", "max_active_nodes"], ascending=[False, False])),
        ("short_branch", baseline.sort_values(by=["dominant_branch_len", "first_active_tick"], ascending=[True, False])),
    ]
    per_category = max(1, int(np.ceil(top_k_samples / len(category_specs))))
    for category_name, frame in category_specs:
        rank = 0
        for _, row in frame.iterrows():
            sample_index = int(row["sample_index"])
            if sample_index in seen:
                continue
            rank += 1
            picks.append(
                {
                    "sample_index": sample_index,
                    "text": str(row["text"]),
                    "category": category_name,
                    "rank": rank,
                    "first_active_tick": float(row["first_active_tick"]),
                    "dominant_branch_len": float(row["dominant_branch_len"]),
                    "mean_active_nodes": float(row["mean_active_nodes"]),
                }
            )
            seen.add(sample_index)
            if rank >= per_category or len(picks) >= top_k_samples:
                break
        if len(picks) >= top_k_samples:
            break
    return pd.DataFrame(picks)


def sanitize_filename(text: str) -> str:
    clean = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text)
    return clean[:120]


def plot_mean_profiles(tick_df: pd.DataFrame, config_names: list[str], output_dir: Path) -> list[Path]:
    figure_paths: list[Path] = []
    profile_specs = [
        ("mean_active_nodes", "active_nodes", "Mean Active Nodes by Tick", "active nodes", "config_mean_active_nodes.svg"),
        ("mean_edges_fired", "edges_fired", "Mean Edges Fired by Tick", "edges fired", "config_mean_edges_fired.svg"),
        ("activity_ratio", "has_activity", "Activity Ratio by Tick", "activity ratio", "config_activity_ratio.svg"),
    ]
    palette = ["#2563eb", "#dc2626", "#16a34a", "#ea580c", "#7c3aed", "#0891b2"]

    for _, source_column, title, ylabel, filename in profile_specs:
        fig, ax = plt.subplots(figsize=(8.0, 5.0))
        for color_idx, config_name in enumerate(config_names):
            config_slice = tick_df.loc[tick_df["config_name"] == config_name]
            grouped = config_slice.groupby("tick", as_index=False)[source_column].mean()
            ax.plot(grouped["tick"], grouped[source_column], label=config_name, color=palette[color_idx % len(palette)], linewidth=2.0)
        ax.set_title(title)
        ax.set_xlabel("tick")
        ax.set_ylabel(ylabel)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        output_path = output_dir / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, format="svg")
        plt.close(fig)
        figure_paths.append(output_path)
    return figure_paths


def plot_sample_traces(
    tick_df: pd.DataFrame,
    representative_df: pd.DataFrame,
    config_names: list[str],
    output_dir: Path,
    progress_every: int,
) -> list[Path]:
    figure_paths: list[Path] = []
    palette = ["#2563eb", "#dc2626", "#16a34a", "#ea580c", "#7c3aed", "#0891b2"]
    start_time = time.perf_counter()
    for idx, row in enumerate(representative_df.to_dict(orient="records"), start=1):
        sample_index = int(row["sample_index"])
        category = str(row["category"])
        filename_prefix = sanitize_filename(f"sample{sample_index}_{category}")
        sample_tick = tick_df.loc[tick_df["sample_index"] == sample_index]

        for metric_name, column_name, ylabel in (
            ("active_nodes", "active_nodes", "active nodes"),
            ("edges_fired", "edges_fired", "edges fired"),
        ):
            fig, ax = plt.subplots(figsize=(8.0, 4.8))
            for color_idx, config_name in enumerate(config_names):
                config_slice = sample_tick.loc[sample_tick["config_name"] == config_name]
                ax.plot(
                    config_slice["tick"],
                    config_slice[column_name],
                    label=config_name,
                    color=palette[color_idx % len(palette)],
                    linewidth=2.0,
                )
            ax.set_title(f"Sample {sample_index} | {category} | {metric_name}")
            ax.set_xlabel("tick")
            ax.set_ylabel(ylabel)
            ax.legend(loc="best", fontsize=8)
            fig.tight_layout()
            output_path = output_dir / f"{filename_prefix}_{metric_name}.svg"
            fig.savefig(output_path, format="svg")
            plt.close(fig)
            figure_paths.append(output_path)
        maybe_print_progress("trace-plots", idx, len(representative_df), start_time, every=max(1, progress_every), unit="samples")
    return figure_paths


def build_pairwise_delta_table(comparison_df: pd.DataFrame, baseline_name: str) -> pd.DataFrame:
    baseline_rows = comparison_df.loc[comparison_df["config_name"] == baseline_name]
    if baseline_rows.empty:
        return pd.DataFrame()
    baseline = baseline_rows.iloc[0]
    rows: list[dict[str, Any]] = []
    for _, row in comparison_df.iterrows():
        if row["config_name"] == baseline_name:
            continue
        rows.append(
            {
                "config_name": row["config_name"],
                "delta_mean_branch_len": float(row["mean_branch_len"] - baseline["mean_branch_len"]),
                "delta_len1_ratio": float(row["len1_ratio"] - baseline["len1_ratio"]),
                "delta_hit_max_ticks_ratio": float(row["hit_max_ticks_ratio"] - baseline["hit_max_ticks_ratio"]),
                "delta_mean_first_active_tick": float(row["mean_first_active_tick"] - baseline["mean_first_active_tick"]),
                "delta_late_ignition_ratio_ge_15": float(row["late_ignition_ratio_ge_15"] - baseline["late_ignition_ratio_ge_15"]),
                "delta_mean_active_nodes": float(row["mean_active_nodes"] - baseline["mean_active_nodes"]),
                "delta_mean_edges_fired": float(row["mean_edges_fired"] - baseline["mean_edges_fired"]),
            }
        )
    return pd.DataFrame(rows)


def write_report(
    output_path: Path,
    *,
    baseline_name: str,
    config_names: list[str],
    comparison_df: pd.DataFrame,
    representative_df: pd.DataFrame,
    figure_paths: list[Path],
) -> None:
    lines = [
        "# Branch Trace Analysis Report",
        "",
        f"- baseline: `{baseline_name}`",
        f"- compared_configs: `{config_names}`",
        "",
        "## Config Comparison",
        "",
    ]
    for _, row in comparison_df.iterrows():
        lines.extend(
            [
                f"### {row['config_name']}",
                "",
                f"- balanced_score: `{row['balanced_score']:.4f}`",
                f"- constraint_penalty: `{row['constraint_penalty']:.6f}`",
                f"- constraint_failures: `{row['constraint_failures']}`",
                f"- mean_branch_len: `{row['mean_branch_len']:.4f}`",
                f"- len1_ratio: `{row['len1_ratio']:.4f}`",
                f"- hit_max_ticks_ratio: `{row['hit_max_ticks_ratio']:.4f}`",
                f"- mean_first_active_tick: `{row['mean_first_active_tick']:.4f}`",
                f"- late_ignition_ratio_ge_15: `{row['late_ignition_ratio_ge_15']:.4f}`",
                f"- mean_active_nodes: `{row['mean_active_nodes']:.4f}`",
                f"- mean_edges_fired: `{row['mean_edges_fired']:.4f}`",
                f"- p10/p50/p90 activity ticks: `{row['p10_activity_tick']}` / `{row['p50_activity_tick']}` / `{row['p90_activity_tick']}`",
                "",
            ]
        )

    lines.extend(["## Representative Samples", ""])
    for _, row in representative_df.iterrows():
        lines.extend(
            [
                f"- sample_index={int(row['sample_index'])} | category={row['category']} | first_active_tick={row['first_active_tick']} | dominant_branch_len={row['dominant_branch_len']}",
                f"  - text: {row['text']}",
            ]
        )
    lines.extend(["", "## Figures", ""])
    for path in figure_paths:
        lines.append(f"- `{path.name}`")
    lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-csv", required=True)
    parser.add_argument("--details-csv", required=True)
    parser.add_argument("--tick-csv", required=True)
    parser.add_argument("--baseline-name", default="baseline")
    parser.add_argument("--config-name", action="append", default=[])
    parser.add_argument("--top-k-candidates", type=int, default=3)
    parser.add_argument("--top-k-samples", type=int, default=6)
    parser.add_argument("--progress-every", type=int, default=1)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_df = pd.read_csv(Path(args.summary_csv))
    details_df = pd.read_csv(Path(args.details_csv))
    tick_df = pd.read_csv(Path(args.tick_csv))
    validate_columns(summary_df, REQUIRED_SUMMARY_COLUMNS, "summary_csv")
    validate_columns(details_df, REQUIRED_DETAILS_COLUMNS, "details_csv")
    validate_columns(tick_df, REQUIRED_TICK_COLUMNS, "tick_csv")

    config_names = choose_configs(summary_df, args.baseline_name, args.config_name, args.top_k_candidates)
    summary_slice = summary_df.loc[summary_df["config_name"].isin(config_names)].copy()
    details_slice = details_df.loc[details_df["config_name"].isin(config_names)].copy()
    tick_slice = tick_df.loc[tick_df["config_name"].isin(config_names)].copy()

    comparison_df = compute_config_comparison(summary_slice, details_slice, tick_slice, config_names)
    representative_df = choose_representative_samples(details_slice, args.baseline_name, args.top_k_samples)
    pairwise_df = build_pairwise_delta_table(comparison_df, args.baseline_name)

    selected_configs_json = output_dir / "selected_configs.json"
    selected_configs_json.write_text(json.dumps(config_names, ensure_ascii=False, indent=2), encoding="utf-8")
    comparison_csv = output_dir / "config_comparison.csv"
    representative_csv = output_dir / "representative_samples.csv"
    pairwise_csv = output_dir / "pairwise_deltas.csv"
    comparison_df.to_csv(comparison_csv, index=False, encoding="utf-8-sig")
    representative_df.to_csv(representative_csv, index=False, encoding="utf-8-sig")
    pairwise_df.to_csv(pairwise_csv, index=False, encoding="utf-8-sig")

    figure_paths = []
    figure_paths.extend(plot_mean_profiles(tick_slice, config_names, output_dir))
    figure_paths.extend(
        plot_sample_traces(
            tick_df=tick_slice,
            representative_df=representative_df,
            config_names=config_names,
            output_dir=output_dir,
            progress_every=args.progress_every,
        )
    )

    write_report(
        output_dir / "TRACE_ANALYSIS_REPORT.md",
        baseline_name=args.baseline_name,
        config_names=config_names,
        comparison_df=comparison_df,
        representative_df=representative_df,
        figure_paths=figure_paths,
    )

    payload = {
        "selected_configs": config_names,
        "comparison_csv": str(comparison_csv),
        "representative_csv": str(representative_csv),
        "pairwise_csv": str(pairwise_csv),
        "figure_paths": [str(path) for path in figure_paths],
        "report_path": str(output_dir / "TRACE_ANALYSIS_REPORT.md"),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
