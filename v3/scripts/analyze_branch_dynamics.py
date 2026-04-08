from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from emonet.cli import build_model as build_emonet_model, maybe_print_progress, resolve_text_column
from emonet.cli import (
    _init_parallel_model,
    _require_parallel_model,
    estimate_executor_chunksize,
    prepare_parallel_model_payload,
    resolve_num_workers,
)


DEFAULT_COMPARE_PATHS = [
    PROJECT_ROOT / "outputs" / "z" / "out_z_training_extended40.csv",
    PROJECT_ROOT / "outputs" / "z" / "out_z_training_extended40_branchfix.csv",
    PROJECT_ROOT / "outputs" / "z" / "out_z_training_extended40_branchfix_v2.csv",
    PROJECT_ROOT / "outputs" / "z" / "out_z_training_extended40_structfix.csv",
]


def summarize_branch_lengths(lengths: pd.Series) -> dict[str, float]:
    values = lengths.astype(int)
    return {
        "rows": int(len(values)),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "len1_count": int((values == 1).sum()),
        "len1_ratio": float((values == 1).mean()),
        "p90": float(values.quantile(0.90)),
        "p95": float(values.quantile(0.95)),
        "max": float(values.max()),
    }


def load_version_summary(paths: list[Path]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in paths:
        if not path.exists():
            continue
        df = pd.read_csv(path, usecols=["dominant_branch_len"])
        summary = summarize_branch_lengths(df["dominant_branch_len"])
        summary["dataset"] = path.stem
        summary["source_csv"] = str(path)
        rows.append(summary)
    if not rows:
        raise FileNotFoundError("no comparison CSVs with dominant_branch_len were found")
    return pd.DataFrame(rows)


def sample_input_rows(df: pd.DataFrame, sample_size: int | None, sample_mode: str, seed: int) -> pd.DataFrame:
    if sample_size is None or sample_size <= 0 or len(df) <= sample_size:
        return df.reset_index(drop=True).copy()
    if sample_mode == "random":
        return df.sample(n=sample_size, random_state=seed).reset_index(drop=True)
    return df.head(sample_size).copy()


def _analyze_single_text(model, sample_index: int, text: str) -> tuple[dict[str, object], list[dict[str, object]]]:
    outputs = model.forward(text)
    branch_log = list(model.state.branch_log)
    topk_paths = list(model.topk_branches)
    ticks_run = int(model.state.tick)
    max_ticks = int(model.config.max_ticks)
    active_counts = [len(record.active_nodes) for record in branch_log]
    edge_counts = [len(record.edges_fired) for record in branch_log]
    nonempty_ticks = [record.tick for record in branch_log if record.active_nodes]
    last_nonempty_tick = int(max(nonempty_ticks)) if nonempty_ticks else -1
    active_tick_count = int(sum(count > 0 for count in active_counts))
    dominant_branch_len = int(len(outputs["dominant_branch"]))
    termination_reason = "max_ticks" if ticks_run >= max_ticks else "delta_k"
    silent_tail = max(0, ticks_run - 1 - last_nonempty_tick) if ticks_run > 0 else 0
    path_coverage = float(dominant_branch_len / active_tick_count) if active_tick_count > 0 else 0.0
    sample_row = {
        "sample_index": sample_index,
        "text": text,
        "ticks_run": ticks_run,
        "max_ticks": max_ticks,
        "termination_reason": termination_reason,
        "final_delta_k": float(getattr(model, "_last_delta_k", np.nan)),
        "dominant_branch_len": dominant_branch_len,
        "active_tick_count": active_tick_count,
        "path_coverage": path_coverage,
        "silent_tail_ticks": int(silent_tail),
        "max_active_nodes": int(max(active_counts)) if active_counts else 0,
        "mean_active_nodes": float(np.mean(active_counts)) if active_counts else 0.0,
        "final_active_nodes": int(active_counts[-1]) if active_counts else 0,
        "max_edges_fired": int(max(edge_counts)) if edge_counts else 0,
        "mean_edges_fired": float(np.mean(edge_counts)) if edge_counts else 0.0,
        "final_edges_fired": int(edge_counts[-1]) if edge_counts else 0,
        "top1_path_len": int(len(topk_paths[0].steps)) if topk_paths else 0,
        "topk_path_count": int(len(topk_paths)),
    }
    tick_rows = [
        {
            "sample_index": sample_index,
            "tick": int(record.tick),
            "active_nodes": int(active_count),
            "edges_fired": int(edge_count),
            "has_activity": int(active_count > 0),
        }
        for record, active_count, edge_count in zip(branch_log, active_counts, edge_counts, strict=True)
    ]
    return sample_row, tick_rows


def _parallel_analyze_record(task: tuple[int, str]) -> tuple[dict[str, object], list[dict[str, object]]]:
    sample_index, text = task
    model = _require_parallel_model()
    return _analyze_single_text(model, sample_index, text)


def analyze_sample_runs(
    model,
    texts: list[str],
    *,
    progress_every: int,
    num_workers: int = 1,
    model_args: argparse.Namespace | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    sample_rows: list[dict[str, object]] = []
    tick_rows: list[dict[str, object]] = []
    start_time = time.perf_counter()
    worker_count = resolve_num_workers(num_workers)

    if worker_count <= 1:
        if model is None:
            raise ValueError("model is required when num_workers <= 1")
        for idx, text in enumerate(texts, start=1):
            sample_row, sample_tick_rows = _analyze_single_text(model, idx, text)
            sample_rows.append(sample_row)
            tick_rows.extend(sample_tick_rows)
            maybe_print_progress("branch-analysis", idx, len(texts), start_time, every=progress_every, unit="samples")
    else:
        if model_args is None:
            raise ValueError("model_args is required when num_workers > 1")
        payload = prepare_parallel_model_payload(model_args)
        chunksize = estimate_executor_chunksize(len(texts), worker_count, preferred=32)
        task_iter = ((idx, text) for idx, text in enumerate(texts, start=1))
        with ProcessPoolExecutor(
            max_workers=worker_count,
            initializer=_init_parallel_model,
            initargs=(payload,),
        ) as executor:
            for idx, (sample_row, sample_tick_rows) in enumerate(
                executor.map(_parallel_analyze_record, task_iter, chunksize=chunksize),
                start=1,
            ):
                sample_rows.append(sample_row)
                tick_rows.extend(sample_tick_rows)
                maybe_print_progress("branch-analysis", idx, len(texts), start_time, every=progress_every, unit="samples")

    sample_df = pd.DataFrame(sample_rows)
    tick_df = pd.DataFrame(tick_rows)
    termination_counts = (
        sample_df["termination_reason"].value_counts(dropna=False).rename_axis("termination_reason").reset_index(name="count")
    )
    summary = {
        "rows": int(len(sample_df)),
        "mean_branch_len": float(sample_df["dominant_branch_len"].mean()),
        "p95_branch_len": float(sample_df["dominant_branch_len"].quantile(0.95)),
        "mean_ticks_run": float(sample_df["ticks_run"].mean()),
        "p95_ticks_run": float(sample_df["ticks_run"].quantile(0.95)),
        "hit_max_ticks_ratio": float((sample_df["termination_reason"] == "max_ticks").mean()),
        "mean_path_coverage": float(sample_df["path_coverage"].mean()),
        "mean_silent_tail_ticks": float(sample_df["silent_tail_ticks"].mean()),
        "termination_counts": termination_counts.to_dict(orient="records"),
    }
    return sample_df, tick_df, summary


def plot_version_summary(summary_df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    labels = summary_df["dataset"].tolist()

    axes[0].bar(labels, summary_df["mean"], color="#3a6ea5")
    axes[0].set_title("Mean Branch Length")
    axes[0].tick_params(axis="x", rotation=25)

    axes[1].bar(labels, summary_df["len1_ratio"], color="#c8553d")
    axes[1].set_title("L1 Ratio")
    axes[1].tick_params(axis="x", rotation=25)

    axes[2].bar(labels, summary_df["p95"], color="#4c9f70")
    axes[2].set_title("P95 Branch Length")
    axes[2].tick_params(axis="x", rotation=25)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="svg")
    plt.close(fig)


def plot_ticks_vs_branch(sample_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    colors = sample_df["termination_reason"].map({"max_ticks": "#c8553d", "delta_k": "#3a6ea5"}).fillna("#777777")
    ax.scatter(sample_df["ticks_run"], sample_df["dominant_branch_len"], c=colors, alpha=0.75, s=26)
    max_ticks = int(sample_df["max_ticks"].max()) if not sample_df.empty else 0
    if max_ticks > 0:
        ax.axvline(max_ticks, color="#999999", linestyle="--", linewidth=1.0, label=f"max_ticks={max_ticks}")
    ax.set_xlabel("Ticks Run")
    ax.set_ylabel("Dominant Branch Length")
    ax.set_title("Ticks Run vs Dominant Branch Length")
    ax.legend(loc="lower right")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="svg")
    plt.close(fig)


def plot_tick_activity(tick_df: pd.DataFrame, output_path: Path) -> None:
    grouped = tick_df.groupby("tick", as_index=False).agg(
        mean_active_nodes=("active_nodes", "mean"),
        p90_active_nodes=("active_nodes", lambda values: float(np.quantile(values, 0.90))),
        mean_edges_fired=("edges_fired", "mean"),
        active_ratio=("has_activity", "mean"),
    )
    fig, ax1 = plt.subplots(figsize=(7.0, 5.0))
    ax1.plot(grouped["tick"], grouped["mean_active_nodes"], color="#3a6ea5", label="mean active nodes")
    ax1.plot(grouped["tick"], grouped["p90_active_nodes"], color="#7aa6d1", linestyle="--", label="p90 active nodes")
    ax1.set_xlabel("Tick")
    ax1.set_ylabel("Active Nodes")

    ax2 = ax1.twinx()
    ax2.plot(grouped["tick"], grouped["mean_edges_fired"], color="#c8553d", label="mean edges fired")
    ax2.plot(grouped["tick"], grouped["active_ratio"], color="#4c9f70", linestyle=":", label="activity ratio")
    ax2.set_ylabel("Edges / Activity Ratio")

    lines = ax1.get_lines() + ax2.get_lines()
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="upper right")
    ax1.set_title("Tick-wise Activity Profile")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="svg")
    plt.close(fig)


def plot_termination_counts(sample_df: pd.DataFrame, output_path: Path) -> None:
    counts = sample_df["termination_reason"].value_counts(dropna=False)
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    ax.bar(counts.index.tolist(), counts.values.tolist(), color=["#c8553d", "#3a6ea5"])
    ax.set_title("Termination Reason Counts")
    ax.set_ylabel("Samples")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="svg")
    plt.close(fig)


def plot_path_coverage(sample_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    ax.hist(sample_df["path_coverage"], bins=20, color="#6c5b7b", edgecolor="white")
    ax.set_title("Dominant Path Coverage")
    ax.set_xlabel("dominant_branch_len / active_tick_count")
    ax.set_ylabel("Samples")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="svg")
    plt.close(fig)


def build_ignition_metrics(tick_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    rows: list[dict[str, object]] = []
    for sample_index, group in tick_df.groupby("sample_index"):
        active = group[group["active_nodes"] > 0]
        if active.empty:
            first_active_tick = -1
            last_active_tick = -1
            active_window_ticks = 0
        else:
            first_active_tick = int(active["tick"].min())
            last_active_tick = int(active["tick"].max())
            active_window_ticks = int(last_active_tick - first_active_tick + 1)
        rows.append(
            {
                "sample_index": int(sample_index),
                "first_active_tick": first_active_tick,
                "last_active_tick": last_active_tick,
                "active_window_ticks": active_window_ticks,
            }
        )

    ignition_df = pd.DataFrame(rows)
    valid_first = ignition_df.loc[ignition_df["first_active_tick"] >= 0, "first_active_tick"]
    summary = {
        "rows": int(len(ignition_df)),
        "no_activity_rows": int((ignition_df["first_active_tick"] < 0).sum()),
        "mean_first_active_tick": float(valid_first.mean()) if len(valid_first) else None,
        "median_first_active_tick": float(valid_first.median()) if len(valid_first) else None,
        "p90_first_active_tick": float(valid_first.quantile(0.90)) if len(valid_first) else None,
        "late_ignition_ratio_ge_15": float((ignition_df["first_active_tick"] >= 15).mean()),
        "mean_active_window_ticks": float(ignition_df["active_window_ticks"].mean()),
        "median_active_window_ticks": float(ignition_df["active_window_ticks"].median()),
    }
    return ignition_df, summary


def plot_first_active_tick(ignition_df: pd.DataFrame, output_path: Path) -> None:
    valid_first = ignition_df.loc[ignition_df["first_active_tick"] >= 0, "first_active_tick"]
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    ax.hist(valid_first, bins=15, color="#355c7d", edgecolor="white")
    ax.set_title("First Active Tick Distribution")
    ax.set_xlabel("First tick with any active node")
    ax.set_ylabel("Samples")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="svg")
    plt.close(fig)


def plot_active_window(ignition_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    ax.hist(ignition_df["active_window_ticks"], bins=15, color="#4c9f70", edgecolor="white")
    ax.set_title("Active Window Length Distribution")
    ax.set_xlabel("Ticks between first and last active tick")
    ax.set_ylabel("Samples")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="svg")
    plt.close(fig)


def write_report(
    output_path: Path,
    *,
    version_df: pd.DataFrame,
    sample_summary: dict[str, object],
    sample_df: pd.DataFrame,
    config: dict[str, object],
) -> None:
    best_current = version_df.iloc[-1]
    max_tick_ratio = float(sample_summary["hit_max_ticks_ratio"])
    coverage = float(sample_summary["mean_path_coverage"])
    mean_ticks = float(sample_summary["mean_ticks_run"])
    max_ticks = int(config["max_ticks"])

    findings: list[str] = []
    findings.append(
        f"- Current hard cap is `max_ticks={max_ticks}` with `min_ticks_before_converged={config['min_ticks_before_converged']}`."
    )
    findings.append(
        f"- Full export improved mean dominant branch length to `{best_current['mean']:.2f}` and reduced `L1` ratio to `{best_current['len1_ratio']:.4f}`, but the upper tail is still bounded (`p95={best_current['p95']:.1f}`, `max={best_current['max']:.0f}`)."
    )
    if max_tick_ratio >= 0.25:
        findings.append(
            f"- `{max_tick_ratio:.1%}` of sampled runs terminated by hitting `max_ticks`, so the hard cap is already a material bottleneck."
        )
    else:
        findings.append(
            f"- Only `{max_tick_ratio:.1%}` of sampled runs hit `max_ticks`; most runs stopped by the convergence rule before exhausting depth."
        )
    if mean_ticks < 0.8 * max_ticks:
        findings.append(
            f"- Mean ticks run is `{mean_ticks:.2f}`, well below the cap, which indicates the current `delta_k` convergence test is still aggressive."
        )
    if coverage < 0.8:
        findings.append(
            f"- Mean dominant-path coverage is only `{coverage:.3f}`, so many active ticks do not survive into the selected branch. This suggests a second bottleneck in path selection or connectivity continuity, not only raw depth."
        )
    if float(sample_df["silent_tail_ticks"].mean()) > 1.0:
        findings.append(
            f"- Silent tail after the last active tick averages `{float(sample_df['silent_tail_ticks'].mean()):.2f}` ticks, which means the model often keeps stepping after meaningful branch activity has already decayed."
        )

    lines = [
        "# Branch Dynamics Research",
        "",
        "## Configuration",
        "",
        f"- max_ticks: {config['max_ticks']}",
        f"- min_ticks_before_converged: {config['min_ticks_before_converged']}",
        f"- k_threshold_base: {config['k_threshold_base']}",
        f"- k_decay: {config['k_decay']}",
        f"- input_topk: {config['input_topk']}",
        f"- input_signal_clip: {config['input_signal_clip']}",
        "",
        "## Key Findings",
        "",
        *findings,
        "",
        "## Version Summary",
        "",
        version_df.to_string(index=False),
        "",
        "## Sample Probe Summary",
        "",
        json.dumps(sample_summary, ensure_ascii=False, indent=2),
        "",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze branch-depth bottlenecks with graphs and tables.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--sample-mode", choices=["head", "random"], default="random")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-csv", dest="dataset_csv", type=str, default=None)
    parser.add_argument("--benchmark-csv", dest="benchmark_csv", type=str, default=None)
    parser.add_argument("--model-cache-path", dest="model_cache_path", type=str, default=None)
    parser.add_argument("--max-samples", dest="max_samples", type=int, default=None)
    parser.add_argument("--force-refit", action="store_true")
    parser.add_argument("--z-dim", dest="z_dim", type=int, default=64)
    parser.add_argument("--z-encoder-mode", choices=["auto", "stat", "transformer"], default="stat")
    parser.add_argument("--z-encoder-path", default=None)
    parser.add_argument("--max-ticks", dest="max_ticks", type=int, default=None)
    parser.add_argument("--min-ticks-before-converged", dest="min_ticks_before_converged", type=int, default=None)
    parser.add_argument("--k-threshold-base", dest="k_threshold_base", type=float, default=None)
    parser.add_argument("--k-remem-base", dest="k_remem_base", type=float, default=None)
    parser.add_argument("--k-decay", dest="k_decay", type=float, default=None)
    parser.add_argument("--refractory-ticks", dest="refractory_ticks", type=int, default=None)
    parser.add_argument("--input-topk", dest="input_topk", type=int, default=None)
    parser.add_argument("--input-signal-clip", dest="input_signal_clip", type=float, default=None)
    parser.add_argument("--memory-decay", dest="memory_decay", type=float, default=None)
    parser.add_argument("--memory-stim-mix", dest="memory_stim_mix", type=float, default=None)
    parser.add_argument("--memory-k-mix", dest="memory_k_mix", type=float, default=None)
    parser.add_argument("--recent-activity-decay", dest="recent_activity_decay", type=float, default=None)
    parser.add_argument("--global-recovery-rate", dest="global_recovery_rate", type=float, default=None)
    parser.add_argument("--ignition-topk", dest="ignition_topk", type=int, default=None)
    parser.add_argument("--ignition-strength-scale", dest="ignition_strength_scale", type=float, default=None)
    parser.add_argument("--branch-end-window", dest="branch_end_window", type=int, default=None)
    parser.add_argument("--branch-length-bonus", dest="branch_length_bonus", type=float, default=None)
    parser.add_argument("--compare-z-csv", dest="compare_z_csvs", action="append", default=None)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    compare_paths = [Path(path) for path in args.compare_z_csvs] if args.compare_z_csvs else list(DEFAULT_COMPARE_PATHS)
    version_df = load_version_summary(compare_paths)
    version_df.to_csv(tables_dir / "branch_version_summary.csv", index=False, encoding="utf-8-sig")
    plot_version_summary(version_df, figures_dir / "branch_version_summary.svg")

    input_df = pd.read_csv(Path(args.input_csv))
    text_column = resolve_text_column(input_df, args.text_column)
    sampled_df = sample_input_rows(input_df, args.sample_size, args.sample_mode, args.seed)
    sampled_df.to_csv(tables_dir / "sampled_inputs.csv", index=False, encoding="utf-8-sig")

    worker_count = resolve_num_workers(getattr(args, "num_workers", 1))
    model = build_emonet_model(args) if worker_count <= 1 else None
    texts = sampled_df[text_column].astype(str).tolist()
    sample_df, tick_df, sample_summary = analyze_sample_runs(
        model,
        texts,
        progress_every=args.progress_every,
        num_workers=worker_count,
        model_args=args,
    )
    sample_df.to_csv(tables_dir / "sample_metrics.csv", index=False, encoding="utf-8-sig")
    tick_df.to_csv(tables_dir / "tick_metrics.csv", index=False, encoding="utf-8-sig")
    (tables_dir / "sample_probe_summary.json").write_text(
        json.dumps(sample_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    ignition_df, ignition_summary = build_ignition_metrics(tick_df)
    ignition_df.to_csv(tables_dir / "ignition_metrics.csv", index=False, encoding="utf-8-sig")
    (tables_dir / "ignition_summary.json").write_text(
        json.dumps(ignition_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    plot_ticks_vs_branch(sample_df, figures_dir / "ticks_vs_branch_length.svg")
    plot_tick_activity(tick_df, figures_dir / "tick_activity_profile.svg")
    plot_termination_counts(sample_df, figures_dir / "termination_reasons.svg")
    plot_path_coverage(sample_df, figures_dir / "path_coverage_histogram.svg")
    plot_first_active_tick(ignition_df, figures_dir / "first_active_tick_histogram.svg")
    plot_active_window(ignition_df, figures_dir / "active_window_histogram.svg")

    config_model = model if model is not None else build_emonet_model(args)
    config = {
        "max_ticks": int(config_model.config.max_ticks),
        "min_ticks_before_converged": int(config_model.config.min_ticks_before_converged),
        "k_threshold_base": float(config_model.config.k_threshold_base),
        "k_decay": float(config_model.config.k_decay),
        "input_topk": int(config_model.config.input_topk),
        "input_signal_clip": float(config_model.config.input_signal_clip),
        "ignition_topk": int(config_model.config.ignition_topk),
        "ignition_strength_scale": float(config_model.config.ignition_strength_scale),
    }
    write_report(
        output_dir / "BRANCH_DYNAMICS_RESEARCH.md",
        version_df=version_df,
        sample_summary=sample_summary,
        sample_df=sample_df,
        config=config,
    )

    final_summary = {
        "output_dir": str(output_dir),
        "sample_rows": int(len(sample_df)),
        "version_rows": int(len(version_df)),
        "num_workers": int(worker_count),
        "config": config,
        "sample_probe_summary_path": str(tables_dir / "sample_probe_summary.json"),
        "report_path": str(output_dir / "BRANCH_DYNAMICS_RESEARCH.md"),
    }
    print(json.dumps(final_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
