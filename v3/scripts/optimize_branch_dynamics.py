from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import math
from pathlib import Path
import random
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

from emonet.cli import (
    DEFAULT_Z_ENCODER_MODEL_PATH,
    build_model,
    load_training_json_as_dataframe,
    maybe_print_progress,
    resolve_num_workers,
    resolve_text_column,
)
from emonet.core import EmoNetConfig


SWEEPABLE_FIELDS: dict[str, type] = {
    "max_ticks": int,
    "min_ticks_before_converged": int,
    "k_threshold_base": float,
    "k_remem_base": float,
    "k_decay": float,
    "refractory_ticks": int,
    "input_topk": int,
    "input_signal_clip": float,
    "memory_decay": float,
    "memory_stim_mix": float,
    "memory_k_mix": float,
    "state_self_stim_mix": float,
    "state_parent_stim_mix": float,
    "state_base_stim_mix": float,
    "state_bias_stim_mix": float,
    "recent_activity_decay": float,
    "hysteresis_threshold_gain": float,
    "hysteresis_remem_gain": float,
    "hysteresis_k_bonus": float,
    "max_out_degree": int,
    "min_out_degree": int,
    "dopa_rewire_gain": float,
    "sero_prune_gain": float,
    "mela_dropout_gain": float,
    "ne_thresh_reduce_gain": float,
    "ne_remem_reduce_gain": float,
    "global_recovery_rate": float,
    "topk_branches": int,
    "branch_end_window": int,
    "branch_length_bonus": float,
}

PRESET_SEARCH_SPACES: dict[str, dict[str, list[Any]]] = {
    "sticky_reduction": {
        "k_threshold_base": [0.72, 0.80, 0.90, 1.00],
        "k_remem_base": [0.95, 1.05, 1.15],
        "k_decay": [0.93, 0.95, 0.97, 0.99],
        "refractory_ticks": [1, 2, 3],
        "input_signal_clip": [0.80, 1.00, 1.20, 1.50],
        "recent_activity_decay": [0.30, 0.50, 0.70, 0.80],
        "hysteresis_threshold_gain": [0.00, 0.03, 0.06, 0.12],
        "hysteresis_remem_gain": [0.00, 0.02, 0.04, 0.08],
        "hysteresis_k_bonus": [0.00, 0.02, 0.04, 0.08],
        "memory_decay": [0.97, 0.98, 0.985],
        "memory_k_mix": [0.00, 0.10, 0.20, 0.35],
        "state_base_stim_mix": [0.05, 0.10, 0.15],
    }
}


def load_analysis_module():
    helper_path = PROJECT_ROOT / "scripts" / "analyze_branch_dynamics.py"
    spec = importlib.util.spec_from_file_location("analyze_branch_dynamics_module", helper_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


ANALYSIS = load_analysis_module()


def parse_assignment(raw: str) -> tuple[str, Any]:
    if "=" not in raw:
        raise ValueError(f"expected key=value assignment, got: {raw}")
    key, value = raw.split("=", 1)
    key = key.strip()
    value = value.strip()
    if key not in SWEEPABLE_FIELDS:
        valid = ", ".join(sorted(SWEEPABLE_FIELDS))
        raise ValueError(f"unknown parameter '{key}'. valid fields: {valid}")
    caster = SWEEPABLE_FIELDS[key]
    return key, caster(value)


def parse_space_assignment(raw: str) -> tuple[str, list[Any]]:
    if "=" not in raw:
        raise ValueError(f"expected key=v1,v2,... assignment, got: {raw}")
    key, values_raw = raw.split("=", 1)
    key = key.strip()
    if key not in SWEEPABLE_FIELDS:
        valid = ", ".join(sorted(SWEEPABLE_FIELDS))
        raise ValueError(f"unknown parameter '{key}'. valid fields: {valid}")
    caster = SWEEPABLE_FIELDS[key]
    values = [token.strip() for token in values_raw.split(",") if token.strip()]
    if not values:
        raise ValueError(f"search space for '{key}' must contain at least one value")
    return key, [caster(value) for value in values]


def parse_assignment_map(raw_values: list[str]) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for raw in raw_values:
        key, value = parse_assignment(raw)
        parsed[key] = value
    return parsed


def get_default_param_values() -> dict[str, Any]:
    config = EmoNetConfig()
    return {field_name: getattr(config, field_name) for field_name in SWEEPABLE_FIELDS}


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
    for field_name in SWEEPABLE_FIELDS:
        payload[field_name] = None
    payload.update(overrides)
    return argparse.Namespace(**payload)


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
        return df.sample(n=sample_size, random_state=seed).reset_index(drop=True)
    return df.head(sample_size).reset_index(drop=True).copy()


def merge_search_space(args: argparse.Namespace, fixed: dict[str, Any]) -> dict[str, list[Any]]:
    search_space = {key: list(values) for key, values in PRESET_SEARCH_SPACES[args.preset].items()}
    for raw in args.space:
        key, values = parse_space_assignment(raw)
        search_space[key] = values
    for key in fixed:
        search_space.pop(key, None)
    if not search_space:
        raise ValueError("search space is empty after applying --fixed overrides")
    return search_space


def count_total_combinations(search_space: dict[str, list[Any]]) -> int:
    total = 1
    for values in search_space.values():
        total *= max(1, len(values))
    return total


def stable_params_key(params: dict[str, Any]) -> str:
    return json.dumps(params, ensure_ascii=False, sort_keys=True)


def build_grid_specs(
    center: dict[str, Any],
    search_space: dict[str, list[Any]],
    include_baseline: bool,
) -> list[dict[str, Any]]:
    keys = list(search_space.keys())
    seen: set[str] = set()
    specs: list[dict[str, Any]] = []
    if include_baseline:
        baseline_key = stable_params_key(center)
        seen.add(baseline_key)
        specs.append({"name": "baseline", "params": dict(center)})

    for combo in itertools.product(*(search_space[key] for key in keys)):
        params = dict(center)
        name_tokens: list[str] = []
        for key, value in zip(keys, combo, strict=True):
            params[key] = value
            name_tokens.append(f"{key}={value}")
        config_key = stable_params_key(params)
        if config_key in seen:
            continue
        seen.add(config_key)
        specs.append({"name": "grid:" + ";".join(name_tokens), "params": params})
    return specs


def build_random_specs(
    center: dict[str, Any],
    search_space: dict[str, list[Any]],
    include_baseline: bool,
    budget: int,
    search_seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(search_seed)
    keys = list(search_space.keys())
    total_combos = count_total_combinations(search_space)
    if budget >= total_combos:
        return build_grid_specs(center, search_space, include_baseline)

    seen: set[str] = set()
    specs: list[dict[str, Any]] = []
    if include_baseline:
        baseline_key = stable_params_key(center)
        seen.add(baseline_key)
        specs.append({"name": "baseline", "params": dict(center)})

    target = budget + (1 if include_baseline else 0)
    max_attempts = max(100, budget * 20)
    attempts = 0
    while len(specs) < target and attempts < max_attempts:
        attempts += 1
        params = dict(center)
        name_tokens: list[str] = []
        for key in keys:
            value = rng.choice(search_space[key])
            params[key] = value
            name_tokens.append(f"{key}={value}")
        config_key = stable_params_key(params)
        if config_key in seen:
            continue
        seen.add(config_key)
        specs.append({"name": "random:" + ";".join(name_tokens), "params": params})

    if len(specs) < target:
        raise RuntimeError(
            f"could not sample {target} unique configs from search space after {attempts} attempts; "
            f"space has {total_combos} total combinations"
        )
    return specs


def build_specs(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, list[Any]]]:
    defaults = get_default_param_values()
    fixed = parse_assignment_map(args.fixed)
    center = dict(defaults)
    center.update(fixed)
    search_space = merge_search_space(args, fixed)
    if args.search_mode == "grid":
        specs = build_grid_specs(center, search_space, args.include_baseline)
    else:
        specs = build_random_specs(center, search_space, args.include_baseline, args.budget, args.search_seed)
    return specs, center, search_space


def closeness_score(value: float, target: float, tolerance: float) -> float:
    if tolerance <= 1e-8:
        return 1.0 if math.isclose(value, target) else 0.0
    return max(0.0, 1.0 - abs(value - target) / tolerance)


def summarize_candidate(
    config_name: str,
    params: dict[str, Any],
    sample_df: pd.DataFrame,
    sample_summary: dict[str, Any],
    ignition_summary: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    max_ticks = float(params.get("max_ticks", EmoNetConfig().max_ticks))
    len1_ratio = float((sample_df["dominant_branch_len"] == 1).mean()) if not sample_df.empty else 0.0
    mean_branch_len = float(sample_summary["mean_branch_len"])
    hit_max_ticks_ratio = float(sample_summary["hit_max_ticks_ratio"])
    mean_first_active_tick = float(ignition_summary["mean_first_active_tick"] or max_ticks)
    late_ignition_ratio = float(ignition_summary["late_ignition_ratio_ge_15"])
    mean_active_window_ticks = float(ignition_summary["mean_active_window_ticks"])
    branch_ratio = mean_branch_len / max_ticks if max_ticks > 0 else 0.0
    active_window_ratio = mean_active_window_ticks / max_ticks if max_ticks > 0 else 0.0

    len1_component = (1.0 - len1_ratio) * 30.0
    max_ticks_component = (1.0 - hit_max_ticks_ratio) * 30.0
    late_component = (1.0 - late_ignition_ratio) * 15.0
    branch_component = closeness_score(
        branch_ratio,
        args.target_branch_ratio,
        args.target_branch_tolerance,
    ) * 10.0
    first_active_component = closeness_score(
        mean_first_active_tick,
        args.target_first_active_tick,
        args.target_first_active_tolerance,
    ) * 10.0
    active_window_component = closeness_score(
        active_window_ratio,
        args.target_active_window_ratio,
        args.target_active_window_tolerance,
    ) * 10.0
    balanced_score = round(
        len1_component
        + max_ticks_component
        + late_component
        + branch_component
        + first_active_component
        + active_window_component,
        4,
    )

    return {
        "config_name": config_name,
        "params_json": stable_params_key(params),
        "rows": int(sample_summary["rows"]),
        "max_ticks": int(max_ticks),
        "mean_branch_len": mean_branch_len,
        "p95_branch_len": float(sample_summary["p95_branch_len"]),
        "mean_ticks_run": float(sample_summary["mean_ticks_run"]),
        "hit_max_ticks_ratio": hit_max_ticks_ratio,
        "mean_path_coverage": float(sample_summary["mean_path_coverage"]),
        "mean_silent_tail_ticks": float(sample_summary["mean_silent_tail_ticks"]),
        "len1_ratio": len1_ratio,
        "mean_first_active_tick": mean_first_active_tick,
        "late_ignition_ratio_ge_15": late_ignition_ratio,
        "mean_active_window_ticks": mean_active_window_ticks,
        "branch_ratio": branch_ratio,
        "active_window_ratio": active_window_ratio,
        "branch_target_delta": abs(branch_ratio - args.target_branch_ratio),
        "first_active_target_delta": abs(mean_first_active_tick - args.target_first_active_tick),
        "active_window_target_delta": abs(active_window_ratio - args.target_active_window_ratio),
        "score_len1_component": round(len1_component, 4),
        "score_hit_max_component": round(max_ticks_component, 4),
        "score_late_component": round(late_component, 4),
        "score_branch_component": round(branch_component, 4),
        "score_first_active_component": round(first_active_component, 4),
        "score_active_window_component": round(active_window_component, 4),
        "balanced_score": balanced_score,
    }


def is_dominated(candidate: pd.Series, other: pd.Series, objective_columns: list[str]) -> bool:
    all_not_worse = True
    strictly_better = False
    for column in objective_columns:
        other_value = float(other[column])
        candidate_value = float(candidate[column])
        if other_value > candidate_value:
            all_not_worse = False
            break
        if other_value < candidate_value:
            strictly_better = True
    return all_not_worse and strictly_better


def mark_pareto_front(summary_df: pd.DataFrame) -> pd.DataFrame:
    objective_columns = [
        "len1_ratio",
        "hit_max_ticks_ratio",
        "late_ignition_ratio_ge_15",
        "branch_target_delta",
        "first_active_target_delta",
        "active_window_target_delta",
    ]
    flags: list[bool] = []
    for idx, row in summary_df.iterrows():
        dominated = False
        for other_idx, other_row in summary_df.iterrows():
            if idx == other_idx:
                continue
            if is_dominated(row, other_row, objective_columns):
                dominated = True
                break
        flags.append(not dominated)
    tagged = summary_df.copy()
    tagged["is_pareto_front"] = flags
    return tagged


def render_score_figure(summary_df: pd.DataFrame, output_path: Path, top_k: int) -> None:
    top_df = summary_df.head(top_k).copy()
    longest_label = max((len(str(label)) for label in top_df["config_name"].tolist()), default=20)
    fig_width = max(10.0, 8.0 + 0.08 * longest_label)
    fig_height = max(4.5, 0.45 * len(top_df))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    colors = ["#0f766e" if bool(flag) else "#334155" for flag in top_df["is_pareto_front"].tolist()]
    ax.barh(top_df["config_name"], top_df["balanced_score"], color=colors)
    ax.invert_yaxis()
    ax.set_title("Branch Dynamics Optimizer: Balanced Score")
    ax.set_xlabel("balanced score")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="svg")
    plt.close(fig)


def render_tradeoff_figure(summary_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.5))
    scatter = ax.scatter(
        summary_df["len1_ratio"],
        summary_df["hit_max_ticks_ratio"],
        c=summary_df["balanced_score"],
        cmap="viridis",
        s=70,
        alpha=0.85,
        edgecolors="black",
        linewidths=0.3,
    )
    front_df = summary_df[summary_df["is_pareto_front"]]
    ax.scatter(
        front_df["len1_ratio"],
        front_df["hit_max_ticks_ratio"],
        facecolors="none",
        edgecolors="#dc2626",
        s=180,
        linewidths=1.4,
        label="pareto front",
    )
    ax.set_xlabel("len1_ratio")
    ax.set_ylabel("hit_max_ticks_ratio")
    ax.set_title("Length-1 vs Max-Ticks Tradeoff")
    ax.legend(loc="upper right")
    fig.colorbar(scatter, ax=ax, label="balanced score")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="svg")
    plt.close(fig)


def render_activation_figure(summary_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.5))
    point_sizes = 40.0 + 4.0 * summary_df["mean_branch_len"].astype(float)
    scatter = ax.scatter(
        summary_df["mean_first_active_tick"],
        summary_df["active_window_ratio"],
        c=summary_df["balanced_score"],
        cmap="plasma",
        s=point_sizes,
        alpha=0.85,
        edgecolors="black",
        linewidths=0.3,
    )
    ax.set_xlabel("mean_first_active_tick")
    ax.set_ylabel("active_window_ratio")
    ax.set_title("Ignition vs Sustained Activity")
    fig.colorbar(scatter, ax=ax, label="balanced score")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="svg")
    plt.close(fig)


def render_top_metric_figure(summary_df: pd.DataFrame, output_path: Path, top_k: int) -> None:
    top_df = summary_df.head(top_k).copy()
    longest_label = max((len(str(label)) for label in top_df["config_name"].tolist()), default=20)
    fig_width = max(15.0, 12.0 + 0.10 * longest_label)
    fig_height = max(4.2, 0.35 * len(top_df))
    fig, axes = plt.subplots(1, 3, figsize=(fig_width, fig_height), sharey=True)

    axes[0].barh(top_df["config_name"], top_df["mean_branch_len"], color="#2563eb")
    axes[0].invert_yaxis()
    axes[0].set_title("Mean Branch Length")

    axes[1].barh(top_df["config_name"], top_df["mean_first_active_tick"], color="#ea580c")
    axes[1].set_title("Mean First Active Tick")

    axes[2].barh(top_df["config_name"], top_df["active_window_ratio"], color="#16a34a")
    axes[2].set_title("Active Window Ratio")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="svg")
    plt.close(fig)


def write_report(
    output_path: Path,
    *,
    args: argparse.Namespace,
    search_space: dict[str, list[Any]],
    center: dict[str, Any],
    summary_df: pd.DataFrame,
    figure_paths: list[Path],
) -> None:
    top_rows = summary_df.head(min(5, len(summary_df)))
    baseline_df = summary_df.loc[summary_df["config_name"] == "baseline"]
    lines = [
        "# Branch Dynamics Optimization Report",
        "",
        f"- search_mode: `{args.search_mode}`",
        f"- preset: `{args.preset}`",
        f"- sample_size: `{args.sample_size}`",
        f"- sample_mode: `{args.sample_mode}`",
        f"- sample_seed: `{args.seed}`",
        f"- model_seed: `{args.model_seed}`",
        f"- num_workers: `{args.num_workers}`",
        "",
        "## Objective",
        "",
        "Balanced score rewards low `len1_ratio`, low `hit_max_ticks_ratio`, low late ignition, and closeness to the configured branch/activation targets.",
        "",
        f"- target_branch_ratio: `{args.target_branch_ratio}`",
        f"- target_first_active_tick: `{args.target_first_active_tick}`",
        f"- target_active_window_ratio: `{args.target_active_window_ratio}`",
        "",
        "## Search Space",
        "",
        "```json",
        json.dumps(search_space, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Fixed Center",
        "",
        "```json",
        json.dumps(center, ensure_ascii=False, indent=2),
        "```",
        "",
    ]
    if not baseline_df.empty:
        baseline = baseline_df.iloc[0]
        lines.extend(
            [
                "## Baseline",
                "",
                f"- balanced_score: `{baseline['balanced_score']:.4f}`",
                f"- mean_branch_len: `{baseline['mean_branch_len']:.4f}`",
                f"- len1_ratio: `{baseline['len1_ratio']:.4f}`",
                f"- hit_max_ticks_ratio: `{baseline['hit_max_ticks_ratio']:.4f}`",
                f"- mean_first_active_tick: `{baseline['mean_first_active_tick']:.4f}`",
                f"- active_window_ratio: `{baseline['active_window_ratio']:.4f}`",
                "",
            ]
        )

    lines.extend(["## Top Candidates", ""])
    for _, row in top_rows.iterrows():
        lines.extend(
            [
                f"### {row['config_name']}",
                "",
                f"- balanced_score: `{row['balanced_score']:.4f}`",
                f"- pareto_front: `{bool(row['is_pareto_front'])}`",
                f"- mean_branch_len: `{row['mean_branch_len']:.4f}`",
                f"- len1_ratio: `{row['len1_ratio']:.4f}`",
                f"- hit_max_ticks_ratio: `{row['hit_max_ticks_ratio']:.4f}`",
                f"- mean_first_active_tick: `{row['mean_first_active_tick']:.4f}`",
                f"- active_window_ratio: `{row['active_window_ratio']:.4f}`",
                f"- params_json: `{row['params_json']}`",
                "",
            ]
        )

    lines.extend(["## Figures", ""])
    for path in figure_paths:
        lines.append(f"- `{path.name}`")
    lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", default=None)
    parser.add_argument("--input-json", default=None)
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--sample-size", type=int, default=60)
    parser.add_argument("--sample-mode", choices=["head", "random"], default="random")
    parser.add_argument("--seed", type=int, default=42, help="sample selection seed")
    parser.add_argument("--model-seed", type=int, default=42, help="graph initialization seed")
    parser.add_argument("--search-seed", type=int, default=42, help="search candidate sampling seed")
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=1, help="0 uses all logical CPU cores")
    parser.add_argument("--dataset-csv", default=None)
    parser.add_argument("--benchmark-csv", default=None)
    parser.add_argument("--model-cache-path", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--force-refit", action="store_true")
    parser.add_argument("--z-dim", type=int, default=64)
    parser.add_argument("--preset", choices=sorted(PRESET_SEARCH_SPACES), default="sticky_reduction")
    parser.add_argument("--search-mode", choices=["random", "grid"], default="random")
    parser.add_argument("--budget", type=int, default=24, help="number of random configs excluding baseline")
    parser.add_argument("--include-baseline", action="store_true")
    parser.add_argument("--space", action="append", default=[], help="override search space with key=v1,v2,...")
    parser.add_argument("--fixed", action="append", default=[], help="fixed key=value override")
    parser.add_argument("--top-k-figures", type=int, default=12)
    parser.add_argument("--target-branch-ratio", type=float, default=0.45)
    parser.add_argument("--target-branch-tolerance", type=float, default=0.35)
    parser.add_argument("--target-first-active-tick", type=float, default=4.0)
    parser.add_argument("--target-first-active-tolerance", type=float, default=12.0)
    parser.add_argument("--target-active-window-ratio", type=float, default=0.45)
    parser.add_argument("--target-active-window-tolerance", type=float, default=0.35)
    parser.add_argument("--output-dir", default=str(Path("outputs") / "branch_optimize" / "latest"))
    args = parser.parse_args()

    input_df = load_input_dataframe(args)
    text_column = resolve_text_column(input_df, args.text_column)
    sampled_df = sample_input_rows(input_df, args.sample_size, args.sample_mode, args.seed)
    texts = sampled_df[text_column].fillna("").astype(str).tolist()
    specs, center, search_space = build_specs(args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sampled_df.to_csv(output_dir / "sampled_inputs.csv", index=False, encoding="utf-8-sig")
    (output_dir / "search_space.json").write_text(json.dumps(search_space, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "center_config.json").write_text(json.dumps(center, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "specs.json").write_text(json.dumps(specs, ensure_ascii=False, indent=2), encoding="utf-8")

    summary_rows: list[dict[str, Any]] = []
    sample_frames: list[pd.DataFrame] = []
    tick_frames: list[pd.DataFrame] = []
    optimize_start = time.perf_counter()
    for idx, spec in enumerate(specs, start=1):
        config_name = str(spec["name"])
        params = dict(spec["params"])
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
        summary_rows.append(summarize_candidate(config_name, params, sample_df, sample_summary, ignition_summary, args))

        sample_df = sample_df.copy()
        sample_df["config_name"] = config_name
        sample_df["params_json"] = stable_params_key(params)
        ignition_df = ignition_df.copy()
        ignition_df["config_name"] = config_name
        tick_df = tick_df.copy()
        tick_df["config_name"] = config_name
        sample_df = sample_df.merge(ignition_df, on=["sample_index", "config_name"], how="left")

        sample_frames.append(sample_df)
        tick_frames.append(tick_df)
        maybe_print_progress(
            "branch-optimize configs",
            idx,
            len(specs),
            optimize_start,
            every=1,
            unit="configs",
            extra=config_name,
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df = mark_pareto_front(summary_df)
    summary_df = summary_df.sort_values(
        by=["balanced_score", "len1_ratio", "hit_max_ticks_ratio", "mean_branch_len"],
        ascending=[False, True, True, False],
    ).reset_index(drop=True)
    detail_df = pd.concat(sample_frames, ignore_index=True) if sample_frames else pd.DataFrame()
    tick_df = pd.concat(tick_frames, ignore_index=True) if tick_frames else pd.DataFrame()

    summary_csv = output_dir / "summary.csv"
    details_csv = output_dir / "details.csv"
    tick_csv = output_dir / "tick_details.csv"
    best_json = output_dir / "best_config.json"
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    detail_df.to_csv(details_csv, index=False, encoding="utf-8-sig")
    tick_df.to_csv(tick_csv, index=False, encoding="utf-8-sig")
    best_payload = summary_df.iloc[0].to_dict() if not summary_df.empty else {}
    best_json.write_text(json.dumps(best_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    figure_paths = [
        output_dir / "optimizer_balanced_score.svg",
        output_dir / "optimizer_len1_vs_hitmax.svg",
        output_dir / "optimizer_activation_tradeoff.svg",
        output_dir / "optimizer_top_metrics.svg",
    ]
    render_score_figure(summary_df, figure_paths[0], args.top_k_figures)
    render_tradeoff_figure(summary_df, figure_paths[1])
    render_activation_figure(summary_df, figure_paths[2])
    render_top_metric_figure(summary_df, figure_paths[3], args.top_k_figures)

    write_report(
        output_dir / "BRANCH_OPTIMIZATION_REPORT.md",
        args=args,
        search_space=search_space,
        center=center,
        summary_df=summary_df,
        figure_paths=figure_paths,
    )

    payload = {
        "input_rows": int(len(input_df)),
        "sample_rows": int(len(sampled_df)),
        "config_rows": int(len(summary_df)),
        "summary_csv": str(summary_csv),
        "details_csv": str(details_csv),
        "tick_csv": str(tick_csv),
        "best_json": str(best_json),
        "figure_paths": [str(path) for path in figure_paths],
        "best_config": best_payload,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
